# client

`ple_remote.py` holds `PLERemoteTable`, the gx10 side of the split. It is a drop-in
for the `PLEMmapTable` in
[Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)
(`patch/ops/ple_mmap.py`): same constructor shape, same `gather` / `gather_cpu` /
`read_rows_contiguous` / `describe`, same uint8 `[n, row_bytes]` staging and the same
`float8_e4m3fn` view out of `gather`. The rows come off the network instead of off disk.

Only `torch` and `numpy` are needed. The vLLM import is guarded, so the module can be
driven from a bench script on a box with no vLLM on it.

## Pointing it at peers

```sh
export PLE_REMOTE_PEERS=node1:9000,node2:9000
```

```python
from ple_remote import PLERemoteTable

table = PLERemoteTable()                 # or PLERemoteTable(model_dir, layer_idx)
rows = table.gather_cpu(ids)             # uint8 [n, 160], rows in the order asked for
print(table.describe())
```

The client does **not** know the split. At connect time it sends `STAT` to every peer,
takes each one's `base_row` and `row_count` from the answer, and asserts the ranges tile
`[0, rows_total)` with no gap and no overlap. A gap, an overlap, a peer that did not come
up, or a wrong `rows_total` all fail at construction with the ranges printed. This is the
one check standing between a misconfigured deployment and silently serving the wrong
embeddings, so none of it is a warning.

## Environment

Every name also works with a `QWEN4EXP_PLE_REMOTE_` prefix, to sit alongside the other
`QWEN4EXP_PLE_*` knobs once this is wired into the patch. Constructor keyword arguments
win over the environment.

| variable | default | what it does |
|---|---|---|
| `PLE_REMOTE_PEERS` | `node1:9000,node2:9000` | comma separated `host:port` list, any number of peers |
| `PLE_REMOTE_ROWS` | `320001536` | rows the peers must add up to; the load-time coverage assertion |
| `PLE_REMOTE_MAX_ROWS` | `131072` | per request row cap, has to match the servers |
| `PLE_REMOTE_TIMEOUT` | `10.0` | socket timeout in seconds, once connected |
| `PLE_REMOTE_CONNECT_TIMEOUT` | `5.0` | connect timeout in seconds |

## What a gather does

Ids are partitioned by peer range with numpy masks, **all** requests go out, and only then
are the responses read and scattered back into one staging tensor in the caller's original
order. Sending first is what makes the peers work at the same time rather than one after
the other; on a 512 row decode gather that is the difference between one round trip and
two. Duplicate ids stay duplicates, in request order, as the protocol says.

Buffers are grown on demand and kept, so a steady state gather allocates nothing beyond the
mask and index arrays. The partition runs to completion before any bytes hit a socket: an
out of range id or a request above `max_rows` raises with nothing sent, which leaves the
connections usable. Anything that fails *after* a request has gone out marks the whole table
unusable, because unread responses are still queued on at least one socket and the next
gather would read them as its own.

There is no retry anywhere. A gather sits inside a forward pass, so it fails loudly instead
of stalling the engine while it hopes.

## Self-check

```sh
python3 ple_remote.py
```

Stands up two loopback servers over a small synthetic table and checks ordering, duplicates,
boundary crossing, `out=` handling, the rejected-request cases and all four ways the split
assertion can fire. No deployment and no model needed.

## What it does not do yet

- **No reconnect.** One long-lived connection per peer, opened at construction. If one
  breaks, the table object is finished and the process has to build a new one.
- **No chunking above `max_rows`.** A request that would send more than `max_rows` rows to a
  single peer raises rather than splitting. A 4096 token prefill chunk is 65,536 rows across
  two peers, so the default leaves plenty of room, but a much larger batch would need the
  cap raised on both ends.
- **Not overlapped with compute.** The gather is synchronous. The latency budget says it can
  be, and prefetching would mean predicting ids a step ahead.
- **Not wired into vLLM.** This is the table object only. The `PLE_MODE=remote` selection in
  the patch is not written.
- **`read_rows_contiguous` is honest, not fast.** It works, in `max_rows` sized requests, so
  the resident-slice path does not break. Pulling a whole half through it over 2.5 GbE is
  minutes.
- **No IPv6 literals** in the peer list: `host:port` splits on the last colon.
- **No TLS, no auth.** It assumes a trusted LAN, same as the servers.
- **Nothing measured.** The row servers do not exist yet, so no latency number here has been
  taken against a real one.

`num_shards` and `shard_rows` survive from `PLEMmapTable` but now describe the peer split,
not the checkpoint's 128 file shards. The client only ever addresses global rows; which file
shard a row came from is the server's business.
