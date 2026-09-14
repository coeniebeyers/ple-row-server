# bench

`bench_remote.py` measures and checks the row servers on their own: no vLLM, no model, no
GPU. It opens the same client the model will use, gathers rows, and either times the call or
proves the bytes are the right bytes.

Measuring before integrating is the point. If the servers are benched only through a running
model, a disappointing number has two suspects and no way to separate them.

## What it needs

- a row server running on each peer, covering the whole table between them
- `../client/ple_remote.py`, which it imports by path (override with `--client`)
- Python 3.10 or later, with `numpy` and `torch`

Torch is not optional. The client imports `numpy` and `torch` at module scope and its gather
entry point is `gather_cpu(ids: torch.Tensor)`, so a torch free bench would have to drive
something other than the client the model uses, which is the one thing worth measuring. The
peers themselves need neither.

`deploy/README.md` covers getting a server onto each node. For a one off, build the binary
with `make -C server` on a box that has a compiler (node2 has none, so it takes the container
image instead) and run:

```sh
rowserverd --file ~/ple/half.bin --base-row 0 --row-count 160000768
```

with `--base-row 160000768` on node2. Port 9000 is the server's default and the one the bench
dials; pass `--peers host:port` if the servers were started on another one.

## Correctness first

Run this before any timing run, and run it on a node:

```sh
bench_remote.py --mode verify
```

`--mode verify` gathers rows through the client and compares every returned byte against the
same rows read straight out of a flat half file. It defaults to `~/ple/half.bin`, which is
where each node keeps its own half.

That file is the source of truth, and it does not need anything from gx10 to be one. Both
halves were exported from the safetensors and checksum verified block by block in Phase 0, so
a row read out of `half.bin` is the row the checkpoint holds. Comparing the server's answer
against it proves the wire format, the server's indexing of its mapping, the ordering
promise, and the client's scatter back into one buffer.

The sampled ids are not just a random draw. They include the first row of the range, the last
row, a row repeated three times, and a descending run, all inside the same request. The
protocol promises rows back in request order with duplicates kept, and a server that sorts or
dedupes internally still looks correct when it is only ever handed sorted unique ids.

### Which rows the file holds

`--verify-base` is the global row id of the first row in the reference file. Left off, the
bench works it out from the peers' own `STAT`: if exactly one peer serves as many rows as the
file holds, that peer's `base_row` is it, and when both halves are the same size the tie goes
to the peer whose address is on this machine. The output line says which and why. If neither
rule settles it, the run stops and prints the candidates rather than guessing, because a
wrong base compares one half's server against the other half's file and reports every single
row as a mismatch.

A base that would put the file outside the rows the peers actually serve is rejected before
anything is sent.

### The check that needs no reference

```sh
bench_remote.py --mode verify --verify-order          # on a box with no half.bin on it
```

`--verify-order` gathers a batch of ids and then gathers each of those ids on its own, and
requires the two to agree. It also sends the same batch twice and requires identical bytes.
Both legs come from the same servers, so this cannot catch a server that indexes its own half
wrongly; that is what the reference file is for. What it does catch is a server that sorts or
dedupes a request internally, a client that scatters the peers' responses back in the wrong
order, and a stale or torn staging buffer. It is included automatically in `--mode verify`,
and the id set spans every peer, so the split boundary is exercised in both directions.

The single row leg is one round trip per id, so it is capped at 256 ids.

### Verifying both halves

One reference file proves one half, because its rows only route to the peer that owns them.
The output says how many sampled ids landed on each peer, which is how you see that. Run
`--mode verify` on node1 and again on node2 to cover the whole table.

## Running it

```sh
# prove the rows are the right rows, on the node that holds the half
bench_remote.py --mode verify

# decode and prefill against the default peers
bench_remote.py

# find where latency starts to bend
bench_remote.py --mode sweep

# machine readable, for committing next to a change
bench_remote.py --mode sweep --json > ../docs/bench-2node.json

# check first, then time, in one run
bench_remote.py --mode decode --verify-against ~/ple/half.bin
```

| flag | meaning |
|---|---|
| `--peers` | comma separated `host[:port]`, default `172.16.0.28,172.16.0.33` on port 9000 |
| `--client` | row client module, default `../client/ple_remote.py` |
| `--mode` | `all` (default), `decode`, `prefill`, `sweep`, `verify` |
| `--rows` | row count per gather, for `--mode decode` or `--mode prefill` |
| `--sweep` | comma separated row counts, replaces the default sweep |
| `--iters` / `--warmup` | timed iterations and untimed ones, warmup excluded from every number |
| `--seed` | id draw seed, so a run can be repeated |
| `--max-rows` | the servers' `--max-rows`, default 131,072; also the largest gather the bench will ask for |
| `--timeout` | socket timeout in seconds once connected, default 10 |
| `--json` | report to stdout, human output to stderr |
| `--verify-against` | flat half file to check returned bytes against, default `~/ple/half.bin` in `--mode verify` |
| `--verify-base` | global row id of the first row in that file, default worked out from `STAT` |
| `--verify-order` | also check a batched gather against one row per request |
| `--verify-samples` | rows checked per verification, default 8192 |

`--verify-against` and `--verify-order` are not ignored outside `--mode verify`: passing
either one runs that check first and then the timings.

## Workloads

| workload | rows per gather | why |
|---|---|---|
| decode | 512 | a steady state decode step at 16 rows per token |
| prefill | 65,536 | a 4096 token prefill chunk at 16 rows per token |
| sweep | 16 to 131,072 | shows where the cost stops being fixed overhead and starts being bytes |

Ids are drawn uniformly across all 320,001,536 rows, which matches how the model's 16 rows
per token actually land: spread across the whole table with no locality to exploit.

The sweep stops at 131,072 because that is the server's `max_rows` for one request, and
131,072 is also the largest gather the bench will ask for even with two peers sharing it.
A uniform draw splits binomially rather than exactly in half, so a request of
`peers x max_rows` rows only fits when the split comes out dead even and is rejected on
almost every draw. The count that cannot overshoot is the one where a single peer taking
every id in the request is still a legal request. Raising `--max-rows` raises both, and has
to match what the servers were started with.

## Reading the output

```
bench host node1  client PLERemoteTable.gather_cpu  seed 1234  warmup 10
peers (STAT as reported by each server)
  172.16.0.28:9000     rows            0 ..  160,000,768  row_bytes 160  resident  23.84 of  23.84 GiB (100.0%, 4 KiB pages)  served 1,234
  172.16.0.33:9000     rows  160,000,768 ..  320,001,536  row_bytes 160  resident  23.84 of  23.84 GiB (100.0%, 4 KiB pages)  served 1,220
  coverage 320,001,536 of 320,001,536 rows

reference /home/coenie/ple/half.bin
  covers global rows 0 .. 160,000,768 (25,600,122,880 bytes, base from 172.16.0.28:9000 is this machine)
  checked 8,192 rows, 0 mismatched
  ids landed on: 172.16.0.28:9000 8,192, 172.16.0.33:9000 0
  the reference only spans one peer's range, so only that peer was checked

order and duplicates (no reference needed)
  256 ids in one request, 17 of them repeats, boundary rows and a descending run included
  batched against one row per request: 0 mismatched
  the same request twice: identical

workload     rows  iters   p50 ms   p90 ms   p99 ms   max ms     MB/s   us/row
decode        512    200    0.412    0.605    1.143    2.011    199.0    0.805
```

The peer banner is there because a bench against the wrong half would otherwise look
perfectly healthy. It prints what each server said about itself, not what was asked of it.
The client refuses to build a table at all unless the peers tile `[0, 320,001,536)` with no
gap and no overlap and all report 160 byte rows, so a wrong half or a missing server stops
the run before this banner with one line saying which peer and which range.

A `note` line about residency means part of that half is currently coming off disk, so those
numbers are about the disk rather than the network. It is a note and not a failure: a server
that has just started has not faulted its whole file in yet.

Percentiles are nearest rank over the raw samples, no interpolation. At the default 50
prefill iterations, `ceil(0.99 * 50)` is the last sample, so p99 and max are the same number;
raise `--iters` when the tail is the question.

`MB/s` is row payload at the p50 latency: `rows * 160 / p50`. It does not count the request
ids going the other way (4 bytes per row) or the 16 byte headers, so it understates link use
by roughly 2.5%. `us/row` is the same p50 divided by the row count, which is the column that
shows the bend: it falls as fixed cost amortises and then flattens onto the wire.

## What is timed, and what is not

The timed call is `gather_cpu`, which is the one a forward pass sits on: partition the ids by
peer, send every request before reading any response, read the rows back and scatter them
into the staging buffer in the caller's order. Drawing the ids happens before the clock
starts. `gather()` adds a copy to the ids' device on top, which on a real run is the host to
device copy the engine would pay anyway, and there is no device here to measure it against.

Warmup iterations are excluded, and they are not only cache warming: the client allocates its
pinned staging buffer on the first gather of a given size.

The client raises if a peer returns fewer rows than it asked for. The bench checks the shape
it got back as well, off the clock, so a short response fails the run rather than posting the
best numbers in the file.

## What these numbers prove, and what they do not

They prove: the round trip cost of a row gather, from the client call to usable bytes, over
the network path that exists on the day of the run, and that the bytes are the right bytes.

They do not prove:

- **Time to first token.** There is no model here. Prefill row cost is one term in TTFT, and
  the projected win stays a projection until a real serve run measures it.
- **Anything about the current disk path.** This bench does not run the mmap table, so it
  cannot produce the comparison on its own. Bench both and compare, or do not claim a delta.
- **That `half.bin` matches the checkpoint.** Verifying against a peer's own `half.bin` proves
  the wire format, the indexing and the ordering. That the file itself holds the checkpoint's
  rows was settled separately, by the Phase 0 checksum gate, and the shard permutation trap in
  the top level README is why that gate is not optional.
- **What gx10 will see.** Run from a laptop, this measures the laptop's path to the peers. Run
  it from the box that will host the client. gx10 is currently two switch hops from the peers;
  when it moves onto the same switch these numbers should improve, mostly at the tail, so
  treat a two hop run as a pessimistic baseline rather than a best case.

## Load, and node1

The bench only sends GATHER and STAT, so it is safe against a live server. It is not free,
though: a sweep at 131,072 rows holds the link near line rate for the length of the run, and
node1 is a live k3s etcd member on that same 2.5 GbE. Keep sweeps short, or run them when the
cluster is quiet.

`--mode verify` is the cheap one. It reads 8,192 rows out of the local file, sends one gather
for them, and then a few hundred single row requests for the order check, which is nothing
next to a sweep.

## Exit status

`0` when everything checked out, `1` otherwise: a verification that mismatched, a peer that
would not answer or reported a split the client refuses, or a command line that does not make
sense. Each prints a line saying which.

A mismatch is not an abort, so `--json` still writes its report with `"ok": false` and the
first few mismatching rows in it. A run that cannot start writes the reason and no report.
The JSON carries the peers dialled, their `STAT`, the seed and the host, so a committed result
says where it came from.
