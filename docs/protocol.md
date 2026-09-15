# Wire protocol

One request, one response, over TCP. Binary, little-endian throughout, no
negotiation and no optional fields. The whole point of this protocol is to be
boring enough that a decode step can pay for it synchronously.

## Why it looks like this

A decode step gathers roughly 512 rows and an engine step takes 66.9 to 146.3 ms,
so the gather has a budget of about 1 ms before it starts showing up. That rules
out anything with a parse step, a schema, or a round trip to agree on anything.
It also means the request itself has to be small: row ids are `u32` rather than
`u64` because the table has 320,001,536 rows, which fits comfortably, and that
halves request bytes for free.

## Framing

Every message starts with a 16 byte header.

### Request

| offset | size | field | notes |
|---|---|---|---|
| 0 | 4 | `magic` | `0x52454C50` ("PLER" little-endian) |
| 4 | 1 | `version` | `1` |
| 5 | 1 | `op` | `1` = GATHER, `2` = PING, `3` = STAT, `4` = RECHARGE |
| 6 | 2 | `arg` | RECHARGE's slice size in MiB, otherwise zero |
| 8 | 4 | `req_id` | echoed back untouched |
| 12 | 4 | `count` | number of row ids that follow |
| 16 | `4 * count` | `ids` | global row ids, `u32` |

Offset 6 was padding and is still zero for every op but `RECHARGE`, which is the
only one that needs an argument and does not need a row id. The real count is the
`u32` at offset 12. The header is 16 bytes so the id array lands 4 byte aligned.

For `PING`, `STAT` and `RECHARGE`, `count` is 0 and no ids follow.

### Response

| offset | size | field | notes |
|---|---|---|---|
| 0 | 4 | `magic` | `0x50534552` ("RESP" little-endian) |
| 4 | 4 | `req_id` | echoed from the request |
| 8 | 4 | `status` | 0 = OK, see below |
| 12 | 4 | `count` | rows returned; equals request count on success |
| 16 | `160 * count` | `rows` | row data **in request order**, duplicates included |

Status codes:

| value | meaning |
|---|---|
| 0 | OK |
| 1 | bad magic or version |
| 2 | unknown op |
| 3 | a row id fell outside this server's range |
| 4 | count exceeded `max_rows` |
| 5 | internal error |
| 6 | a request argument was out of range |
| 7 | the server is already doing this |

On any non-zero status, `count` is 0 and no body follows, row data or otherwise.
The connection stays usable.

`STAT` returns status 0, count 0, and a 32 byte body: `base_row` (u64),
`row_count` (u64), `row_bytes` (u32), `resident_pages` (u32), `served_requests`
(u64). It exists so the client can assert it is talking to the right half and so
a health check can see whether the table actually stayed in memory.

`RECHARGE` returns status 0, count 0, and a 40 byte body, described in
[recharging the page cache](#recharging-the-page-cache).

## Row ids are global

A server owns a contiguous range `[base_row, base_row + row_count)` and the client
sends **global** ids. The server subtracts `base_row` itself. Sending global ids
means the client never has to know how the table was split in order to build a
request, only which server to send it to, and a misrouted id is caught as status 3
rather than silently returning the wrong row.

With the current 2 node split:

| server | base_row | row_count | bytes |
|---|---|---|---|
| node1 | 0 | 160,000,768 | 25,600,122,880 |
| node2 | 160,000,768 | 160,000,768 | 25,600,122,880 |

## How the client uses it

The client partitions the id array by range, sends **both** requests before
reading **either** response, then scatters the two responses back into one output
buffer using the partition masks. Sending both first is what makes the two servers
work at the same time instead of one after the other, which matters because the
gather is on the critical path.

Duplicate ids are returned as duplicates. Deduplicating would shrink the response
but costs a sort or a hash on the hot path, and the response is already small
enough that it is not worth it. Revisit only if measurement says otherwise.

## Connection handling

`TCP_NODELAY` is required on both ends. Nagle's algorithm would batch the small
request header with the id array and add up to 40 ms, which is 40 engine steps
worth of damage.

Connections are long lived. The client opens one per server at load time and keeps
it for the life of the model.

A connection that breaks is replaced rather than mourned. The client reconnects the
affected server and reissues the gather a small bounded number of times before it
gives up, because the alternative, which is what it did until 2026-09-15, is that a
sub-minute network disturbance kills the engine and costs a nine to thirteen minute
model reload.

The rule the retry has to obey comes out of the framing: **a connection that failed
part way through a response can never be used again**. There is no resynchronisation
point in this protocol, so bytes left unread on a socket would be read as the answer
to whatever is sent next, and since a gather response is nothing but row data that
would deserialise cleanly into embeddings for the wrong ids. So the socket is closed
and a new one opened before anything is reissued, and the reissue carries a fresh
`req_id` that the client checks on the way back.

Server rejections are the other half of the rule. A non-zero status leaves the
connection in sync and means the request or the deployment is wrong, so the client
treats statuses 1 to 4 as fatal and does not retry them. A `req_id` that does not
echo is fatal too, and for a stronger reason: at that point the client does not know
what it is holding, and no further reads would tell it.

## Recharging the page cache

The kernel charges a file page to the cgroup whose task first faults it in. If the
table was warmed from a shell before the daemon started, every page belongs to that
shell's slice, the server's own `memory.low` protects nothing, and the kernel
reclaims the table out from under a server that still reports a healthy residency.
Measured on node2: the container's cgroup held 0 GiB of file cache while
`user.slice` held 30 GiB, residency drifted to 96.5%, and the daemon took 57,799
major faults against node1's 1,097. Gathers were about 10x slower for it.

`RECHARGE` fixes that in place, without a restart and without dropping the
connections the model holds. For each slice of the mapping in turn the server:

1. `madvise(slice, MADV_DONTNEED)`, which zaps its own page table entries so
   nothing maps those pages any more,
2. `posix_fadvise(fd, offset, len, POSIX_FADV_DONTNEED)`, which can now actually
   evict them,
3. touches the slice back in through the mapping, so the pages are faulted by this
   daemon and charged to its cgroup.

Step 1 is the part that is easy to leave out and the reason dropping the cache from
a separate process does nothing: **`posix_fadvise` cannot evict a page that a
running process has mapped**. The daemon maps the whole file, so an external drop
is a no-op for precisely the resident pages that need moving, and it only re-reads
the ones that were already missing.

### Request

`op` is 4, `count` is 0, and `arg` at offset 6 is the slice size in MiB. Zero means
the server's default of 1024 MiB. Above 4096 MiB the request is refused with status
6 rather than clamped, because a slice is how much of the table is out of cache at
once and a client asking for a quarter of a half at a time has misunderstood the
knob.

### Response

Status 0, count 0, and a 40 byte body:

| offset | size | field | notes |
|---|---|---|---|
| 0 | 8 | `pages_before` | resident pages of the whole mapping, before |
| 8 | 8 | `pages_after` | and after |
| 16 | 8 | `pages_recharged` | pages this sweep dropped and faulted back in |
| 24 | 4 | `slices_done` | slices completed |
| 28 | 4 | `slices_total` | slices the mapping is divided into |
| 32 | 4 | `slice_pages` | pages per slice, so a caller that sent 0 sees the default |
| 36 | 4 | `elapsed_ms` | wall time for the whole call |

`pages_before` and `pages_after` come from the same `mincore` walk `STAT` uses, so
the caller can see whether it worked rather than being told that it did.
`slices_done` below `slices_total` means the sweep ran out of time and the rest of
the table was left alone. Running it again starts from the beginning rather than
from where it stopped, which is wasteful but idempotent; the server keeps no
cursor, because a sweep that cannot finish 23.8 GiB in ten minutes is a disk
problem to go and look at rather than something to paper over.

### What it costs, and what it does not break

A sweep re-reads the whole half from disk, 23.8 GiB of it, on a node that is also a
k3s etcd member. It is a deliberate maintenance operation, not something to poll.
Three bounds keep it from being worse than the problem: one slice is out of cache
at a time, only one sweep runs at a time (a second caller gets status 7 rather than
being queued), and a sweep that passes 600 seconds stops where it is and reports
how far it got.

Concurrent `GATHER`s keep working throughout, and no lock is taken. The mapping's
address and length never change, so a gather that races the sweep either finds its
page still mapped or takes a fault that resolves to the same file offset. Either
way it copies the same bytes. What it can pay is one major fault, if the row it
wants happens to be in the slice currently in flight.

That fault is what the slice size buys or spends. Measured on a laptop against a
2 GiB synthetic half with a 512 row gather loop running the whole time: a 1024 MiB
slice swept in 12 s and took the worst gather from 1.8 ms to 112 ms, while a 64 MiB
slice swept in 21 s and held the worst at 26 ms. So a smaller slice costs sweep time
and buys tail latency, and if a sweep has to happen while the model is serving, the
smaller one is the right trade.

`MADV_DONTNEED` is safe here specifically because the mapping is `MAP_SHARED`,
`PROT_READ`, over a file opened `O_RDONLY`. For a shared file mapping the man page
guarantees that a later access repopulates "from the up-to-date contents of the
underlying mapped file", and this process cannot have dirtied a page to begin with,
so there is nothing to lose. The variant that does destroy data is `MADV_DONTNEED`
on a private mapping, where it throws the private copy away.

A sweep takes minutes, and the response only arrives at the end of it, so a client
has to set a read timeout that allows for the 600 second cap. `tools/recharge_cache.py`
is the client for this op.

## Limits

`max_rows` defaults to 131,072 per request, which covers a 4096 token prefill
chunk at 16 rows per token with room to spare. A request above the limit is
rejected rather than truncated.
