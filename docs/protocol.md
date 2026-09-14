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
| 5 | 1 | `op` | `1` = GATHER, `2` = PING, `3` = STAT |
| 6 | 2 | `count` low | see below |
| 8 | 4 | `req_id` | echoed back untouched |
| 12 | 4 | `count` | number of row ids that follow |
| 16 | `4 * count` | `ids` | global row ids, `u32` |

`count` at offset 6 is unused padding kept at zero; the real count is the `u32` at
offset 12. The header is 16 bytes so the id array lands 4 byte aligned.

For `PING` and `STAT`, `count` is 0 and no ids follow.

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

On any non-zero status, `count` is 0 and no row data follows. The connection
stays usable.

`STAT` returns status 0, count 0, and a 32 byte body: `base_row` (u64),
`row_count` (u64), `row_bytes` (u32), `resident_pages` (u32), `served_requests`
(u64). It exists so the client can assert it is talking to the right half and so
a health check can see whether the table actually stayed in memory.

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
it. There is no reconnect logic in the hot path: if a connection breaks the gather
fails loudly rather than silently stalling a forward pass while it retries.

## Limits

`max_rows` defaults to 131,072 per request, which covers a 4096 token prefill
chunk at 16 rows per token with room to spare. A request above the limit is
rejected rather than truncated.
