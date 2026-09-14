# ple-row-server

Serve the Qwen3.8-Flash-Next per-layer embedding (PLE) n-gram table out of a peer
machine's RAM, so the box running the model does not have to hold it.

**Status: spike in progress.** Phase 0 (export and verify the table halves) is done.
The row server itself is not built yet. Numbers below are from a design study plus
Phase 0 measurements, and are marked as such. Nothing here is a finished result.

## The problem

Qwen3.8-Flash-Next carries an n-gram embedding table that is much larger than the
rest of the model's activations budget. On the checkpoint we run it is:

```
128 shards x (2500012, 160) F8_E4M3   =  51,200,245,760 bytes  (47.684 GiB)
```

That is about 160 bytes per row and 320,001,536 rows. Each token gathers 16 rows
from it, spread evenly across the table, so there is no useful locality to exploit
and no small hot subset to cache.

On an ASUS Ascent GX10 (NVIDIA GB10, 128 GiB of memory shared between CPU and GPU)
the weights are roughly 79 GiB. Weights plus the table is more than the box has, so
the table cannot simply live in memory alongside them.

The existing workaround, which this project builds on rather than replaces, is to
leave the table on disk and read the rows needed for each forward pass. That works
and is what we run daily. Its cost is that the reads land in the kernel's page
cache, competing with the memory-mapped model weights for the same limited pool.
When the cache is squeezed, decode stalls on major page faults.

## The idea

Two peer machines each hold half the table in their own RAM and answer row-gather
requests over the network. The inference box keeps its memory for weights and KV
cache.

```
        gx10 (GB10, 128 GiB)                node1            node2
   +--------------------------+        +-----------+    +-----------+
   |  weights ~79 GiB         |        | rows      |    | rows      |
   |  KV cache                |  <---> | 0 .. 63   |    | 64 .. 127 |
   |  no 47.7 GiB table       |        | 23.8 GiB  |    | 23.8 GiB  |
   +--------------------------+        +-----------+    +-----------+
```

Splitting on a shard boundary (64 shards each) keeps each half flat and
directly addressable, and because each token's 16 rows are spread evenly across
the whole table, the load balances itself without any routing logic.

## What this is worth, honestly

The payoff is **not** faster decode. A design study put decode at roughly +2%,
which is noise. The two real wins are:

1. **Prefill / time-to-first-token**, estimated around -21%. Gathering a prefill
   chunk's rows over the network was measured at about 18.1 ms against about 7.08 ms
   per decode step from disk, and the disk path's cost turns out to be syscall count
   rather than I/O: it issues hundreds of small `preadv` calls per step. Network
   gather beat even a *warm* page cache (about 4.7 ms) for that reason.
2. **Giving the page cache back.** This is the one that motivated the work. It
   removes the table from competition with the weights entirely.

Measured gather latency in the study was about 0.45 ms median and 1.17 ms at the
99th percentile for an 8-sequence decode step, against an engine step of 66.9 to
146.3 ms. That is 0.3% to 1.8% of a step, which is why it can be done synchronously
without prefetching or hiding it behind compute.

Those network numbers were taken with the inference box two switch hops from the
peers. Moving it onto the same switch should improve them, mostly at the tail, so
treat them as a pessimistic baseline rather than a best case.

## Phase 0: exporting the table

`tools/ple_export_half.py` streams a contiguous global row range straight out of the
safetensors file. It is deliberately free of any torch dependency and never buffers a
whole shard, because it has to run on a box that is already close to its memory
ceiling with a live model on it.

### The shard permutation trap

Shards 98/99 and 100/101 are **physically swapped** inside the safetensors file.
Their byte offsets do not follow shard order.

```
shard  98 at 40000608370      shard 100 at 39200604530
shard  99 at 40400610290      shard 101 at 39600606450
```

Any exporter that computes a shard's position as `data_begin + shard * bytes_per_shard`
will silently place about 1.6 GB of rows at the wrong indices. The file is still the
right size, the model still loads, and it just gathers the wrong embeddings for part
of the vocabulary. Nothing raises.

This tool always resolves each shard's byte range from the safetensors header by
shard ID, so the permutation costs nothing to handle. Verified directly: reading
shard 100's rows by ID matches a `dd` from its known absolute offset, while naive
arithmetic produces a completely different digest.

For what it is worth, vLLM's own `PLEMmapTable` also resolves by header, so the
running model is not affected. The trap is only for new tooling written against
this checkpoint.

### Usage

```sh
# geometry, permutation report and the shard-aligned split points
ple_export_half.py --info

# stream a row range (raw FP8) to stdout
ple_export_half.py --start 0 --count 160000768 | nc -N <peer> 9000

# per-block sha256 of the same range, read from the safetensors
ple_export_half.py --digest --start 0 --count 160000768

# per-block sha256 of an exported flat file, to compare against the above
ple_export_half.py --digest-plain /path/to/half.bin
```

By default the exporter hands every block it reads back to the kernel with
`posix_fadvise(DONTNEED)`. Without that, streaming 47.7 GB through the page cache
evicts the live model's mapped weights, which is the exact failure this project
exists to avoid.

## Phase 0 results

Both halves exported and size-verified:

| | rows | bytes | time | rate |
|---|---|---|---|---|
| node1 | 0 .. 160,000,768 | 25,600,122,880 | 146 s | 175 MB/s |
| node2 | 160,000,768 .. 320,001,536 | 25,600,122,880 | 151 s | 169 MB/s |

The peers are on 2.5 GbE, so 175 MB/s is about 62% of the wire. During the export
the source NVMe sat at about 10% utilisation with 0.06 ms read latency, so the disk
was never the constraint, and the live model kept serving throughout.

All four out-of-order shards fall in node2's half, which makes node2's checksum the
meaningful test of the permutation handling.

## Layout

```
tools/ple_export_half.py    export and digest a global row range
tools/transfer_halves.sh    stream both halves to the peers
tools/validate_halves.sh    sha256 gate, source against peer, per 64 MiB block
```

## Related

The disk-backed PLE table this builds on comes from
[tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)
(Apache-2.0), where the staged and mmap table modes live. The intended shape of the
client side is a `PLE_MODE=remote` sibling to that repo's existing
`staged` / `mmap` / `offload` modes.

## Licence

Apache-2.0, matching vLLM and the repo above.
