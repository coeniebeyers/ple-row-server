#!/usr/bin/env python3
"""
Export a contiguous GLOBAL ROW RANGE of the Qwen3.8-Flash-Next PLE n-gram table.

Torch-free and allocation-light on purpose: gx10 runs at ~106/121 GiB with the model
live, so this must never pull in torch or buffer a shard.

CRITICAL: shards 98/99 and 100/101 are PHYSICALLY SWAPPED in the safetensors file.
Offset arithmetic (data_begin + i*bytes_per_shard) is WRONG for those four and would
silently place ~1.6 GB of rows at the wrong global indices. This script always resolves
each shard's byte range from the header by shard ID, so the permutation is handled for free.

Modes:
  --start N --count M          stream those global rows (raw FP8) to stdout
  --digest --start N --count M per-block sha256 of the same range, read from the safetensors
  --digest-plain PATH          per-block sha256 of a flat exported half.bin (for comparison)
  --info                       parse + sanity-check the header, print geometry, exit
"""
import argparse, hashlib, json, os, struct, sys

ROW_BYTES = 160           # ple_embed_dim / ngram_heads = 2560 / 16  (NOT config head_dim=256)
ROWS_PER_SHARD = 2500012
N_SHARDS = 128
PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
SUFFIX = ".weight"
DEFAULT_FILE = "/home/coenie/models/Qwen3.8-Flash-Next-NVFP4-nvidia/model-fp8-mtp-ple.safetensors"


def shard_map(path):
    """shard_id -> (abs_begin, abs_end). Resolved from the header, never by arithmetic."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    base = 8 + hlen
    out = {}
    for k, v in hdr.items():
        if k.startswith(PREFIX) and k.endswith(SUFFIX):
            i = int(k[len(PREFIX):-len(SUFFIX)])
            b, e = v["data_offsets"]
            out[i] = (base + b, base + e, tuple(v["shape"]), v["dtype"])
    return out, base


def check(shards):
    assert len(shards) == N_SHARDS, f"expected {N_SHARDS} PLE shards, got {len(shards)}"
    for i in range(N_SHARDS):
        assert i in shards, f"missing shard {i}"
        b, e, shape, dt = shards[i]
        assert shape == (ROWS_PER_SHARD, ROW_BYTES), f"shard {i} shape {shape}"
        assert dt in ("F8_E4M3", "F8_E4M3FN"), f"shard {i} dtype {dt}"
        assert e - b == ROWS_PER_SHARD * ROW_BYTES, f"shard {i} byte span {e-b}"


def drop_cache(fd, pos, n):
    """Release the pages we just read.

    gx10 serves the model out of this same file's page cache while sitting near the
    121 GiB ceiling. Streaming 47.7 GB through the cache would evict the live weights
    and put decode into major-fault stalls, so give each block straight back.
    """
    try:
        os.posix_fadvise(fd, pos, n, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def iter_range(fd, shards, start, count, block_bytes, sink, keep_cache=False):
    """Walk global rows [start, start+count) shard by shard, resolving each shard's offset by ID."""
    g, end = start, start + count
    while g < end:
        sh, local = divmod(g, ROWS_PER_SHARD)
        take = min(ROWS_PER_SHARD - local, end - g)
        pos = shards[sh][0] + local * ROW_BYTES
        remaining = take * ROW_BYTES
        while remaining:
            n = min(block_bytes, remaining)
            buf = os.pread(fd, n, pos)
            if len(buf) != n:
                raise IOError(f"short read at {pos}: {len(buf)} != {n}")
            sink(buf)
            if not keep_cache:
                drop_cache(fd, pos, n)
            pos += n
            remaining -= n
        g += take


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--start", type=int)
    ap.add_argument("--count", type=int)
    ap.add_argument("--digest", action="store_true")
    ap.add_argument("--digest-plain")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--block-mib", type=int, default=64)
    ap.add_argument("--keep-cache", action="store_true",
                    help="do NOT fadvise-away the pages we read (default is to drop them)")
    a = ap.parse_args()
    block = a.block_mib * 1024 * 1024

    if a.digest_plain:
        h_all = hashlib.sha256()
        with open(a.digest_plain, "rb") as f:
            idx = 0
            while True:
                buf = f.read(block)
                if not buf:
                    break
                print(f"block {idx:04d} {hashlib.sha256(buf).hexdigest()} {len(buf)}")
                h_all.update(buf)
                idx += 1
        print(f"TOTAL {h_all.hexdigest()}")
        return

    shards, base = shard_map(a.file)
    check(shards)

    if a.info:
        perm = [i for i in range(N_SHARDS)
                if shards[i][0] != shards[0][0] + i * ROWS_PER_SHARD * ROW_BYTES]
        print(f"file            : {a.file} ({os.path.getsize(a.file)} B)")
        print(f"data_begin      : {base}")
        print(f"shards          : {len(shards)} x {shards[0][2]} {shards[0][3]}")
        print(f"row_bytes       : {ROW_BYTES}   rows/shard: {ROWS_PER_SHARD}")
        print(f"rows_total      : {N_SHARDS*ROWS_PER_SHARD}")
        print(f"PLE row bytes   : {N_SHARDS*ROWS_PER_SHARD*ROW_BYTES}")
        print(f"shard0 abs      : {shards[0][0]}   shard127 end abs: {shards[127][1]}")
        print(f"OUT-OF-ORDER    : {perm}   <- must be handled by ID, not arithmetic")
        for i in perm:
            print(f"   shard {i:3d} abs {shards[i][0]}")
        half = (N_SHARDS // 2) * ROWS_PER_SHARD
        print(f"split (shard-aligned at shard {N_SHARDS//2}):")
        print(f"   node1  --start 0 --count {half}   ({half*ROW_BYTES} B)")
        print(f"   node2  --start {half} --count {half}   ({half*ROW_BYTES} B)")
        print(f"   out-of-order shards land on: "
              f"{sorted({'node1' if i < N_SHARDS//2 else 'node2' for i in perm})}")
        return

    if a.start is None or a.count is None:
        ap.error("--start and --count are required (or use --info / --digest-plain)")

    fd = os.open(a.file, os.O_RDONLY)
    try:
        if a.digest:
            h_all = hashlib.sha256()
            state = {"buf": bytearray(), "idx": 0}

            def sink(b):
                state["buf"] += b
                while len(state["buf"]) >= block:
                    chunk = bytes(state["buf"][:block])
                    del state["buf"][:block]
                    print(f"block {state['idx']:04d} {hashlib.sha256(chunk).hexdigest()} {len(chunk)}")
                    h_all.update(chunk)
                    state["idx"] += 1

            iter_range(fd, shards, a.start, a.count, block, sink, a.keep_cache)
            if state["buf"]:
                chunk = bytes(state["buf"])
                print(f"block {state['idx']:04d} {hashlib.sha256(chunk).hexdigest()} {len(chunk)}")
                h_all.update(chunk)
            print(f"TOTAL {h_all.hexdigest()}")
        else:
            w = sys.stdout.buffer.write
            iter_range(fd, shards, a.start, a.count, block, w, a.keep_cache)
            sys.stdout.buffer.flush()
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
