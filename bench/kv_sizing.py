#!/usr/bin/env python3
"""Size the KV pool and the retention interval from the block accounting, not by feel.

Everything here comes from the vLLM source read on 2026-09-14 plus the startup profile, and
the model is checked against three measured outcomes before it is used to recommend anything.

The block accounting, which is the part that is easy to get wrong:

  A block is 45,260,800 B (43.16 MiB) and holds 3,200 tokens for ONE group. There are five
  groups: one full-attention group and four Mamba groups (3 GDN + 1 PLE conv). So a retained
  prefix costs

      attention:      ceil(C / 3200)                      blocks   (irreducible)
      mamba:          4 * ceil(C / N)                      blocks   (N = retention interval)

  At N = 3200 the interval equals the block size, which short-circuits to dense retention
  (single_type_kv_cache_manager.py:1534-1537), so mamba costs 4 * ceil(C/3200) and a prefix
  is ~5 blocks per 3,200 tokens instead of ~1. That is the bug that was costing 4x capacity.

  A RUNNING request additionally needs live Mamba working state:
      4 groups * (2 + num_speculative_tokens) + 1 scratch = 4 * 5 + 1 = 21 blocks at MTP-3.
  Its attention blocks are the same ones its prefix already owns, so they are not double
  counted.

`kv_cache_size_tokens` in the startup log is NOT the usable retention capacity: it is derived
from the LIVE footprint, where Mamba costs a flat 21 blocks regardless of context length.
Retention is what this calculator sizes.
"""
import argparse, math

BLOCK_BYTES = 45_260_800          # 13 layers x (3,276,800 KV + 204,800 QSA)
BLOCK_TOKENS = 3200
MAMBA_GROUPS = 4
GIB = 1 << 30

# From the startup profile, gpu_worker.py:867
TOTAL_GIB = 121.63
VLLM_NON_KV_GIB = 80.68           # weights+non-torch 78.82 + activation 1.55 + cudagraph 0.31
OS_BASELINE_GIB = 6.57            # 121.63 - 115.06 free on device at startup


def live_overhead(spec_tokens):
    return MAMBA_GROUPS * (2 + spec_tokens) + 1


def retained_blocks(ctx, interval):
    attn = math.ceil(ctx / BLOCK_TOKENS)
    mamba = MAMBA_GROUPS * math.ceil(ctx / interval)
    return attn + mamba


def blocks_needed(sessions, ctx, interval, running, spec_tokens):
    return sessions * retained_blocks(ctx, interval) + running * live_overhead(spec_tokens)


def pool_blocks(kv_gib):
    return int(kv_gib * GIB) // BLOCK_BYTES


def check_model():
    """The model has to reproduce what we actually observed before it gets to advise."""
    cases = [
        # (sessions, ctx, interval, running, observed, note)
        (8, 124056, 3200, 1, "EVICTED", "8 x 124k at interval 3200, pool 664"),
        (12, 20675, 3200, 1, "RETAINED", "12 x 20.7k at interval 3200, pool 664"),
        (8, 124056, 25600, 1, "RETAINED", "8 x 124k at interval 25600, pool 664"),
    ]
    print("model check against measured outcomes (pool = 664 blocks)")
    ok = True
    for sessions, ctx, interval, running, observed, note in cases:
        # +1 for the subject prompt itself, which also has to stay resident
        need = blocks_needed(sessions + 1, ctx, interval, running, 3)
        predicted = "RETAINED" if need <= 664 else "EVICTED"
        mark = "ok  " if predicted == observed else "FAIL"
        ok &= predicted == observed
        print(f"  {mark} {note:<42} needs {need:>5} blocks -> {predicted:<8} (saw {observed})")
    print("  model reproduces all three\n" if ok else "  MODEL IS WRONG, do not trust what follows\n")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=8, help="concurrent agent sessions to keep warm")
    ap.add_argument("--context", type=int, default=125000, help="tokens per session")
    ap.add_argument("--running", type=int, default=8, help="how many run at once, worst case")
    ap.add_argument("--spec-tokens", type=int, default=3)
    ap.add_argument("--os-reserve", type=float, default=8.0,
                    help="GiB left for page cache and headroom, on top of the OS baseline")
    a = ap.parse_args()

    check_model()

    budget = TOTAL_GIB - VLLM_NON_KV_GIB - OS_BASELINE_GIB
    print(f"memory budget")
    print(f"  total                       {TOTAL_GIB:7.2f} GiB")
    print(f"  vLLM weights+activation     {VLLM_NON_KV_GIB:7.2f} GiB")
    print(f"  OS baseline at startup      {OS_BASELINE_GIB:7.2f} GiB")
    print(f"  ------------------------------------")
    print(f"  absolute KV ceiling         {budget:7.2f} GiB   ({pool_blocks(budget)} blocks)")
    print(f"  usable at {a.os_reserve:.0f} GiB reserve       "
          f"{budget - a.os_reserve:7.2f} GiB   ({pool_blocks(budget - a.os_reserve)} blocks)\n")

    print(f"target: {a.sessions} sessions x {a.context:,} tokens, {a.running} running at once\n")
    print(f"  {'interval':>9} {'blocks/session':>15} {'total blocks':>13} {'KV needed':>11} "
          f"{'fits 28 GiB?':>13} {'max ctx @28':>12}")
    for interval in (3200, 6400, 12800, 25600, 51200, 102400, 262144):
        per = retained_blocks(a.context, interval)
        need = blocks_needed(a.sessions, a.context, interval, a.running, a.spec_tokens)
        gib = need * BLOCK_BYTES / GIB
        fits = "yes" if need <= pool_blocks(28.0) else "NO"
        # largest context that still fits 28 GiB at this interval
        lo, hi = 1000, 262144
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if blocks_needed(a.sessions, mid, interval, a.running, a.spec_tokens) <= pool_blocks(28.0):
                lo = mid
            else:
                hi = mid - 1
        print(f"  {interval:>9} {per:>15} {need:>13} {gib:>10.2f}G {fits:>13} {lo:>11,}")

    print(f"\n  (28 GiB = {pool_blocks(28.0)} blocks, which is what is deployed now)")


if __name__ == "__main__":
    main()
