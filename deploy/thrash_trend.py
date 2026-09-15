#!/usr/bin/env python3
"""Is memory pressure building? Rates, not absolutes.

Cumulative counters since container start say nothing on their own: a large pgmajfault is
expected after loading 79 GiB of weights. What matters is whether the counters climb while
the box is only serving, which is the signature the thrash question needs answering with.
"""
import csv, datetime, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/coenie/ple-thrash.csv"
rows = [r for r in csv.DictReader(open(path)) if r.get("pgmajfault")]

# Counters reset when the container restarts, and uptime_s dropping is how that shows up.
# Comparing across a restart produces negative "growth", which is how this bug announced
# itself the first time. Only the current run tells us anything about steady-state pressure.
def up(r):
    try:
        return float(r.get("uptime_s") or 0)
    except ValueError:
        return 0.0

cut = 0
for i in range(1, len(rows)):
    if up(rows[i]) < up(rows[i - 1]):
        cut = i
if cut:
    print(f"  (container restarted at sample {cut} of {len(rows)}; "
          f"analysing only the {len(rows) - cut} samples since)")
rows = rows[cut:]

# Loading reads 123 GiB of safetensors while allocating 79 GiB of weights, which generates
# enormous fault and reclaim counts that say nothing about how the box behaves while serving.
# Skip past it or the verdict is just measuring the startup.
skip_s = float(sys.argv[2]) * 60 if len(sys.argv) > 2 else 3600.0
warm = [r for r in rows if up(r) >= skip_s]
if len(warm) >= 2:
    print(f"  (skipping the first {skip_s/60:.0f} min of load; {len(warm)} of {len(rows)} samples are steady-state)")
    rows = warm
if len(rows) < 2:
    sys.exit("  not enough steady-state samples yet")


def when(r):
    return datetime.datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%SZ")


def num(r, k, default=0.0):
    try:
        return float(r.get(k) or default)
    except ValueError:
        return default


first, last = rows[0], rows[-1]
hours = (when(last) - when(first)).total_seconds() / 3600 or 1e-9
print(f"  window {hours:.2f} h over {len(rows)} samples")


def line(label, key, unit=""):
    a, b = num(first, key), num(last, key)
    print(f"  {label:<18} {a:>12,.0f} -> {b:>12,.0f}   {b - a:+12,.0f}  {(b - a) / hours:>10,.0f}/h{unit}")


line("pgmajfault", "pgmajfault")
line("refault_file", "refault_file")
line("pgscan", "pgscan")
line("pgsteal", "pgsteal")
line("vllm swap kB", "vllm_swap_kb")
print(f"  {'swap GB':<18} {num(first,'swap_used_gb'):>12,.0f} -> {num(last,'swap_used_gb'):>12,.0f}")
print(f"  {'mem free GB':<18} {num(first,'mem_free_gb'):>12,.0f} -> {num(last,'mem_free_gb'):>12,.0f}")
print(f"  {'mem cache GB':<18} {num(first,'mem_cache_gb'):>12,.0f} -> {num(last,'mem_cache_gb'):>12,.0f}")

t0, c0 = num(first, "ttft_sum"), num(first, "ttft_count")
t1, c1 = num(last, "ttft_sum"), num(last, "ttft_count")
if c1 > c0:
    print(f"\n  TTFT across the window: {(t1 - t0) / (c1 - c0):.1f} s mean over {int(c1 - c0)} requests")
g0, d0 = num(first, "gen_tokens"), num(first, "decode_s")
g1, d1 = num(last, "gen_tokens"), num(last, "decode_s")
if g1 > g0 and d1 > d0:
    print(f"  decode across the window: {(d1 - d0) / (g1 - g0) * 1000:.1f} ms/token")

h0, q0 = num(first, "prefix_hits"), num(first, "prefix_queries")
h1, q1 = num(last, "prefix_hits"), num(last, "prefix_queries")
if q1 > q0:
    print(f"  prefix cache across the window: {(h1 - h0) / (q1 - q0) * 100:.1f}% "
          f"({int(h1 - h0):,} / {int(q1 - q0):,})")

# the actual verdict
maj_rate = (num(last, "pgmajfault") - num(first, "pgmajfault")) / hours
swap_delta = num(last, "vllm_swap_kb") - num(first, "vllm_swap_kb")
print()
if maj_rate < 1000 and swap_delta <= 0:
    print("  VERDICT: no thrash. Faults are flat and the engine's heap is not growing in swap.")
elif maj_rate < 10000:
    print(f"  VERDICT: mild fault activity ({maj_rate:,.0f}/h). Watch it; not yet a problem.")
else:
    print(f"  VERDICT: faults climbing at {maj_rate:,.0f}/h. Consider dropping KV to 24 GiB.")
