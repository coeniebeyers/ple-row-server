#!/usr/bin/env python3
"""What fills the single largest GPU-idle gap, and how much PLE-ish work is in the trace.

Must be fed on stdin by a pipe. Do not run it from a heredoc: the heredoc becomes stdin and
silently replaces the trace, which produces a confident answer about nothing.
"""
import re, sys
from collections import defaultdict

FIELD = re.compile(r'"(ph|cat|name|ts|dur)":\s*("(?:[^"\\]|\\.)*"|[0-9.eE+-]+)')

gpu, host = [], []
ple = defaultdict(float)
ple_n = defaultdict(int)
cur = {}


def take(ev):
    if ev.get("ph") != "X":
        return
    ts, dur = ev.get("ts"), ev.get("dur")
    if ts is None or dur is None:
        return
    cat = (ev.get("cat") or "").lower()
    name = ev.get("name", "?")
    low = name.lower()
    if any(k in low for k in ("ple", "ngram", "stage_rows", "gather")):
        ple[name] += dur
        ple_n[name] += 1
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        gpu.append((ts, ts + dur))
    elif cat in ("cpu_op", "user_annotation", "cuda_runtime", "python_function"):
        host.append((ts, ts + dur, name))


for line in sys.stdin:
    if line.lstrip().startswith("{"):
        if cur:
            take(cur)
        cur = {}
    for k, raw in FIELD.findall(line):
        if k not in cur:
            cur[k] = raw[1:-1] if raw.startswith('"') else float(raw)
if cur:
    take(cur)

print(f"  parsed: {len(gpu):,} gpu spans, {len(host):,} host spans")
if not gpu:
    sys.exit("  no GPU spans parsed, check the pipe")

gpu.sort()
merged = []
cs, ce = gpu[0]
for s, e in gpu[1:]:
    if s > ce:
        merged.append((cs, ce))
        cs, ce = s, e
    else:
        ce = max(ce, e)
merged.append((cs, ce))
wall0 = merged[0][0]

gaps = [(merged[i - 1][1], merged[i][0]) for i in range(1, len(merged))]
gaps = [g for g in gaps if g[1] - g[0] > 500]
if not gaps:
    sys.exit("  no gaps over 0.5 ms")

g0, g1 = max(gaps, key=lambda g: g[1] - g[0])
print(f"  biggest gap: {(g1 - g0) / 1e6:.2f}s, starting {(g0 - wall0) / 1e6:.2f}s into the trace")
print(f"  (trace spans {(merged[-1][1] - wall0) / 1e6:.2f}s)\n")

# only spans that start inside the gap and do not straddle the whole thing
inside = [(s, e, n) for s, e, n in host if g0 <= s < g1 and (e - s) < (g1 - g0) * 0.98]
agg, cnt = defaultdict(float), defaultdict(int)
for s, e, n in inside:
    agg[n] += min(e, g1) - s
    cnt[n] += 1
print(f"  host spans starting inside the gap ({len(inside):,} of them):")
for n, t in sorted(agg.items(), key=lambda kv: -kv[1])[:16]:
    print(f"    {t / 1e6:8.3f}s {cnt[n]:>7}x  {n[:68]}")

print(f"\n  spans matching ple / ngram / stage_rows / gather anywhere in the trace:")
if not ple:
    print("    none (the staged gather runs in prepare_inputs, outside the profiled region)")
for n, t in sorted(ple.items(), key=lambda kv: -kv[1])[:10]:
    print(f"    {t / 1e6:8.3f}s {ple_n[n]:>7}x  {n[:68]}")
