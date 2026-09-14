#!/usr/bin/env python3
"""Summarise a vLLM torch profiler trace: where the time actually goes.

Reads a decompressed chrome trace on stdin, so drive it as

    zcat trace.json.gz | trace_summary.py --label "decode"

torch pretty-prints the event array, one field per line, and the captures are large (242 MB
decompressed for a 5 s decode capture, over a gigabyte for a 61 s prefill). A line-based scan
for the four fields that matter is both the fastest and the least fragile way through that;
JSON parsing every event, or buffering to find object boundaries, is far slower and buys
nothing here.

It answers the two questions the capture was taken for:

  1. Is the GPU busy, or idling while the host works? GPU-busy is the union of kernel spans
     over wall clock. Low means host-bound and the host work is worth attacking; high means
     compute-bound and no host-side tuning will help.
  2. What does the host do between steps? The staged PLE path does ids.to("cpu") once per
     forward, which is a device synchronisation. A large CPU span with an idle GPU under it
     would make that a structural cost of the staged design rather than of the row source.
"""
import argparse, re, sys
from collections import defaultdict

FIELD = re.compile(r'"(ph|cat|name|ts|dur)":\s*("(?:[^"\\]|\\.)*"|[0-9.eE+-]+)')


def stream_events(fh):
    cur = {}
    for line in fh:
        stripped = line.lstrip()
        if stripped.startswith("{"):
            if cur:
                yield cur
            cur = {}
        for key, raw in FIELD.findall(line):
            if key in cur:
                continue                      # first wins; nested args repeat "name"
            cur[key] = raw[1:-1] if raw.startswith('"') else float(raw)
    if cur:
        yield cur


def union_duration(spans):
    """Wall time covered by at least one span, so overlapping kernels are not double counted."""
    if not spans:
        return 0.0
    spans.sort()
    total = 0.0
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--label", default="trace")
    a = ap.parse_args()

    kernel_time, kernel_calls = defaultdict(float), defaultdict(int)
    cpu_time, cpu_calls = defaultdict(float), defaultdict(int)
    gpu_spans = []
    t_min = t_max = None
    n = 0

    for ev in stream_events(sys.stdin):
        if ev.get("ph") != "X":
            continue
        ts, dur = ev.get("ts"), ev.get("dur")
        if ts is None or dur is None:
            continue
        name = ev.get("name", "?")
        cat = (ev.get("cat") or "").lower()
        n += 1
        t_min = ts if t_min is None else min(t_min, ts)
        t_max = max(t_max or 0.0, ts + dur)
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernel_time[name] += dur
            kernel_calls[name] += 1
            gpu_spans.append((ts, ts + dur))
        elif cat in ("cpu_op", "user_annotation", "cuda_runtime"):
            cpu_time[name] += dur
            cpu_calls[name] += 1

    wall = (t_max - t_min) / 1e6 if t_min is not None else 0.0
    busy = union_duration(gpu_spans) / 1e6
    summed = sum(kernel_time.values()) / 1e6

    print(f"=== {a.label} ===")
    print(f"  timed events         {n:,}")
    print(f"  wall clock           {wall:8.3f} s")
    print(f"  GPU busy (union)     {busy:8.3f} s  = {busy / wall * 100 if wall else 0:5.1f}% of wall")
    print(f"  GPU idle             {wall - busy:8.3f} s  = "
          f"{(wall - busy) / wall * 100 if wall else 0:5.1f}%   <- host-bound if large")
    print(f"  kernel time summed   {summed:8.3f} s  (concurrency {summed / busy if busy else 0:.2f}x)")

    print(f"\n  top GPU kernels")
    for name, t in sorted(kernel_time.items(), key=lambda kv: -kv[1])[:a.top]:
        print(f"    {t / 1e6:7.3f}s {kernel_calls[name]:>7}x {t / (busy * 1e6) * 100 if busy else 0:5.1f}%  {name[:74]}")

    suspect = re.compile(r"ple|gather|cudaMemcpy|Synchronize|item|copy_|to\b|prepare", re.I)
    print(f"\n  top host spans   (* = PLE / copy / sync, the staged-path suspects)")
    for name, t in sorted(cpu_time.items(), key=lambda kv: -kv[1])[:a.top]:
        mark = "*" if suspect.search(name) else " "
        print(f"    {t / 1e6:7.3f}s {cpu_calls[name]:>7}x {mark} {name[:74]}")


if __name__ == "__main__":
    main()
