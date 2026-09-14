#!/usr/bin/env python3
"""Find the GPU idle gaps in a trace and attribute them to whatever the host was doing.

A summary that says "GPU idle 61%" is a symptom, not a cause. This locates the gaps on the
timeline and reports which host spans were open across them, which is the difference between
"the model is slow" and "the model is waiting for this specific thing".

Drive it as

    zcat trace.json.gz | trace_gaps.py --min-gap-ms 1

Attribution is by containment: a host span counts toward a gap if it overlaps it. Spans nest,
so a parent and its children both count; read the output as a tree of suspects rather than a
partition of the time. The innermost frequent name is usually the real answer.
"""
import argparse, re, sys
from collections import defaultdict

FIELD = re.compile(r'"(ph|cat|name|ts|dur)":\s*("(?:[^"\\]|\\.)*"|[0-9.eE+-]+)')


def events(fh):
    cur = {}
    for line in fh:
        if line.lstrip().startswith("{"):
            if cur:
                yield cur
            cur = {}
        for k, raw in FIELD.findall(line):
            if k not in cur:
                cur[k] = raw[1:-1] if raw.startswith('"') else float(raw)
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gap-ms", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=18)
    a = ap.parse_args()
    min_gap = a.min_gap_ms * 1000.0        # trace units are microseconds

    gpu, host = [], []
    for ev in events(sys.stdin):
        if ev.get("ph") != "X":
            continue
        ts, dur = ev.get("ts"), ev.get("dur")
        if ts is None or dur is None:
            continue
        cat = (ev.get("cat") or "").lower()
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append((ts, ts + dur))
        elif cat in ("cpu_op", "user_annotation", "cuda_runtime", "python_function"):
            host.append((ts, ts + dur, ev.get("name", "?")))

    if not gpu:
        print("no GPU spans in this trace")
        return
    gpu.sort()

    # merge GPU spans, then the holes between them are the idle gaps
    merged = []
    cs, ce = gpu[0]
    for s, e in gpu[1:]:
        if s > ce:
            merged.append((cs, ce))
            cs, ce = s, e
        else:
            ce = max(ce, e)
    merged.append((cs, ce))

    gaps = []
    for i in range(1, len(merged)):
        g0, g1 = merged[i - 1][1], merged[i][0]
        if g1 - g0 >= min_gap:
            gaps.append((g0, g1))

    total_gap = sum(e - s for s, e in gaps)
    wall = merged[-1][1] - merged[0][0]
    print(f"  wall (first to last kernel)  {wall / 1e6:8.3f} s")
    print(f"  gaps over {a.min_gap_ms:g} ms            {len(gaps):>8} totalling {total_gap / 1e6:.3f} s "
          f"({total_gap / wall * 100:.1f}% of wall)")
    if not gaps:
        return
    biggest = sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:5]
    print(f"  largest single gaps          "
          f"{', '.join(f'{(e - s) / 1000:.0f}ms' for s, e in biggest)}")

    # attribute: for each host span, how much of it lands inside a gap
    host.sort()
    blame = defaultdict(float)
    hits = defaultdict(int)
    gi = 0
    for hs, he, name in host:
        while gi < len(gaps) and gaps[gi][1] < hs:
            gi += 1
        j = gi
        covered = 0.0
        while j < len(gaps) and gaps[j][0] < he:
            covered += max(0.0, min(he, gaps[j][1]) - max(hs, gaps[j][0]))
            j += 1
        if covered > 0:
            blame[name] += covered
            hits[name] += 1

    print(f"\n  host spans covering GPU-idle time (nested, so read as a tree of suspects)")
    for name, t in sorted(blame.items(), key=lambda kv: -kv[1])[:a.top]:
        print(f"    {t / 1e6:8.3f}s {hits[name]:>7}x  {t / total_gap * 100:5.1f}%  {name[:66]}")


if __name__ == "__main__":
    main()
