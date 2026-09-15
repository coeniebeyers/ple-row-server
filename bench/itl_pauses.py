#!/usr/bin/env python3
"""Catch inter-token pauses and say what the engine was doing during them.

A mean hides this completely: a stream can average 19 ms per token and still stall for half a
second when another session's prefill chunk lands in the same engine step. So this records
every inter-chunk gap with a timestamp, and samples the server's counters in a background
thread, then lines the two up.

The question it answers: are the pauses correlated with OTHER sessions prefilling (the
chunked-prefill interference hypothesis), with major faults, or with neither.
"""
import argparse, json, threading, time, urllib.request
from collections import deque


class Sampler(threading.Thread):
    """Poll the metrics endpoint so a pause can be attributed to what the engine was doing."""

    daemon = True

    def __init__(self, base, interval=0.25):
        super().__init__()
        self.base, self.interval = base, interval
        self.samples = deque(maxlen=20000)
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            try:
                text = urllib.request.urlopen(self.base + "/metrics", timeout=5).read().decode()
                got = {}
                for line in text.splitlines():
                    for key in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
                                "vllm:num_requests_running", "vllm:num_requests_waiting"):
                        if line.startswith(key):
                            got[key] = float(line.rsplit(" ", 1)[1])
                self.samples.append((time.time(), got))
            except Exception:
                pass
            time.sleep(self.interval)

    def between(self, t0, t1):
        """Prompt tokens consumed and requests running across a window."""
        pts = [s for s in self.samples if t0 - 0.5 <= s[0] <= t1 + 0.5]
        if len(pts) < 2:
            return None, None
        first, last = pts[0][1], pts[-1][1]
        prompt = last.get("vllm:prompt_tokens_total", 0) - first.get("vllm:prompt_tokens_total", 0)
        running = max(s[1].get("vllm:num_requests_running", 0) for s in pts)
        return prompt, running


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://gx10.lan:8080")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--pause-ms", type=float, default=150.0, help="gap worth reporting")
    a = ap.parse_args()

    sampler = Sampler(a.base_url)
    sampler.start()
    time.sleep(1.0)

    body = {"model": a.model,
            "messages": [{"role": "user", "content":
                          "Write out the numbers from 1 to 250, one per line, nothing else."}],
            "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.base_url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})

    t0 = time.time()
    prev = None
    gaps = []
    completion = 0
    for raw in urllib.request.urlopen(req, timeout=900):
        line = raw.decode().strip()
        if not line.startswith("data: ") or line.endswith("[DONE]"):
            continue
        payload = json.loads(line[6:])
        if payload.get("usage"):
            completion = payload["usage"].get("completion_tokens") or completion
        if not payload.get("choices"):
            continue
        if not payload["choices"][0].get("delta", {}).get("content"):
            continue
        now = time.time()
        if prev is not None:
            gaps.append((prev, now, (now - prev) * 1000.0))
        prev = now
    sampler.stop_flag = True
    wall = time.time() - t0

    if not gaps:
        print("no chunks received")
        return
    ms = sorted(g[2] for g in gaps)
    n = len(ms)
    print(f"  {completion} tokens in {wall:.2f}s, {n + 1} chunks")
    print(f"  inter-chunk gap  p50 {ms[n // 2]:6.1f}  p90 {ms[int(n * .9)]:6.1f}  "
          f"p99 {ms[int(n * .99)]:6.1f}  max {ms[-1]:7.1f} ms")

    pauses = [g for g in gaps if g[2] >= a.pause_ms]
    print(f"\n  pauses over {a.pause_ms:.0f} ms: {len(pauses)} of {n} gaps "
          f"({sum(g[2] for g in pauses) / 1000:.2f}s of {wall:.2f}s wall)")
    if not pauses:
        print("  none caught this run; rerun while other sessions are active")
        return
    print(f"\n  {'at':>7}  {'gap':>8}  {'prompt tok consumed':>20}  {'peak running':>12}")
    for s, e, g in sorted(pauses, key=lambda x: -x[2])[:12]:
        prompt, running = sampler.between(s, e)
        pt = f"{prompt:,.0f}" if prompt is not None else "?"
        rn = f"{running:.0f}" if running is not None else "?"
        print(f"  {s - t0:6.1f}s  {g:7.1f}ms  {pt:>20}  {rn:>12}")
    withp = [g for g in pauses if (sampler.between(g[0], g[1])[0] or 0) > 500]
    print(f"\n  {len(withp)} of {len(pauses)} pauses had another session's prefill running "
          f"(>500 prompt tokens consumed during the gap)")


if __name__ == "__main__":
    main()
