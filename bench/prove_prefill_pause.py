#!/usr/bin/env python3
"""Does another session's prefill cause the pauses? Cause one and watch.

Streams a steady generation, and partway through fires a large prompt from another thread.
If the pauses are chunked-prefill interference, gaps should stay small before the interferer
starts and jump to roughly max_num_batched_tokens / prefill_rate while it runs.

At 4096 batched tokens and ~3,900 tok/s of prefill that predicts ~1.05 s gaps, which is the
number to check against.
"""
import json, sys, threading, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
MODEL = "qwen3.8-flash-next"
gaps = []
interferer_window = [None, None]


def post_chat(body, timeout=900):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def interferer():
    import random
    random.seed(4242)
    corpus = " ".join(f"w{random.randint(0, 999999)}" for _ in range(17000))
    time.sleep(12.0)                      # let the baseline settle first
    interferer_window[0] = time.time()
    try:
        json.load(post_chat({"model": MODEL,
                             "messages": [{"role": "user", "content": corpus + "\nSay OK"}],
                             "max_tokens": 16, "temperature": 0,
                             "chat_template_kwargs": {"enable_thinking": False}}))
    except Exception as exc:
        print(f"  interferer failed: {exc}")
    interferer_window[1] = time.time()


threading.Thread(target=interferer, daemon=True).start()

body = {"model": MODEL,
        "messages": [{"role": "user", "content":
                      "Write the numbers 1 to 600, one per line, nothing else."}],
        "max_tokens": 2400, "temperature": 0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}}
t0 = time.time()
prev = None
for raw in post_chat(body):
    line = raw.decode().strip()
    if not line.startswith("data: ") or line.endswith("[DONE]"):
        continue
    p = json.loads(line[6:])
    if not p.get("choices") or not p["choices"][0].get("delta", {}).get("content"):
        continue
    now = time.time()
    if prev is not None:
        gaps.append((now - t0, (now - prev) * 1000.0))
    prev = now
    if interferer_window[1] and now - t0 > (interferer_window[1] - t0) + 8:
        break

s, e = interferer_window
if not s:
    sys.exit("  interferer never started")
s, e = s - t0, (e or time.time()) - t0
before = [g for at, g in gaps if at < s]
during = [g for at, g in gaps if s <= at <= e]
after = [g for at, g in gaps if at > e]


def describe(label, xs):
    if not xs:
        print(f"  {label:<28} (no samples)")
        return
    xs = sorted(xs)
    n = len(xs)
    over = len([x for x in xs if x >= 400])
    print(f"  {label:<28} n={n:>4}  p50 {xs[n//2]:7.1f}  p90 {xs[int(n*.9)]:7.1f}  "
          f"max {xs[-1]:8.1f} ms   pauses>=400ms: {over}")


print(f"  interferer prefill ran from {s:.1f}s to {e:.1f}s ({e - s:.1f}s)\n")
describe("BEFORE the interferer", before)
describe("DURING its prefill", during)
describe("AFTER it finished", after)

if before and during:
    b = sorted(before)[len(before) // 2]
    d = sorted(during)[len(during) // 2]
    print(f"\n  median gap went {b:.0f} ms -> {d:.0f} ms while it prefilled ({d / b:.1f}x)")
    print("  predicted from max_num_batched_tokens 4096 / prefill ~3900 tok/s: ~1050 ms")
