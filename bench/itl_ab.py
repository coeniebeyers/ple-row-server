#!/usr/bin/env python3
"""Single-stream inter-token latency and prefill rate, for A/B-ing the two PLE paths.

Context length is deliberately modest and fixed. ITL was measured flat at ~88 ms from
1.4k to 165k tokens, so a long prompt only buys a long wait for the same answer. The
prompt still has to be unique per run or the prefix cache turns TTFT into noise.

Run against an IDLE server. With anything else in flight the engine batches the other
sequences into the same step and the number stops meaning what it says.
"""
import argparse, json, random, statistics, time, urllib.request


def measure(base, model, words, max_tokens, tag, seed):
    random.seed(seed)
    corpus = " ".join(f"w{random.randint(0, 999999)}" for _ in range(words))
    body = {"model": model,
            "messages": [{"role": "user",
                          "content": "Ignore this corpus.\n" + corpus
                                     + "\nCount slowly from 1 to 80, one number per line."}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    prev = None
    gaps = []
    for raw in urllib.request.urlopen(req, timeout=1800):
        line = raw.decode().strip()
        if not line.startswith("data: ") or line.endswith("[DONE]"):
            continue
        chunk = json.loads(line[6:])["choices"][0].get("delta", {}).get("content")
        if not chunk:
            continue
        now = time.time()
        if first is None:
            first = now - t0
        else:
            gaps.append((now - prev) * 1000.0)
        prev = now
    gaps.sort()
    return {"tag": tag, "ttft_s": first, "chunks": len(gaps) + 1,
            "itl_p50": statistics.median(gaps) if gaps else None,
            "itl_p90": gaps[int(len(gaps) * 0.9)] if gaps else None,
            "itl_min": gaps[0] if gaps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://gx10.lan:8080")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--label", required=True, help="which arm this is, e.g. 'staged disk'")
    ap.add_argument("--words", type=int, default=1500, help="~6.9 tokens per synthetic word")
    ap.add_argument("--max-tokens", type=int, default=260)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--json")
    a = ap.parse_args()

    print(f"=== {a.label} ===")
    rows = []
    for i in range(a.runs):
        r = measure(a.base_url, a.model, a.words, a.max_tokens, a.label, 4000 + i)
        rows.append(r)
        print(f"  run {i + 1}  TTFT {r['ttft_s']:6.2f}s   ITL p50 {r['itl_p50']:6.1f}  "
              f"p90 {r['itl_p90']:6.1f}  min {r['itl_min']:6.1f} ms   ({r['chunks']} chunks)")
    p50s = [r["itl_p50"] for r in rows]
    best = min(p50s)
    print(f"  --> ITL p50 median-of-runs {statistics.median(p50s):.1f} ms, best {best:.1f} ms "
          f"= {1000.0 / best:.1f} tok/s single stream")
    if a.json:
        json.dump({"label": a.label, "runs": rows}, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
