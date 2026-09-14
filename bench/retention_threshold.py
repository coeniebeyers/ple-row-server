#!/usr/bin/env python3
"""How many other requests does it take to lose a cached prefix?

The earlier probe showed a 124k prompt evicted by ~992k tokens of other work, at both a
1.02M and a 1.69M pool. That rules out simple LRU-by-capacity but does not say what the
real limit is. This narrows it by holding the token volume LOW and varying the COUNT.

Prompts here are ~20k tokens, so filling a 1.69M pool by volume would take ~84 of them.
If the subject dies after a handful, the binding constraint is a number of somethings
(sequence slots, retained state entries) rather than bytes, and buying KV cannot fix it.

Run against an idle server.
"""
import argparse, json, random, time, urllib.request


def counters(base):
    text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = {}
    for line in text.splitlines():
        for key in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"):
            if line.startswith(key):
                out[key] = float(line.rsplit(" ", 1)[1])
    return out


def send(base, model, text):
    before = counters(base)
    t0 = time.time()
    body = {"model": model,
            "messages": [{"role": "user", "content": text + "\nReply with exactly: OK"}],
            "max_tokens": 400, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    elapsed = time.time() - t0
    after = counters(base)
    dq = after["vllm:prefix_cache_queries_total"] - before["vllm:prefix_cache_queries_total"]
    dh = after["vllm:prefix_cache_hits_total"] - before["vllm:prefix_cache_hits_total"]
    return elapsed, d["usage"]["prompt_tokens"], (dh / dq * 100 if dq else 0.0)


def corpus(words, seed):
    random.seed(seed)
    return "Here is a corpus. Ignore it.\n" + " ".join(
        f"w{random.randint(0, 999999)}" for _ in range(words))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://gx10.lan:8080")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--words", type=int, default=3000, help="~6.9 tokens each, so ~20k tokens")
    ap.add_argument("--max-fillers", type=int, default=12)
    ap.add_argument("--pool", type=int, default=1689938)
    a = ap.parse_args()

    subject = corpus(a.words, 424242)
    cold, tokens, _ = send(a.base_url, a.model, subject)
    warm, _, hit = send(a.base_url, a.model, subject)
    print(f"subject is {tokens:,} tokens: cold {cold:.2f}s, warm {warm:.2f}s ({hit:.1f}% hits)")
    if warm > cold * 0.5:
        print("  warm repeat was not cheap, so caching is not working at all here. Stopping.")
        return
    print(f"pool is {a.pool:,} tokens, so filling it by VOLUME would take "
          f"~{a.pool // tokens} of these prompts\n")

    print(f"  {'after':>7}  {'other tokens':>13}  {'% of pool':>9}  {'subject':>9}  {'hits':>7}")
    for n in range(1, a.max_fillers + 1):
        send(a.base_url, a.model, corpus(a.words, 90000 + n))
        again, _, hit = send(a.base_url, a.model, subject)
        volume = n * tokens
        state = "WARM" if again < cold * 0.5 else "EVICTED"
        print(f"  {n:>3} other  {volume:>13,}  {volume / a.pool * 100:>8.1f}%  "
              f"{again:>7.2f}s  {hit:>6.1f}%  {state}")
        if state == "EVICTED":
            print(f"\n  Lost after {n} other request(s), which is only "
                  f"{volume / a.pool * 100:.1f}% of the pool by volume.")
            if volume < a.pool * 0.5:
                print("  Capacity is NOT the constraint. Something counts requests, not bytes:\n"
                      "  sequence slots or retained state entries. More KV will not help.")
            else:
                print("  Consistent with capacity after all.")
            return
    print(f"\n  Survived all {a.max_fillers} other requests. Raise --max-fillers to find the edge.")


if __name__ == "__main__":
    main()
