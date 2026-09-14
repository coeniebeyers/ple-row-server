#!/usr/bin/env python3
"""Does the KV pool actually retain a session's prefix between turns?

A low prefix-cache hit rate in real use has two candidate causes that look identical from
outside: the blocks were evicted for capacity, or the context changed so the prefix no
longer matches. This separates them, using a prompt that provably does not change.

  1. send prompt A cold, note TTFT
  2. send A again, which should be nearly free if caching works at all
  3. push enough OTHER large prompts through to exceed the pool
  4. send A a third time

If step 4 is slow again, A was evicted for capacity, and the fix is a bigger pool. If it
stays fast, capacity is fine and the real-world misses come from the context changing
(compaction rewriting from the top, most likely), which no amount of pool buys back.

Run against an idle server.
"""
import argparse, json, random, time, urllib.request


def counters(base):
    text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = {}
    for line in text.splitlines():
        for key in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
                    "vllm:kv_cache_usage_perc"):
            if line.startswith(key):
                out[key] = float(line.rsplit(" ", 1)[1])
    return out


def corpus(words, seed):
    random.seed(seed)
    return " ".join(f"w{random.randint(0, 999999)}" for _ in range(words))


def send(base, model, text, tag, quiet=False):
    before = counters(base)
    t0 = time.time()
    body = {"model": model,
            "messages": [{"role": "user", "content": text + "\nReply with exactly: OK"}],
            "max_tokens": 500, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    elapsed = time.time() - t0
    after = counters(base)
    dq = after["vllm:prefix_cache_queries_total"] - before["vllm:prefix_cache_queries_total"]
    dh = after["vllm:prefix_cache_hits_total"] - before["vllm:prefix_cache_hits_total"]
    prompt = d["usage"]["prompt_tokens"]
    if not quiet:
        print(f"  {tag:<34} {elapsed:7.2f}s  prompt {prompt:>7}  "
              f"hits {dh:>8.0f}/{dq:<8.0f} = {(dh / dq * 100 if dq else 0):5.1f}%  "
              f"kv {after['vllm:kv_cache_usage_perc'] * 100:4.1f}%")
    return elapsed, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://gx10.lan:8080")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--words", type=int, default=18000, help="~6.9 tokens each")
    ap.add_argument("--fillers", type=int, default=8)
    a = ap.parse_args()

    subject = "Here is a corpus. Ignore it.\n" + corpus(a.words, 777)

    print("phase 1: establish that caching works at all")
    cold, prompt_tokens = send(a.base_url, a.model, subject, "A cold")
    warm, _ = send(a.base_url, a.model, subject, "A again (should be cheap)")
    if cold <= 0:
        return
    print(f"  -> repeat is {cold / max(warm, 1e-6):.1f}x faster, so caching itself works\n")

    pool_needed = a.fillers * prompt_tokens
    print(f"phase 2: push {a.fillers} other prompts through "
          f"(~{pool_needed:,} tokens, pool is ~1,046,030)")
    for i in range(a.fillers):
        send(a.base_url, a.model, "Here is a corpus. Ignore it.\n" + corpus(a.words, 1000 + i),
             f"filler {i + 1}/{a.fillers}")

    print("\nphase 3: is A still cached?")
    again, _ = send(a.base_url, a.model, subject, "A after the fillers")

    print()
    print(f"  A cold            {cold:7.2f}s")
    print(f"  A warm            {warm:7.2f}s")
    print(f"  A after fillers   {again:7.2f}s")
    if again > warm * 3:
        print(f"\n  EVICTED. A cost {again / warm:.0f}x its warm time after other work pushed it "
              f"out.\n  The pool cannot hold concurrent sessions' prefixes, so each turn re-prefills.\n"
              f"  Lever: a bigger KV pool (raise --gpu-memory-utilization).")
    else:
        print(f"\n  RETAINED. A stayed cheap, so capacity is not what loses the hit rate in real\n"
              f"  use. Look at the context changing instead: compaction rewrites from the top and\n"
              f"  invalidates everything after it, which no pool size can fix.")


if __name__ == "__main__":
    main()
