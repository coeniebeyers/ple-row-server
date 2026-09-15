#!/usr/bin/env python3
"""End-to-end benchmark: how long does a real review take, and what did it cost?

Component metrics mislead. The long-prefill cap improved streaming smoothness 4x and made
cold prefill 1.7x slower at the same time; decode tok/s and prefill tok/s each told half that
story and neither told which way the trade went. Wall-clock time to a finished review is the
number that settles it, so this measures that directly and records the component metrics
alongside as explanation rather than as the goal.

Runs one or more opencode reviews against a fixed worktree, and records for each config:
  - wall time to completion, which is the thing being optimised
  - requests, prompt and generated tokens, so cost is comparable across runs
  - TTFT, decode ms/token, prefix-cache hit rate, peak concurrency, preemptions
  - the review's own output, kept so quality can be judged later or compared against a
    reference set of findings

Parallel mode runs N reviews at once, because the real question is throughput across agents,
not single-stream speed.

Results append to a JSONL ledger so two configs can be compared without rerunning the first.
"""
import argparse, json, os, re, statistics, subprocess, sys, threading, time, urllib.request


def metrics(base):
    try:
        text = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        for key in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
                    "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count",
                    "vllm:request_decode_time_seconds_sum", "vllm:prefix_cache_hits_total",
                    "vllm:prefix_cache_queries_total", "vllm:num_requests_running",
                    "vllm:num_preemptions_total", "vllm:kv_cache_usage_perc"):
            if line.startswith(key):
                out[key] = float(line.rsplit(" ", 1)[1])
    return out


class Watcher(threading.Thread):
    """Sample concurrency while the review runs; the peak is what exercises the seat limit."""

    daemon = True

    def __init__(self, base):
        super().__init__()
        self.base, self.peak, self.kv_peak, self.stop_flag = base, 0.0, 0.0, False

    def run(self):
        while not self.stop_flag:
            m = metrics(self.base)
            self.peak = max(self.peak, m.get("vllm:num_requests_running", 0))
            self.kv_peak = max(self.kv_peak, m.get("vllm:kv_cache_usage_perc", 0))
            time.sleep(2)


def run_one(idx, workdir, prompt, model, outdir, timeout_s):
    out_path = os.path.join(outdir, f"review_{idx}.txt")
    t0 = time.time()
    with open(out_path, "w") as fh, open(os.devnull) as devnull:
        rc = subprocess.call(["timeout", str(timeout_s), "opencode", "run", "--model", model, prompt],
                             cwd=workdir, stdout=fh, stderr=subprocess.STDOUT, stdin=devnull)
    return {"idx": idx, "rc": rc, "wall_s": time.time() - t0,
            "out_bytes": os.path.getsize(out_path), "out_path": out_path}


def count_findings(path):
    """Rough quality proxy: severity-tagged findings, and whether it did the adversarial pass."""
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return {}
    sev = {s: len(re.findall(r"\[%s" % s, text, re.I)) for s in ("HIGH", "MEDIUM", "LOW")}
    return {"severity_counts": sev,
            "has_confirmed_section": bool(re.search(r"CONFIRMED", text)),
            "has_discarded_section": bool(re.search(r"DISCARD", text, re.I)),
            "file_line_citations": len(re.findall(r"\w+\.(?:sol|ts|sh|md):\d+", text))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, help="worktree the review runs against")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--label", required=True, help="what config this run is measuring")
    ap.add_argument("--base-url", default="http://gx10.lan:8080")
    ap.add_argument("--model", default="gx10/qwen3.8-flash-next")
    ap.add_argument("--parallel", type=int, default=1, help="reviews to run at once")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--outdir", default="/tmp/review_bench")
    ap.add_argument("--ledger", default=os.path.expanduser("~/review_bench.jsonl"))
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    prompt = open(a.prompt_file).read()
    before = metrics(a.base_url)
    watcher = Watcher(a.base_url)
    watcher.start()

    print(f"=== {a.label}: {a.parallel} review(s) in parallel ===")
    t0 = time.time()
    results = [None] * a.parallel
    threads = []
    for i in range(a.parallel):
        def worker(i=i):
            results[i] = run_one(i, a.workdir, prompt, a.model, a.outdir, a.timeout)
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.time() - t0
    watcher.stop_flag = True
    after = metrics(a.base_url)

    def delta(key):
        return after.get(key, 0) - before.get(key, 0)

    gen, dec = delta("vllm:generation_tokens_total"), delta("vllm:request_decode_time_seconds_sum")
    ttft_s, ttft_c = delta("vllm:time_to_first_token_seconds_sum"), delta("vllm:time_to_first_token_seconds_count")
    hits, queries = delta("vllm:prefix_cache_hits_total"), delta("vllm:prefix_cache_queries_total")

    record = {
        "label": a.label,
        "parallel": a.parallel,
        "wall_s": round(wall, 1),
        "slowest_review_s": round(max(r["wall_s"] for r in results if r), 1),
        "requests": int(ttft_c),
        "prompt_tokens": int(delta("vllm:prompt_tokens_total")),
        "generated_tokens": int(gen),
        "ttft_mean_s": round(ttft_s / ttft_c, 2) if ttft_c else None,
        "decode_ms_per_token": round(dec / gen * 1000, 2) if gen else None,
        "aggregate_gen_tok_s": round(gen / wall, 1) if wall else None,
        "prefix_hit_pct": round(hits / queries * 100, 1) if queries else None,
        "peak_concurrency": watcher.peak,
        "peak_kv_pct": round(watcher.kv_peak * 100, 1),
        "preemptions": int(delta("vllm:num_preemptions_total")),
        "reviews": [{k: v for k, v in r.items() if k != "out_path"} | count_findings(r["out_path"])
                    for r in results if r],
    }
    print(json.dumps(record, indent=2))
    with open(a.ledger, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"\nappended to {a.ledger}")
    print("  the number to optimise is wall_s; everything else explains it")


if __name__ == "__main__":
    main()
