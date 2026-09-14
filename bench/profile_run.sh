#!/usr/bin/env bash
# Capture torch profiler traces for a decode-bound and a prefill-bound workload.
#
# The point is proportions, not absolutes. Profiling perturbs timing, so do not quote tok/s
# out of a traced run; quote where the time went. Two captures because the two regimes have
# completely different shapes: a token costs ~22 ms and a 124k cold prefill costs ~61 s, and
# nothing about the first explains the second.
#
# Needs the server started with the profiler enabled, which is a restart:
#   EXTRA="... --profiler-config.profiler=torch \
#              --profiler-config.torch_profiler_dir=/profiles \
#              --profiler-config.detailed_trace_annotation=true"
# and a host directory bind-mounted at /profiles.
#
# Specific questions these traces are meant to answer:
#   - how a 22 ms token splits across MoE, full attention, GDN/Mamba and host work
#   - whether the GPU idles across the ids.to("cpu") sync in the staged PLE path, which
#     happens once per step and is the one place host work can stall the device
#   - what the in-engine PLE gather actually costs, rather than the 0.35 ms an external
#     probe measures
#   - whether MTP-3's three draft forwards earn their keep at long context
set -u
BASE="${BASE:-http://gx10.lan:8080}"
MODEL="${MODEL:-qwen3.8-flash-next}"
OUT="${OUT:-/profiles}"

post() { curl -s --max-time 60 -X POST "$BASE/$1" -o /dev/null -w "  $1 -> HTTP %{http_code}\n"; }

ask() {  # ask <words> <max_tokens> <label>
  python3 - "$1" "$2" "$3" "$BASE" "$MODEL" <<'PY'
import json, random, sys, time, urllib.request
words, max_tokens, label, base, model = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
random.seed(hash(label) & 0xFFFF)
corpus = " ".join(f"w{random.randint(0, 999999)}" for _ in range(words))
body = {"model": model,
        "messages": [{"role": "user", "content": "Ignore this corpus.\n" + corpus
                      + "\nCount slowly from 1 to 60, one per line."}],
        "max_tokens": max_tokens, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}}
t0 = time.time()
r = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                           headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=1800))
print(f"  {label}: {time.time()-t0:.2f}s, prompt {d['usage']['prompt_tokens']}, "
      f"completion {d['usage']['completion_tokens']}")
PY
}

echo "=== capture 1: decode-bound (short prompt, long generation) ==="
ask 200 40 "warmup"                      # get past cold prefill and graph capture
post start_profile
ask 200 300 "traced decode"
post stop_profile

echo
echo "=== capture 2: prefill-bound (124k cold prompt) ==="
post start_profile
ask 18000 20 "traced prefill"
post stop_profile

echo
echo "traces are in $OUT on gx10. Load them in https://ui.perfetto.dev"
echo "Look first at: the gap between consecutive engine steps, and whether the GPU"
echo "track is idle while the CPU track is inside the PLE gather."
