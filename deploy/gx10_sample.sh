#!/usr/bin/env bash
# Sample the numbers that would show memory thrash developing, once every few minutes.
#
# Thrash on this box is a slow phenomenon: KV peaks across concurrent sessions gradually
# evict weight pages until decode starts taking major faults. A snapshot taken minutes
# after a restart cannot see it, which is exactly the mistake that produced the earlier
# "no memory win" conclusion. This exists so the question gets answered with a week of
# data instead of another guess.
#
# Append-only CSV. Compare pgmajfault and workingset_refault_file growth per hour against
# TTFT growth over the same window.
set -u
OUT="${OUT:-$HOME/ple-thrash.csv}"
NAME="${NAME:-qwen-fn-remote-on}"
PORT="${PORT:-8080}"

if [ ! -f "$OUT" ]; then
  echo "ts,uptime_s,mem_used_gb,mem_cache_gb,mem_free_gb,swap_used_gb,vllm_swap_kb,pgmajfault,refault_file,pgscan,pgsteal,kv_usage,running,waiting,prefix_hits,prefix_queries,ttft_sum,ttft_count,gen_tokens,decode_s" > "$OUT"
fi

ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
read -r mem_used mem_cache mem_free <<<"$(free -g | awk '/^Mem:/{print $3, $6, $4}')"
swap_used=$(free -g | awk '/^Swap:/{print $3}')

cid=$(docker inspect -f '{{.Id}}' "$NAME" 2>/dev/null || true)
pid=$(docker inspect -f '{{.State.Pid}}' "$NAME" 2>/dev/null || echo 0)
started=$(docker inspect -f '{{.State.StartedAt}}' "$NAME" 2>/dev/null || true)
uptime_s=0
[ -n "$started" ] && uptime_s=$(( $(date +%s) - $(date -d "$started" +%s 2>/dev/null || date +%s) ))

vllm_swap=0
for p in $(pgrep -f "VLLM::|vllm serve" 2>/dev/null); do
  v=$(awk '/^VmSwap/{print $2}' "/proc/$p/status" 2>/dev/null || echo 0)
  vllm_swap=$(( vllm_swap + ${v:-0} ))
done

maj=0; refault=0; pgscan=0; pgsteal=0
cg="/sys/fs/cgroup/system.slice/docker-$cid.scope/memory.stat"
if [ -n "$cid" ] && [ -f "$cg" ]; then
  maj=$(awk '/^pgmajfault /{print $2}' "$cg")
  refault=$(awk '/^workingset_refault_file /{print $2}' "$cg")
  pgscan=$(awk '/^pgscan /{print $2}' "$cg")
  pgsteal=$(awk '/^pgsteal /{print $2}' "$cg")
fi

m=$(curl -s --max-time 8 "localhost:$PORT/metrics" 2>/dev/null || true)
pick() { printf '%s\n' "$m" | grep -m1 "^$1{" | awk '{print $2}'; }
kv=$(pick "vllm:kv_cache_usage_perc"); run=$(pick "vllm:num_requests_running")
wait=$(pick "vllm:num_requests_waiting"); ph=$(pick "vllm:prefix_cache_hits_total")
pq=$(pick "vllm:prefix_cache_queries_total")
ts_sum=$(printf '%s\n' "$m" | grep -m1 "^vllm:time_to_first_token_seconds_sum" | awk '{print $2}')
ts_cnt=$(printf '%s\n' "$m" | grep -m1 "^vllm:time_to_first_token_seconds_count" | awk '{print $2}')
gen=$(pick "vllm:generation_tokens_total")
dec=$(printf '%s\n' "$m" | grep -m1 "^vllm:request_decode_time_seconds_sum" | awk '{print $2}')

echo "$ts,$uptime_s,$mem_used,$mem_cache,$mem_free,$swap_used,$vllm_swap,${maj:-0},${refault:-0},${pgscan:-0},${pgsteal:-0},${kv:-},${run:-},${wait:-},${ph:-},${pq:-},${ts_sum:-},${ts_cnt:-},${gen:-},${dec:-}" >> "$OUT"
