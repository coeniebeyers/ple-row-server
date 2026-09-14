#!/usr/bin/env bash
# Live throughput for the gx10 vLLM server. Ctrl-C to stop.
#
#   tps          sample every 5s
#   tps 10       sample every 10s
#
# GEN/S is decode (tokens produced). PRE/S is prefill (prompt tokens consumed).
# The two alternate: a long-context agent turn spends most of its time in prefill, during
# which GEN/S reads near zero. Low GEN/S on its own does NOT mean decode is slow, which is
# why both columns are here.
#
# One fetch per sample, every field parsed from it. Separate curls per metric time out
# under load and silently blank the columns.
set -u
URL="${URL:-http://gx10.lan:8080/metrics}"
IV="${1:-5}"

snap() { curl -s --max-time 4 "$URL" 2>/dev/null; }
field() { printf '%s\n' "$1" | awk -v k="^vllm:$2([{ ]|$)" '$0 ~ k {print $2; exit}'; }

m=$(snap)
[ -n "$m" ] || { echo "cannot reach $URL" >&2; exit 1; }
pg=$(field "$m" generation_tokens_total); pp=$(field "$m" prompt_tokens_total)
pd=$(field "$m" spec_decode_num_draft_tokens_total)
pa=$(field "$m" spec_decode_num_accepted_tokens_total)

printf "%-9s %8s %9s %5s %5s %8s %5s\n" TIME GEN/S PRE/S RUN WAIT MTP-ACC KV
while sleep "$IV"; do
  m=$(snap)
  [ -n "$m" ] || { echo "$(date +%H:%M:%S)  (unreachable)"; continue; }
  g=$(field "$m" generation_tokens_total); p=$(field "$m" prompt_tokens_total)
  d=$(field "$m" spec_decode_num_draft_tokens_total)
  a=$(field "$m" spec_decode_num_accepted_tokens_total)
  r=$(field "$m" num_requests_running); w=$(field "$m" num_requests_waiting)
  kv=$(field "$m" kv_cache_usage_perc)

  read -r gps pps acc <<<"$(awk -v g0="${pg:-0}" -v g1="${g:-0}" -v p0="${pp:-0}" -v p1="${p:-0}" \
      -v d0="${pd:-0}" -v d1="${d:-0}" -v a0="${pa:-0}" -v a1="${a:-0}" -v i="$IV" 'BEGIN {
        n = d1 - d0
        printf "%.1f %.0f %s", (g1-g0)/i, (p1-p0)/i, (n > 0 ? sprintf("%.0f%%", (a1-a0)/n*100) : "-")
      }')"
  printf "%-9s %8s %9s %5.0f %5.0f %8s %4.0f%%\n" \
    "$(date +%H:%M:%S)" "$gps" "$pps" "${r:-0}" "${w:-0}" "$acc" \
    "$(awk -v x="${kv:-0}" 'BEGIN { printf "%.0f", x * 100 }')"
  pg=$g; pp=$p; pd=$d; pa=$a
done
