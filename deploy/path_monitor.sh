#!/usr/bin/env bash
# Continuously prove whether the gx10 <-> peer path is up, and record exactly when it is not.
#
# Why this exists: on 2026-09-15 at 10:17 a PLE gather timed out and the following reconnect got
# EHOSTUNREACH, which killed the engine and cost ~15 minutes. Neither endpoint logged anything:
# no carrier change, no NetworkManager event, no NIC errors, and the peers were healthy
# throughout. That points at an intermediate switch, because gx10 is two L2 hops away and all
# three hosts share 172.16.0.0/24, so a reconvergence upstream is invisible to both ends.
#
# Absence of evidence was the problem, so this produces evidence: it probes both peers every
# second at both layers and writes a line ONLY when something is wrong, plus a heartbeat so a
# silent log can be distinguished from a dead monitor.
#
#   L3/TCP  a connect() to the row server port, which is what the client actually does and
#           what returned EHOSTUNREACH
#   L2/ARP  the neighbour state, because an entry going FAILED is the direct signature of the
#           ARP resolution failure behind EHOSTUNREACH
#
# Run under systemd or nohup. Output is append-only and tiny.
set -u
OUT="${OUT:-$HOME/path-monitor.log}"
PEERS="${PEERS:-172.16.0.28 172.16.0.33}"
PORT="${PORT:-9000}"
INTERVAL="${INTERVAL:-1}"
HEARTBEAT="${HEARTBEAT:-900}"        # seconds between "still fine" lines

say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$OUT"; }

declare -A down_since=()
last_beat=0
say "monitor started, probing [$PEERS]:$PORT every ${INTERVAL}s"

while :; do
  now=$(date +%s)
  for ip in $PEERS; do
    if timeout 2 bash -c "cat </dev/null >/dev/tcp/$ip/$PORT" 2>/dev/null; then
      if [ -n "${down_since[$ip]:-}" ]; then
        say "RECOVERED $ip after $(( now - down_since[$ip] ))s"
        unset "down_since[$ip]"
      fi
    else
      err=$?
      # The neighbour state is the interesting half: FAILED means ARP could not resolve,
      # which is what EHOSTUNREACH is reported for.
      neigh=$(ip neigh show "$ip" 2>/dev/null | awk '{print $NF}')
      route=$(ip route get "$ip" 2>&1 | head -1)
      if [ -z "${down_since[$ip]:-}" ]; then
        down_since[$ip]=$now
        say "DOWN $ip (connect rc=$err) neigh=${neigh:-none} route=${route}"
        say "     link=$(cat /sys/class/net/enP7s7/operstate 2>/dev/null) carrier_changes=$(cat /sys/class/net/enP7s7/carrier_changes 2>/dev/null)"
      fi
    fi
  done
  if [ $(( now - last_beat )) -ge "$HEARTBEAT" ]; then
    say "ok (${#down_since[@]} peer(s) currently down)"
    last_beat=$now
  fi
  sleep "$INTERVAL"
done
