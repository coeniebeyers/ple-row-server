#!/usr/bin/env bash
# Sample one peer's residency and fault counters, so cache drift is visible before a
# client feels it as slow decode.
#
# Residency alone is not enough to know you are safe: a peer can report 100% while every
# page is charged to another cgroup, where memory.low does not protect it. That is exactly
# how node2 drifted to 96.5% and 57,799 major faults while looking healthy. So this records
# the cgroup charge next to the residency, and the gap between them is the warning sign.
set -u
OUT="${OUT:-$HOME/ple-peer.csv}"
NAME="${NAME:-ple-row}"
PORT="${PORT:-9000}"

if [ ! -f "$OUT" ]; then
  echo "ts,resident_pages,total_pages,resident_pct,served,cgroup_file_gb,cgroup_low_gb,pgmajfault,refault_file,mem_used_gb,mem_cache_gb" > "$OUT"
fi

read -r resident total served <<<"$(python3 - "$PORT" <<'PY'
import socket, struct, sys
s = socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=10)
s.sendall(struct.pack("<IBBHII", 0x52454C50, 1, 3, 0, 1, 0))
def rd(n):
    b = bytearray(n); mv = memoryview(b); g = 0
    while g < n:
        k = s.recv_into(mv[g:], n - g)
        if not k: raise IOError("closed")
        g += k
    return b
magic, rid, status, count = struct.unpack("<IIII", rd(16))
base, rows, rb, resident, served = struct.unpack("<QQIIQ", rd(32))
print(resident, (rows * rb + 4095) // 4096, served)
PY
)" 2>/dev/null || { resident=0; total=1; served=0; }

cid=$(docker inspect -f '{{.Id}}' "$NAME" 2>/dev/null || true)
cg="/sys/fs/cgroup/system.slice/docker-$cid.scope"
file_gb=0; low_gb=0; maj=0; refault=0
if [ -n "$cid" ] && [ -f "$cg/memory.stat" ]; then
  file_gb=$(( $(awk '/^file /{print $2}' "$cg/memory.stat") / 1073741824 ))
  low_gb=$(( $(cat "$cg/memory.low" 2>/dev/null || echo 0) / 1073741824 ))
  maj=$(awk '/^pgmajfault /{print $2}' "$cg/memory.stat")
  refault=$(awk '/^workingset_refault_file /{print $2}' "$cg/memory.stat")
fi
read -r mem_used mem_cache <<<"$(free -g | awk '/^Mem:/{print $3, $6}')"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$resident,$total,$(( resident * 100 / (total > 0 ? total : 1) )),$served,$file_gb,$low_gb,${maj:-0},${refault:-0},$mem_used,$mem_cache" >> "$OUT"
