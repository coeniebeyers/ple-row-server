#!/usr/bin/env bash
# Phase 0.2: stream the two PLE halves from gx10's safetensors straight onto node1/node2.
#
# Sequential on purpose. gx10 serves a live model off the same NVMe and sits at ~107/121 GiB,
# so we keep the instantaneous read rate to one stream and let the exporter's fadvise give
# every block back to the kernel. Wire cost is the same either way; only wall time differs.
#
# nc rather than scp/rsync: gx10 has no key to the nodes, and the SHA-256 gate in Phase 0.3
# is a stronger integrity check than anything the transport would give us.
set -u
ROWS_PER_HALF=160000768          # 64 shards x 2500012 -- shard-aligned, so each half is flat-addressable
BYTES_PER_HALF=25600122880
PORT="${PORT:-9000}"
DST="${DST:-$HOME/ple/half.bin}"
SRC="${SRC:-gx10}"                          # host holding the checkpoint, reachable by ssh
PEERS="${PEERS:-node1:172.16.0.28 node2:172.16.0.33}"   # ssh-name:address-the-source-dials

send_half() {
  local node=$1 ip=$2 start=$3
  echo "=== $node ($ip): rows [$start, $((start + ROWS_PER_HALF))) -> $DST ==="

  ssh "$node" "mkdir -p \$(dirname $DST); rm -f $DST; \
               nohup sh -c 'nc -l $PORT > $DST' >/dev/null 2>&1 & sleep 0.3" || return 1

  local i
  for i in $(seq 40); do
    ssh "$node" "ss -ltn 2>/dev/null | grep -q ':$PORT '" && break
    sleep 0.5
  done
  ssh "$node" "ss -ltn 2>/dev/null | grep -q ':$PORT '" || { echo "  NO LISTENER on $node"; return 1; }
  echo "  listener up, streaming..."

  local t0=$(date +%s)
  ssh "$SRC" "python3 ~/bin/ple_export_half.py --start $start --count $ROWS_PER_HALF | nc -N $ip $PORT" || {
    echo "  SENDER FAILED"; return 1; }
  local dt=$(( $(date +%s) - t0 ))

  # Hand the freshly written pages back so we don't sit on 23.8 GiB of cache next to etcd.
  ssh "$node" "python3 -c \"
import os
fd = os.open('$DST', os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)\"" 2>/dev/null

  local got=$(ssh "$node" "stat -c %s $DST")
  echo "  wrote $got B in ${dt}s ($(( got / (dt>0?dt:1) / 1000000 )) MB/s)"
  if [ "$got" != "$BYTES_PER_HALF" ]; then
    echo "  SIZE MISMATCH: got $got expected $BYTES_PER_HALF"
    return 1
  fi
  echo "  size OK"
}

echo "### peer link speeds (physical NICs only; k3s veths are noise)"
for spec in $PEERS; do
  h=${spec%%:*}
  ssh "$h" 'for d in $(ls /sys/class/net | grep -vE "^(lo|veth|cni|docker|flannel|br-)"); do
              [ "$(cat /sys/class/net/$d/operstate 2>/dev/null)" = up ] &&
              echo "  '"$h"' $d $(cat /sys/class/net/$d/speed 2>/dev/null)Mb/s"; done' 2>/dev/null
done

start=0
for spec in $PEERS; do
  send_half "${spec%%:*}" "${spec##*:}" "$start" || exit 1
  start=$((start + ROWS_PER_HALF))
done
echo "### TRANSFER COMPLETE"
