#!/usr/bin/env bash
# Phase 0.3: prove each node's half.bin is byte-identical to the rows it claims to hold.
#
# This is the mandatory gate. The failure it exists to catch is the shard permutation
# (98/99 and 100/101 are physically swapped in the safetensors): a wrong exporter still
# produces a plausible 23.84 GiB file, and the model would just quietly gather garbage
# n-gram rows for ~1.6 GB of the vocab. Nothing downstream would raise.
#
# Both sides digest in identical 64 MiB blocks, so a mismatch also localises WHERE.
set -u
ROWS_PER_HALF=160000768
BLOCK_MIB=64
OUT="${OUT:-$(dirname "$0")/digests}"
SRC="${SRC:-gx10}"
PEERS="${PEERS:-node1 node2}"
mkdir -p "$OUT"

digest_half() {
  local node=$1 start=$2
  echo "=== $node: rows [$start, $((start + ROWS_PER_HALF))) ==="

  # Source of truth: read straight out of the safetensors on gx10, resolving by shard ID.
  # Backgrounded alongside the node so the two reads overlap instead of queueing.
  ssh "$SRC" "python3 ~/bin/ple_export_half.py --digest --start $start --count $ROWS_PER_HALF \
            --block-mib $BLOCK_MIB" > "$OUT/$node.src" 2>"$OUT/$node.src.err" &
  local src_pid=$!

  ssh "$node" "python3 ~/bin/ple_export_half.py --digest-plain ~/ple/half.bin \
               --block-mib $BLOCK_MIB" > "$OUT/$node.dst" 2>"$OUT/$node.dst.err" &
  local dst_pid=$!

  wait $src_pid || { echo "  source digest FAILED"; cat "$OUT/$node.src.err"; return 1; }
  wait $dst_pid || { echo "  node digest FAILED";   cat "$OUT/$node.dst.err"; return 1; }

  local s d
  s=$(grep '^TOTAL' "$OUT/$node.src" | awk '{print $2}')
  d=$(grep '^TOTAL' "$OUT/$node.dst" | awk '{print $2}')
  echo "  source : $s"
  echo "  $node  : $d"
  if [ -n "$s" ] && [ "$s" = "$d" ]; then
    echo "  ✅ MATCH ($(grep -c '^block' "$OUT/$node.src") blocks)"
    return 0
  fi

  echo "  ❌ MISMATCH — first differing blocks:"
  diff <(grep '^block' "$OUT/$node.src") <(grep '^block' "$OUT/$node.dst") | head -10
  return 1
}

# The node side only needs --digest-plain, which is stdlib-only (no torch, no model).
for h in $PEERS; do
  ssh "$h" 'mkdir -p ~/bin'
  scp -q "$(dirname "$0")/ple_export_half.py" "$h:~/bin/ple_export_half.py"
done

rc=0
start=0
for h in $PEERS; do
  digest_half "$h" "$start" || rc=1
  start=$((start + ROWS_PER_HALF))
done
echo
[ $rc -eq 0 ] && echo "### GATE GREEN — both halves verified, safe to proceed to Phase 1" \
              || echo "### GATE RED — do NOT build on these halves"
exit $rc
