#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Prove RECHARGE (docs/protocol.md, op 4) on this machine only: build the daemon,
# lay down a synthetic table, warm it from *this* shell so the page cache is
# charged to the wrong cgroup exactly the way node2 was, start the daemon with
# --no-warm on top of it, and then drive one RECHARGE while a gather loop runs.
#
# Nothing here reaches node1, node2 or gx10. It is all driven over 127.0.0.1 on an
# ephemeral port, and everything else lives in a temp directory that goes away on
# exit. The daemon itself has no bind address option and listens on INADDR_ANY, so
# for the half minute a run lasts that port is open on every address this machine
# holds, serving nothing but the synthetic table.
#
#   tests/recharge_local.sh                      run it
#   SLICE_MIB=8 tests/recharge_local.sh          smaller slices, more of them
#   MUTATE=skip-touch tests/recharge_local.sh    break one step and watch it fail
#   PYTHON=/path/to/venv/bin/python tests/recharge_local.sh
#
# MUTATE patches the copy of rowserverd.c this run builds, never the repo, and is
# how the assertions are shown to have teeth. The three of them take out one step
# of the sequence each:
#
#   skip-madvise   leave the PTEs in place, so posix_fadvise cannot evict anything
#   skip-fadvise   zap the PTEs but never evict, so the touch refaults from cache
#   skip-touch     evict and walk away, leaving the slice cold
#   remap          drop the slice by unmapping and remapping it, the unsafe variant
#   wrong-row      serve the neighbouring row, so the byte comparison has a job
#
# Every one of them has to fail this test. The first two are the interesting ones:
# they are what the old external drop-the-cache tool was doing.
#
# The client imports torch and numpy at module scope, so PYTHON has to point at an
# interpreter that has both.

set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
ROWS="${ROWS:-2097152}"          # 320 MiB of 160 byte rows, on purpose an exact one
SLICE_MIB="${SLICE_MIB:-32}"     # 10 slices of the above, so the loop runs a few times
MAX_ROWS="${MAX_ROWS:-131072}"   # what the daemon runs with on the nodes
MUTATE="${MUTATE:-}"
ROW_BYTES=160

TMP="$(mktemp -d "${TMPDIR:-/tmp}/ple-recharge.XXXXXX")"
PIDS=()
CGROUP=""
STATUS=1

cleanup() {
    local pid
    for pid in ${PIDS+"${PIDS[@]}"}; do
        kill "$pid" 2>/dev/null
    done
    for pid in ${PIDS+"${PIDS[@]}"}; do
        wait "$pid" 2>/dev/null
    done
    # The cgroup only goes once the daemon has left it. Whatever page cache was
    # charged there is reparented, which is the kernel's business and not ours.
    [ -n "$CGROUP" ] && rmdir "$CGROUP" 2>/dev/null
    if [ "$STATUS" -ne 0 ]; then
        local log
        for log in "$TMP"/*.log; do
            [ -e "$log" ] || continue
            echo
            echo "--- $(basename "$log") ---"
            tail -n 40 "$log"
        done
    fi
    rm -rf "$TMP"
}
trap cleanup EXIT

die() {
    echo "recharge: $*" >&2
    exit 1
}

step() {
    echo
    echo "== $*"
}

"$PYTHON" - <<'PY' || die "PYTHON=$PYTHON is missing numpy or torch, which the client imports at module scope"
import importlib.util, sys
missing = [m for m in ("numpy", "torch") if importlib.util.find_spec(m) is None]
sys.exit(f"missing: {', '.join(missing)}" if missing else 0)
PY

BYTES=$(( ROWS * ROW_BYTES ))
[ $(( BYTES / (SLICE_MIB * 1048576) )) -ge 2 ] \
    || die "ROWS=$ROWS is $BYTES bytes, which is not more than one $SLICE_MIB MiB slice"

# ------------------------------------------------------------------- build

step "build${MUTATE:+ (mutated: $MUTATE)}"
# Out of tree, from those sources with that Makefile, so a run leaves nothing
# behind in server/ and a mutation cannot touch the repo.
cp -RL "$REPO/server" "$TMP/build" || die "cannot copy $REPO/server"

if [ -n "$MUTATE" ]; then
    "$PYTHON" - "$TMP/build/rowserverd.c" "$MUTATE" <<'PY' || die "mutation failed"
import sys

path, which = sys.argv[1], sys.argv[2]
# Whole lines, compared exactly, so a mutation that no longer matches the source is
# a failed run rather than a control that quietly stopped controlling anything. The
# indentation is part of the match: the touch in the fadvise error path sits one
# level deeper and has to be left alone, or the slice it rescues goes cold too.
edits = {
    "skip-madvise": ("        if (madvise(s->map + off, len, MADV_DONTNEED) != 0) {",
                     "        if (0) {"),
    "skip-fadvise": ("        err = posix_fadvise(s->fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);",
                     "        err = 0;"),
    "skip-touch":   ("        touch_range(s, off, len);",
                     None),
    # Not a step that was left out but the variant this one was chosen over: drop
    # the slice by unmapping and remapping it instead of by zapping the PTEs. The
    # nanosleep only widens a window that is there either way, so a racing gather
    # reads an address that is not mapped at all. Nothing else in this test is
    # allowed to make the daemon do that.
    "remap":        ("        if (madvise(s->map + off, len, MADV_DONTNEED) != 0) {",
                     "        munmap(s->map + off, len);\n"
                     "        { struct timespec gap = { 0, 2000000 }; nanosleep(&gap, NULL); }\n"
                     "        if (mmap(s->map + off, len, PROT_READ, MAP_SHARED | MAP_FIXED,\n"
                     "                 s->fd, (off_t)off) == MAP_FAILED) {"),
    # The gather itself, broken by one row, so the byte comparison in the loop has
    # something to catch.
    "wrong-row":    ("        memcpy(dst + (size_t)i * ROW_BYTES, table + (size_t)local * ROW_BYTES, ROW_BYTES);",
                     "        memcpy(dst + (size_t)i * ROW_BYTES, table + (size_t)(local ^ 1u) * ROW_BYTES, ROW_BYTES);"),
}
if which not in edits:
    sys.exit(f"unknown mutation {which}, expected one of {', '.join(edits)}")
old, new = edits[which]
lines = open(path).read().split("\n")
hits = [i for i, line in enumerate(lines) if line == old]
if len(hits) != 1:
    sys.exit(f"mutation {which}: {len(hits)} lines match\n  {old.strip()}\nexpected exactly one")
if new is None:
    del lines[hits[0]]
else:
    lines[hits[0]] = new
open(path, "w").write("\n".join(lines))
print(f"mutation {which}: line {hits[0] + 1} was\n  {old.strip()}\nand is now\n"
      f"  {'(deleted)' if new is None else new.strip()}")
PY
fi

make -C "$TMP/build" rowserverd || die "build failed"
SERVER="$TMP/build/rowserverd"
[ -x "$SERVER" ] || die "$SERVER was not built"

# ---------------------------------------------------------------- the table

step "synthetic table: $ROWS rows of $ROW_BYTES bytes, $(( BYTES / 1048576 )) MiB"
"$PYTHON" "$REPO/tests/e2e_local.py" gen --out "$TMP/half.bin" --base 0 --count "$ROWS" \
    || die "gen"
# Dirty pages cannot be evicted, and a sweep that finds nothing to drop would pass
# this test for the wrong reason.
sync

step "warm it from this shell, which is how the pages end up on the wrong cgroup"
"$PYTHON" "$REPO/tests/recharge_local.py" warm --file "$TMP/half.bin" || die "warm"

# ------------------------------------------------------- a cgroup of its own

# The daemon needs its own cgroup for any of this to be measurable, since the
# whole question is which cgroup the pages are charged to. Inside the delegated
# user subtree that is a mkdir, with no root and no systemd unit involved. If it
# is not available the run still proves the eviction and the re-read, just not the
# accounting, and says so.
MY_CG="$(awk -F: '$1 == "0" { print $3 }' /proc/self/cgroup)"
CG_PARENT="/sys/fs/cgroup$(dirname "$MY_CG")"
CANDIDATE="$CG_PARENT/ple-recharge-test-$$"
if mkdir "$CANDIDATE" 2>/dev/null && [ -r "$CANDIDATE/memory.stat" ]; then
    CGROUP="$CANDIDATE"
    echo "cgroup: $CGROUP"
else
    rmdir "$CANDIDATE" 2>/dev/null
    echo "cgroup: not available under $CG_PARENT, the run will skip the charge assertions"
fi

# ---------------------------------------------------------------- the server

PORT="$("$PYTHON" - <<'PY'
import socket

# Bound and closed rather than guessed, so a run does not collide with whatever
# else this machine is listening on.
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"
[ -n "$PORT" ] || die "could not pick a free port"

step "server on port $PORT, --no-warm on top of the already warm table"
# --no-warm is the point: the daemon must not fault a single page in for itself,
# so the mapping starts fully resident and fully charged to this shell.
if [ -n "$CGROUP" ]; then
    # The pid moves itself into the cgroup and then execs, so the daemon is in it
    # from its first instruction and $! is the daemon.
    bash -c 'echo $$ >"$1/cgroup.procs" || exit 1; shift; exec "$@"' _ "$CGROUP" \
        "$SERVER" --file "$TMP/half.bin" --base-row 0 --row-count "$ROWS" \
        --port "$PORT" --max-rows "$MAX_ROWS" --threads 8 --no-warm \
        >"$TMP/server.log" 2>&1 &
else
    "$SERVER" --file "$TMP/half.bin" --base-row 0 --row-count "$ROWS" \
        --port "$PORT" --max-rows "$MAX_ROWS" --threads 8 --no-warm \
        >"$TMP/server.log" 2>&1 &
fi
PIDS+=("$!")
SERVER_PID="${PIDS[-1]}"

waited=0
while ! "$PYTHON" -c "
import socket, sys
try:
    socket.create_connection(('127.0.0.1', $PORT), timeout=0.5).close()
except OSError:
    sys.exit(1)
" 2>/dev/null; do
    kill -0 "$SERVER_PID" 2>/dev/null || die "the server died before it listened (log below)"
    waited=$(( waited + 1 ))
    [ "$waited" -lt 100 ] || die "the server did not listen on port $PORT within 10 s"
    sleep 0.1
done
echo "server: pid $SERVER_PID, port $PORT, rows [0, $ROWS)"

if [ -n "$CGROUP" ]; then
    grep -qx "0::${CGROUP#/sys/fs/cgroup}" "/proc/$SERVER_PID/cgroup" \
        || die "the server is not in $CGROUP: $(cat "/proc/$SERVER_PID/cgroup")"
fi
"$PYTHON" "$REPO/tests/recharge_local.py" report --tool "$REPO/tools/recharge_cache.py" \
    --pid "$SERVER_PID" --cgroup "$CGROUP"

# --------------------------------------------------------------- the checks

step "one RECHARGE of $SLICE_MIB MiB slices, with a gather loop running through it"
if "$PYTHON" "$REPO/tests/recharge_local.py" check \
        --client "$REPO/client/ple_remote.py" \
        --tool "$REPO/tools/recharge_cache.py" \
        --port "$PORT" \
        --rows-total "$ROWS" \
        --max-rows "$MAX_ROWS" \
        --slice-mib "$SLICE_MIB" \
        --pid "$SERVER_PID" \
        --cgroup "$CGROUP"; then
    STATUS=0
else
    STATUS=1
fi

"$PYTHON" "$REPO/tests/recharge_local.py" report --tool "$REPO/tools/recharge_cache.py" \
    --pid "$SERVER_PID" --cgroup "$CGROUP"

# A pass while the server was on its way out would be a pass that proved nothing.
kill -0 "$SERVER_PID" 2>/dev/null || { echo "recharge: the server is gone" >&2; STATUS=1; }

step "result"
if [ "$STATUS" -eq 0 ]; then
    echo "recharge: PASS${MUTATE:+ (mutation $MUTATE was NOT caught)}"
else
    echo "recharge: FAIL${MUTATE:+ (expected, mutation $MUTATE)}"
fi
exit "$STATUS"
