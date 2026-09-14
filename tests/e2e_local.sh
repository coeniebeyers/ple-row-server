#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# End to end check of the row servers on this machine only: build the daemon,
# make two small synthetic halves, run one server per half on loopback, and
# drive both through the real client in client/ple_remote.py.
#
# Nothing here reaches node1, node2 or gx10. The servers bind 127.0.0.1 through
# an ephemeral port picked at run time, and everything else lives in a temp
# directory that goes away on exit.
#
#   tests/e2e_local.sh            run it
#   ROWS=16384 tests/e2e_local.sh bigger halves
#   PYTHON=/path/to/venv/bin/python tests/e2e_local.sh
#
# The client imports torch and numpy at module scope, so PYTHON has to point at
# an interpreter that has both.

set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
ROWS="${ROWS:-8192}"            # whole table, split in two
MAX_ROWS="${MAX_ROWS:-2048}"    # what both servers are started with
ROW_BYTES=160

TMP="$(mktemp -d "${TMPDIR:-/tmp}/ple-e2e.XXXXXX")"
PIDS=()
STATUS=1

cleanup() {
    local pid
    for pid in ${PIDS+"${PIDS[@]}"}; do
        kill "$pid" 2>/dev/null
    done
    for pid in ${PIDS+"${PIDS[@]}"}; do
        wait "$pid" 2>/dev/null
    done
    if [ "$STATUS" -ne 0 ]; then
        local log
        for log in "$TMP"/node*.log; do
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
    echo "e2e: $*" >&2
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

# ------------------------------------------------------------------- build

step "build"
# Out of tree so a run leaves nothing behind in server/, but with that Makefile
# and those sources, so this is the build the deploy path uses.
# -L so a symlinked server directory is copied rather than followed back into
# the repo, which would build in tree after all.
cp -RL "$REPO/server" "$TMP/build" || die "cannot copy $REPO/server"
make -C "$TMP/build" rowserverd || die "build failed"
SERVER="$TMP/build/rowserverd"
[ -x "$SERVER" ] || die "$SERVER was not built"

# ---------------------------------------------------------------- fixtures

HALF=$(( ROWS / 2 ))
[ $(( HALF * 2 )) -eq "$ROWS" ] || die "ROWS must be even, got $ROWS"
[ "$HALF" -gt "$MAX_ROWS" ] || die "ROWS/2 ($HALF) must be over MAX_ROWS ($MAX_ROWS)"

step "synthetic halves: $ROWS rows of $ROW_BYTES bytes, split at $HALF"
"$PYTHON" "$REPO/tests/e2e_local.py" gen --out "$TMP/half0.bin" --base 0 --count "$HALF" || die "gen half0"
"$PYTHON" "$REPO/tests/e2e_local.py" gen --out "$TMP/half1.bin" --base "$HALF" --count "$HALF" || die "gen half1"

# --------------------------------------------------------------- the servers

read -r PORT0 PORT1 <<<"$("$PYTHON" - <<'PY'
import socket

# Bound and closed rather than guessed, so a run does not collide with whatever
# else this machine happens to be listening on.
socks = [socket.socket() for _ in range(2)]
for s in socks:
    s.bind(("127.0.0.1", 0))
ports = [s.getsockname()[1] for s in socks]
for s in socks:
    s.close()
print(*ports)
PY
)"
[ -n "${PORT1:-}" ] || die "could not pick two free ports"

start_server() {
    local name="$1" file="$2" base="$3" port="$4"

    "$SERVER" --file "$file" --base-row "$base" --row-count "$HALF" \
        --port "$port" --max-rows "$MAX_ROWS" --threads 4 --warm \
        >"$TMP/$name.log" 2>&1 &
    PIDS+=("$!")

    local waited=0
    while ! "$PYTHON" -c "
import socket, sys
try:
    socket.create_connection(('127.0.0.1', $port), timeout=0.5).close()
except OSError:
    sys.exit(1)
" 2>/dev/null; do
        kill -0 "${PIDS[-1]}" 2>/dev/null || die "$name died before it listened (log below)"
        waited=$(( waited + 1 ))
        [ "$waited" -lt 100 ] || die "$name did not listen on port $port within 10 s"
        sleep 0.1
    done
    echo "$name: pid ${PIDS[-1]}, port $port, rows [$base, $(( base + HALF )))"
}

step "servers"
start_server node0 "$TMP/half0.bin" 0 "$PORT0"
start_server node1 "$TMP/half1.bin" "$HALF" "$PORT1"

# --------------------------------------------------------------- the checks

step "checks through client/ple_remote.py"
if "$PYTHON" "$REPO/tests/e2e_local.py" check \
        --client "$REPO/client/ple_remote.py" \
        --peers "127.0.0.1:$PORT0,127.0.0.1:$PORT1" \
        --rows-total "$ROWS" \
        --max-rows "$MAX_ROWS"; then
    STATUS=0
else
    STATUS=1
fi

# A check that passed while a server was dying would be a check that proved
# nothing, so both have to still be up at the end.
for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "e2e: server pid $pid is gone" >&2; STATUS=1; }
done

step "result"
if [ "$STATUS" -eq 0 ]; then
    echo "e2e: PASS"
else
    echo "e2e: FAIL"
fi
exit "$STATUS"
