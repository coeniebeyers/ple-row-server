#!/usr/bin/env bash
# Build and run one half of the PLE row server on a peer node.
#
# Run this ON the node: the container bind-mounts that node's own half.bin and the build
# needs the repo. node1 is a live k3s etcd member, so every limit below is picked to make
# the row server the first thing the kernel gives up on and etcd the last.
#
# Every daemon flag here comes from server/rowserverd.c, which takes --file, --base-row,
# --row-count, --port, --max-rows, --threads, --warm and --no-warm. It has no bind address
# option: it listens on INADDR_ANY, so with --network host it is reachable on every address
# the node holds. deploy/README.md says what to do about that if you care.
set -euo pipefail

ROW_BYTES=160                    # ple_embed_dim / ngram_heads = 2560 / 16, not config head_dim
ROWS_PER_HALF=160000768          # 64 shards x 2500012, shard aligned so each half stays flat

NODE1_ADDR=172.16.0.28
NODE2_ADDR=172.16.0.33

# Image and container names are server/Dockerfile's own build and run lines.
IMAGE="${IMAGE:-ple-row-server}"
CONTAINER="${CONTAINER:-ple-row}"

# DOCKER="sudo docker" is two words. Holding it as an array is what stops it being looked
# up as a single command name.
DOCKER="${DOCKER:-docker}"
read -r -a DOCKER_CMD <<<"$DOCKER" || true

# The container runs as this uid, from server/Dockerfile's USER line, and docker does not
# remap it. Whatever half.bin's permissions are is what the daemon gets.
CONTAINER_UID=65534
CONTAINER_GID=65534
# Empty means run as the Dockerfile's nobody. Set by --user/--as-me when half.bin lives
# somewhere nobody cannot reach, which is the normal case for a file under a home dir.
RUN_USER=""
MOUNT_PATH=/ple/half.bin

# Half of this script wants root (docker, the cgroup files) and Ubuntu's sudo resets HOME
# to /root, so resolve the table against the invoking user rather than the effective one.
OWNER_HOME=$HOME
if [ -n "${SUDO_USER:-}" ]; then
  OWNER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
fi
TABLE="${TABLE:-$OWNER_HOME/ple/half.bin}"

# 9000 is rowserverd's DEFAULT_PORT, server/Dockerfile's EXPOSE and client/ple_remote.py's
# DEFAULT_PORT. Changing it here means changing it in PLE_REMOTE_PEERS as well.
PORT="${PORT:-9000}"
MAX_ROWS="${MAX_ROWS:-131072}"   # a 4096 token prefill chunk is 65,536 rows, so 2x headroom
THREADS="${THREADS:-8}"          # one thread per connection; the client opens one per peer

# A full half is 23.842 GiB. MEM_LOW sits just above it so reclaim protection covers the
# mapping and nothing else; the rest of MEM_MAX is for in-flight responses (20 MiB each at
# max_rows) and allocator slop. MEM_HIGH is a brake on a leak, not a level we expect to
# reach. Override all three if you serve a range that is not a full half.
MEM_MAX="${MEM_MAX:-26g}"
MEM_HIGH="${MEM_HIGH:-25g}"
MEM_LOW="${MEM_LOW:-24g}"

# Docker's json-file driver does not rotate on its own, and a full disk on a k3s node stops
# etcd writing. Both options are set explicitly, driver included, so a daemon configured
# with a different default driver cannot quietly drop the bound.
LOG_MAX_SIZE="${LOG_MAX_SIZE:-10m}"
LOG_MAX_FILE="${LOG_MAX_FILE:-3}"

PAGE_SIZE="${PAGE_SIZE:-4096}"
RESIDENT_MIN_PCT="${RESIDENT_MIN_PCT:-95}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
BUILD_CONTEXT="$REPO_ROOT/server"

note() { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<EOF
usage: run-node.sh <command> [options]

commands:
  build                  build $IMAGE from $BUILD_CONTEXT
  start                  run the container for this node's half
  stop                   stop and remove the container, leaving the image and the table
  status                 container state plus the cgroup memory numbers that matter
  health [target]        PING + STAT against a running server; target is node1, node2
                         or host[:port] (default: this node)
  purge                  stop, remove container and image, hand the page cache back

start options:
  --base-row N           first global row this server owns
  --row-count N          how many rows it owns
  --table PATH           host path to half.bin, mounted at $MOUNT_PATH (default $TABLE)
  --port N               default $PORT
  --max-rows N           default $MAX_ROWS
  --threads N            default $THREADS
  --user UID:GID         run the daemon as this user instead of the image's nobody
  --as-me                shorthand for --user with the invoking user's ids, for a table
                         under a home directory that nobody cannot traverse
  --no-warm              skip the startup fault-in pass and let rows arrive on demand
  --no-build             fail instead of building a missing image
  --yes                  proceed on a k3s control-plane node without being asked
  --dry-run              print the docker command and stop

health options:
  --base-row N           geometry to assert for a target that is not node1 or node2
  --row-count N          likewise

environment:
  NODE=node1|node2       override host detection
  YES=1                  same as --yes, for scripted runs over ssh
  DOCKER="sudo docker"   if your user is not in the docker group
  TABLE PORT MAX_ROWS THREADS IMAGE CONTAINER
  MEM_MAX MEM_HIGH MEM_LOW LOG_MAX_SIZE LOG_MAX_FILE PAGE_SIZE RESIDENT_MIN_PCT

The daemon writes its log to stderr. Docker captures it either way: docker logs $CONTAINER
EOF
}

bytes_of() {
  local v=$1 n unit
  n=${v%[gGmMkK]}
  unit=${v#"$n"}
  case "$unit" in
    g|G) echo $(( n * 1024 * 1024 * 1024 )) ;;
    m|M) echo $(( n * 1024 * 1024 )) ;;
    k|K) echo $(( n * 1024 )) ;;
    "")  echo "$n" ;;
    *)   die "cannot parse size '$v'" ;;
  esac
}

gib() { awk -v b="$1" 'BEGIN { printf "%.3f", b / 1073741824 }'; }

# The nodes were netbooted and carry generated hostnames, so the address they hold is the
# only identifier that is reliably right.
detect_node() {
  if [ -n "${NODE:-}" ]; then printf '%s\n' "$NODE"; return 0; fi
  case "$(hostname -s 2>/dev/null || true)" in
    node1|*0b2ad6*) printf 'node1\n'; return 0 ;;
    node2|*0b2904*) printf 'node2\n'; return 0 ;;
  esac
  local addrs
  addrs=" $(ip -o -4 addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | tr '\n' ' ')"
  case "$addrs" in
    *" $NODE1_ADDR "*) printf 'node1\n'; return 0 ;;
    *" $NODE2_ADDR "*) printf 'node2\n'; return 0 ;;
  esac
  return 1
}

half_geometry() {
  case "$1" in
    node1) BASE_ROW=0 ;;
    node2) BASE_ROW=$ROWS_PER_HALF ;;
    *) die "unknown node '$1', pass --base-row and --row-count explicitly" ;;
  esac
  ROW_COUNT=$ROWS_PER_HALF
}

node_addr() {
  case "$1" in
    node1) printf '%s\n' "$NODE1_ADDR" ;;
    node2) printf '%s\n' "$NODE2_ADDR" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

cgroup_path() {
  local pid cg
  pid=$("${DOCKER_CMD[@]}" inspect -f '{{.State.Pid}}' "$CONTAINER" 2>/dev/null) || return 1
  [ -n "$pid" ] && [ "$pid" != 0 ] || return 1
  cg=$(awk -F: '$1=="0"{print $3}' "/proc/$pid/cgroup" 2>/dev/null) || return 1
  [ -n "$cg" ] || return 1
  printf '/sys/fs/cgroup%s\n' "$cg"
}

require_docker() {
  command -v "${DOCKER_CMD[0]}" >/dev/null 2>&1 || die "${DOCKER_CMD[0]} not found"
  "${DOCKER_CMD[@]}" info >/dev/null 2>&1 ||
    die "cannot talk to the docker daemon, try DOCKER='sudo docker'"
}

container_exists() {
  "${DOCKER_CMD[@]}" inspect "$CONTAINER" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------- build
do_build() {
  require_docker
  local node
  node=$(detect_node || true)
  if [ "$node" = node1 ]; then
    warn "building on node1 pulls layers and compiler output through the page cache,"
    warn "which is the one thing that reliably evicts the table. Prefer building"
    warn "elsewhere and moving the image:"
    warn "  docker save $IMAGE | ssh node1 docker load"
  fi
  [ -f "$BUILD_CONTEXT/Dockerfile" ] || die "no Dockerfile at $BUILD_CONTEXT"
  "${DOCKER_CMD[@]}" build -t "$IMAGE" "$BUILD_CONTEXT"
}

# ---------------------------------------------------------------------------- start
# POSIX permission classes: the first matching class decides, so a file owned by the
# container uid with no owner read bit is unreadable however open the other bits are.
# ACLs, AppArmor and SELinux are not consulted here.
accessible_by_container() {
  local path=$1 bit=$2 mode owner group
  mode=$(stat -c '%a' "$path")
  owner=$(stat -c '%u' "$path")
  group=$(stat -c '%g' "$path")
  if [ "$owner" = "$CONTAINER_UID" ]; then
    [ $(( 10#$mode / 100 % 10 & bit )) -ne 0 ]
  elif [ "$group" = "$CONTAINER_GID" ]; then
    [ $(( 10#$mode / 10 % 10 & bit )) -ne 0 ]
  else
    [ $(( 10#$mode % 10 & bit )) -ne 0 ]
  fi
}

check_table_access() {
  local dir bad=0
  if ! accessible_by_container "$TABLE" 4; then
    warn "$TABLE (mode $(stat -c '%a %U:%G' "$TABLE")) is not readable by uid $CONTAINER_UID"
    warn "  fix: chmod o+r $TABLE"
    bad=1
  fi
  dir=$(dirname "$TABLE")
  while :; do
    if ! accessible_by_container "$dir" 1; then
      warn "$dir (mode $(stat -c '%a %U:%G' "$dir")) is not searchable by uid $CONTAINER_UID"
      warn "  fix: chmod o+x $dir"
      bad=1
    fi
    [ "$dir" = / ] && break
    dir=$(dirname "$dir")
  done
  [ "$bad" = 0 ] || die "the container runs as $CONTAINER_UID:$CONTAINER_GID and cannot open the table.
       Open the path up as above, or use the systemd install instead, which runs as a
       real user. The daemon only ever opens the file O_RDONLY."
}

# Runs over ssh with no tty in every documented invocation, so it cannot be a prompt that
# quietly skips itself. Without an acknowledgement it fails closed.
k3s_gate() {
  systemctl is-active --quiet k3s 2>/dev/null ||
    systemctl is-active --quiet k3s-agent 2>/dev/null ||
    return 0

  warn "k3s is running on this box. If node1, that is an etcd member of a three member"
  warn "cluster, and a row server that misbehaves here costs the control plane."
  if [ "${YES:-0}" = 1 ]; then
    note "k3s acknowledged (--yes)"
    return 0
  fi
  if [ -t 0 ] && [ -t 1 ]; then
    local answer
    read -r -p "continue? [y/N] " answer
    case "$answer" in y|Y|yes) return 0 ;; *) die "aborted" ;; esac
  fi
  die "no terminal to ask on. Re-run with --yes (or YES=1) once you have read
       deploy/README.md's section on what this does to node1."
}

preflight() {
  local size want avail_kb table_bytes

  [ -f "$TABLE" ] || die "no table at $TABLE (tools/transfer_halves.sh puts it there)"
  size=$(stat -c %s "$TABLE")
  want=$(( ROW_COUNT * ROW_BYTES ))
  [ "$size" = "$want" ] ||
    die "$TABLE is $size bytes, expected $want for $ROW_COUNT rows of $ROW_BYTES.
       rowserverd makes the same check and refuses to start; a half that does not hold
       the rows it claims would just serve wrong embeddings."
  check_table_access

  table_bytes=$want
  if [ "$(bytes_of "$MEM_MAX")" -lt "$table_bytes" ]; then
    warn "MEM_MAX $MEM_MAX is below the $(gib "$table_bytes") GiB this range occupies."
    warn "The server will run, it will just never hold all of it. Set MEM_MAX, MEM_HIGH"
    warn "and MEM_LOW to match the range you are serving."
  fi

  # MemAvailable already counts reclaimable page cache, so a table still warm from a
  # previous run passes this rather than looking like a shortfall.
  avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  if [ "$(( avail_kb * 1024 ))" -lt "$table_bytes" ]; then
    warn "MemAvailable is $(gib "$(( avail_kb * 1024 ))") GiB, less than the"
    warn "$(gib "$table_bytes") GiB table. Expect the half to sit partly on disk."
  fi

  [ -f /sys/fs/cgroup/cgroup.controllers ] ||
    warn "this box is not on cgroup v2, so memory.low and memory.high do not apply as documented"

  if command -v ss >/dev/null 2>&1; then
    if ss -ltn 2>/dev/null | grep -q "[:.]$PORT "; then
      die "something already listens on port $PORT"
    fi
  else
    warn "ss not installed, skipping the port check; docker will fail on a clash instead"
  fi

  k3s_gate
}

do_start() {
  local node build=1 dry=0 warm_flag=--warm
  node=$(detect_node || true)
  if [ -n "$node" ]; then half_geometry "$node"; else BASE_ROW=; ROW_COUNT=; fi

  while [ $# -gt 0 ]; do
    case "$1" in
      --base-row)  BASE_ROW=${2:?--base-row needs a value}; shift 2 ;;
      --row-count) ROW_COUNT=${2:?--row-count needs a value}; shift 2 ;;
      --table)     TABLE=${2:?--table needs a path}; shift 2 ;;
      --port)      PORT=${2:?--port needs a value}; shift 2 ;;
      --max-rows)  MAX_ROWS=${2:?--max-rows needs a value}; shift 2 ;;
      --threads)   THREADS=${2:?--threads needs a value}; shift 2 ;;
      --user)      RUN_USER=${2:?--user needs UID:GID}
                   CONTAINER_UID=${RUN_USER%%:*}; CONTAINER_GID=${RUN_USER##*:}; shift 2 ;;
      --as-me)     RUN_USER="$(id -u):$(id -g)"
                   CONTAINER_UID=$(id -u); CONTAINER_GID=$(id -g); shift ;;
      --no-warm)   warm_flag=--no-warm; shift ;;
      --no-build)  build=0; shift ;;
      --yes)       YES=1; shift ;;
      --dry-run)   dry=1; shift ;;
      *) die "unknown option '$1'" ;;
    esac
  done

  [ -n "${BASE_ROW:-}" ] && [ -n "${ROW_COUNT:-}" ] ||
    die "cannot tell which half this box serves, pass NODE= or --base-row/--row-count"

  # --network host: bridge networking puts NAT and the userland proxy in front of every
  #   request, and the gather budget is about 1 ms for a whole decode step.
  # --read-only: the daemon opens one file and writes to a socket and to stderr. Nothing
  #   in the image is writable, and there is nothing in the image but the binary.
  # --ulimit memlock=0: mlock would make the table unreclaimable, which is exactly the
  #   failure mode that takes etcd down with it.
  # --oom-score-adj 1000: if the node OOMs anyway, we are the intended victim.
  # no --cpus: CFS quota enforces over a 100 ms period, so exceeding it parks us for
  #   longer than a whole engine step. Weight only bites when the CPU is contended.
  # --restart on-failure:3: enough for a transient crash, few enough that a node under
  #   pressure is not fed a container that faults 23.8 GiB back in on every loop.
  local -a cmd=(
    "${DOCKER_CMD[@]}" run -d
    --name "$CONTAINER"
    --network host
    --read-only
    --cap-drop ALL
    --security-opt no-new-privileges
    --ulimit memlock=0:0
    --ulimit nofile=1024:1024
    --pids-limit 256
    --oom-score-adj 1000
    ${RUN_USER:+--user "$RUN_USER"}
    --memory "$MEM_MAX"
    --memory-swap "$MEM_MAX"
    --memory-reservation "$MEM_LOW"
    --cpu-shares 512
    --restart on-failure:3
    --log-driver json-file
    --log-opt "max-size=$LOG_MAX_SIZE"
    --log-opt "max-file=$LOG_MAX_FILE"
    -v "$TABLE:$MOUNT_PATH:ro"
    "$IMAGE"
    --file "$MOUNT_PATH"
    --base-row "$BASE_ROW"
    --row-count "$ROW_COUNT"
    --port "$PORT"
    --max-rows "$MAX_ROWS"
    --threads "$THREADS"
    "$warm_flag"
  )

  # Printed before anything is checked, so the exact command can be reviewed on a box
  # where docker is not installed and nothing should be started.
  if [ "$dry" = 1 ]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  require_docker
  if container_exists; then
    die "container $CONTAINER already exists, run 'run-node.sh stop' first"
  fi
  preflight
  if ! "${DOCKER_CMD[@]}" image inspect "$IMAGE" >/dev/null 2>&1; then
    [ "$build" = 1 ] || die "image $IMAGE is missing and --no-build was given"
    do_build
  fi

  note "starting ${node:-this node}: rows [$BASE_ROW, $((BASE_ROW + ROW_COUNT))) on port $PORT"
  "${cmd[@]}" >/dev/null
  set_memory_high
  if [ "$warm_flag" = --warm ]; then
    note "started. The daemon is faulting the half in now, which takes a few minutes"
    note "from cold. Residency and progress:"
  else
    note "started without a warm pass, so rows arrive as they are asked for:"
  fi
  note "  $0 health"
  note "  ${DOCKER_CMD[*]} logs -f $CONTAINER     (the daemon logs to stderr)"
}

# Docker exposes memory.max and memory.reservation but has no flag for memory.high, and
# memory.high is the one that reclaims early instead of at the wall. Best effort: root only.
set_memory_high() {
  local cg path
  cg=$(cgroup_path) || { warn "could not find the container cgroup, memory.high not set"; return 0; }
  path="$cg/memory.high"
  if [ -w "$path" ]; then
    bytes_of "$MEM_HIGH" > "$path"
    note "memory.high = $MEM_HIGH"
  else
    warn "memory.high not set (needs root): echo $(bytes_of "$MEM_HIGH") | sudo tee $path"
  fi
}

# ---------------------------------------------------------------------------- stop
do_stop() {
  require_docker
  if ! container_exists; then
    note "container $CONTAINER does not exist"
    return 0
  fi
  # SIGTERM first: the daemon closes its listener and logs a request count on the way out.
  "${DOCKER_CMD[@]}" stop -t 15 "$CONTAINER" >/dev/null 2>&1 || true
  "${DOCKER_CMD[@]}" rm -f "$CONTAINER" >/dev/null
  note "container removed. The table's pages are still cached; 'purge' hands them back."
}

drop_cache() {
  if [ ! -f "$TABLE" ]; then
    warn "no table at $TABLE, so its pages may still be cached under some other path"
    return 0
  fi
  # Removing a cgroup does not free its page cache, it reparents the charge, so the half
  # stays cached until something else asks for it. Same fadvise the Phase 0 transfer uses.
  # It only drops pages nothing else has mapped, so run it after the container is gone.
  python3 -c "
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)" "$TABLE"
  note "page cache for $TABLE handed back to the kernel"
}

do_purge() {
  require_docker
  "${DOCKER_CMD[@]}" rm -f "$CONTAINER" >/dev/null 2>&1 || true
  "${DOCKER_CMD[@]}" rmi "$IMAGE" >/dev/null 2>&1 || true
  drop_cache
  note "removed container and image. $TABLE is untouched."
}

# ---------------------------------------------------------------------------- status
do_status() {
  require_docker
  if ! container_exists; then
    note "container $CONTAINER does not exist"
    return 1
  fi
  "${DOCKER_CMD[@]}" inspect -f \
    'state       {{.State.Status}} (exit {{.State.ExitCode}}, oom-killed {{.State.OOMKilled}}, restarts {{.RestartCount}})
started     {{.State.StartedAt}}' "$CONTAINER"

  local cg
  if cg=$(cgroup_path); then
    local cur file anon
    cur=$(cat "$cg/memory.current" 2>/dev/null || echo 0)
    file=$(awk '$1=="file"{print $2}' "$cg/memory.stat" 2>/dev/null || echo 0)
    anon=$(awk '$1=="anon"{print $2}' "$cg/memory.stat" 2>/dev/null || echo 0)
    # anon should be a few hundred MiB at most. If anon is large, the daemon is copying
    # rows into its own memory instead of serving them out of the mapping, and the
    # reasoning in deploy/README.md stops applying.
    awk -v c="$cur" -v f="$file" -v a="$anon" \
      'BEGIN { printf "memory      current %.2f GiB  file %.2f GiB  anon %.2f GiB\n",
               c/1073741824, f/1073741824, a/1073741824 }'
    printf 'limits      max %s  high %s  low %s\n' \
      "$(cat "$cg/memory.max" 2>/dev/null || echo ?)" \
      "$(cat "$cg/memory.high" 2>/dev/null || echo ?)" \
      "$(cat "$cg/memory.low" 2>/dev/null || echo ?)"
    # low  climbing means we were reclaimed despite protection, so the node is genuinely tight.
    # high climbing means our own footprint grew, which at this workload means a leak.
    # max / oom_kill climbing means the cap is wrong, not that the node is out of memory.
    printf 'events      %s\n' "$(tr '\n' ' ' < "$cg/memory.events" 2>/dev/null || echo ?)"
    printf 'pressure    %s\n' "$(awk '/^some/{print $0}' "$cg/memory.pressure" 2>/dev/null || echo ?)"
  else
    note "cgroup      not readable (container not running, or needs root)"
  fi
  printf 'node psi    %s\n' "$(awk '/^some/{print $0}' /proc/pressure/memory 2>/dev/null || echo ?)"
  note "residency is a server side number, not a cgroup one: $0 health"
}

# ---------------------------------------------------------------------------- health
do_health() {
  local target="" host port want_base=-1 want_rows=-1 node

  while [ $# -gt 0 ]; do
    case "$1" in
      --base-row)  want_base=${2:?--base-row needs a value}; shift 2 ;;
      --row-count) want_rows=${2:?--row-count needs a value}; shift 2 ;;
      -*) die "unknown option '$1'" ;;
      *)  target=$1; shift ;;
    esac
  done

  if [ -z "$target" ]; then
    node=$(detect_node || true)
    [ -n "$node" ] || die "cannot tell which node this is, pass node1, node2 or host[:port]"
    target=$node
  fi

  # Split the port off first, so the geometry a name implies survives an explicit port.
  case "$target" in
    *:*) host=${target%:*}; port=${target##*:} ;;
    *)   host=$target; port=$PORT ;;
  esac
  if [ "$want_base" -lt 0 ] && [ "$want_rows" -lt 0 ]; then
    case "$host" in
      node1|"$NODE1_ADDR") want_base=0;             want_rows=$ROWS_PER_HALF ;;
      node2|"$NODE2_ADDR") want_base=$ROWS_PER_HALF; want_rows=$ROWS_PER_HALF ;;
    esac
  fi
  host=$(node_addr "$host")

  python3 - "$host" "$port" "$want_base" "$want_rows" "$ROW_BYTES" "$PAGE_SIZE" "$RESIDENT_MIN_PCT" <<'PY'
import socket, struct, sys, time

REQ_MAGIC, RESP_MAGIC = 0x52454C50, 0x50534552
PING, STAT = 2, 3
STATUS = {0: "ok", 1: "bad magic or version", 2: "unknown op", 3: "row id out of range",
          4: "count over max_rows", 5: "internal error"}

host, port = sys.argv[1], int(sys.argv[2])
want_base, want_rows = int(sys.argv[3]), int(sys.argv[4])
row_bytes, page_size, min_pct = int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])

def recvn(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("server closed after %d of %d bytes" % (len(buf), n))
        buf += chunk
    return bytes(buf)

def call(sock, op, req_id, body_len=0):
    # 16 byte header: magic, version, op, two pad bytes, req_id, count.
    t0 = time.perf_counter()
    sock.sendall(struct.pack("<IBBHII", REQ_MAGIC, 1, op, 0, req_id, 0))
    magic, rid, status, count = struct.unpack("<IIII", recvn(sock, 16))
    if magic != RESP_MAGIC:
        raise ValueError("response magic %#010x, expected %#010x" % (magic, RESP_MAGIC))
    if rid != req_id:
        raise ValueError("req_id %d came back as %d" % (req_id, rid))
    if status != 0:
        raise ValueError("status %d (%s)" % (status, STATUS.get(status, "unknown")))
    body = recvn(sock, body_len) if body_len else b""
    return body, (time.perf_counter() - t0) * 1000.0

print("%s:%d" % (host, port))
try:
    sock = socket.create_connection((host, port), timeout=5.0)
except OSError as exc:
    print("  unreachable: %s" % exc)
    sys.exit(1)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
sock.settimeout(5.0)

try:
    pings = sorted(call(sock, PING, i + 1)[1] for i in range(5))
    body, _ = call(sock, STAT, 100, 32)
except (OSError, EOFError, ValueError, struct.error) as exc:
    print("  protocol error: %s" % exc)
    sys.exit(1)

base, rows, rb, resident, served = struct.unpack("<QQIIQ", body)
print("  ping         %.3f ms min, %.3f ms median of 5" % (pings[0], pings[2]))
print("  range        rows [%d, %d)  row_bytes %d" % (base, base + rows, rb))

rc = 0
if rb != row_bytes:
    print("  MISMATCH     row_bytes %d, expected %d" % (rb, row_bytes))
    rc = 2
if want_base >= 0 and want_rows >= 0:
    if base != want_base or rows != want_rows:
        print("  MISMATCH     expected rows [%d, %d), talking to the wrong half"
              % (want_base, want_base + want_rows))
        rc = 2
else:
    print("  note         no expected geometry for this target, range not asserted"
          " (pass --base-row and --row-count)")

# Residency is the whole point of the check. A half that has been partly reclaimed still
# answers every request correctly, it just pays a major fault per evicted row, which shows
# up at the far end as a gather that got slow for no visible reason.
expect_pages = (rows * rb + page_size - 1) // page_size
pct = 100.0 * resident / expect_pages if expect_pages else 0.0
print("  resident     %d / %d pages  (%.2f of %.2f GiB, %.1f%%)"
      % (resident, expect_pages, resident * page_size / 2**30,
         expect_pages * page_size / 2**30, pct))
print("  served       %d requests" % served)
if resident == 0:
    print("  RESIDENT 0   either the warm pass has not run yet, or mincore is blocked:"
          " check the server log for 'mincore failed'")
if pct < min_pct:
    print("  EVICTED      below %.0f%% resident, expect major faults on the gather path" % min_pct)
    rc = rc or 3
if rc == 0:
    print("  OK")
sys.exit(rc)
PY
}

# ---------------------------------------------------------------------------- main
[ $# -ge 1 ] || { usage; exit 1; }
cmd=$1; shift
case "$cmd" in
  build)  do_build "$@" ;;
  start)  do_start "$@" ;;
  stop)   do_stop "$@" ;;
  status) do_status "$@" ;;
  health) do_health "$@" ;;
  purge)  do_purge "$@" ;;
  -h|--help|help) usage ;;
  *) usage; exit 1 ;;
esac
