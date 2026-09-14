# Deploying a row server onto node1 and node2

Each node serves one half of the PLE n-gram table out of its own page cache. The half is
already on the node as a flat file at `~/ple/half.bin`, exported and checksum-verified in
Phase 0, so nothing here re-derives geometry or re-reads the safetensors.

| | base_row | row_count | file size |
|---|---|---|---|
| node1 (172.16.0.28) | 0 | 160,000,768 | 25,600,122,880 B (23.842 GiB) |
| node2 (172.16.0.33) | 160,000,768 | 160,000,768 | 25,600,122,880 B (23.842 GiB) |

**node1 is a live k3s control-plane node with embedded etcd on it.** Everything in this
directory is sized around that one fact. Read the section on memory before changing a limit,
because the interesting part is not the numbers, it is that the memory in question is page
cache rather than anonymous memory, and that changes what a limit does.

## What actually gets run

The daemon is `rowserverd`, built by `server/Makefile` from `server/rowserverd.c`. Its whole
command line is:

```
rowserverd --file PATH --base-row N --row-count N [options]

  --file PATH       flat file of row_count x 160 byte rows
  --base-row N      global id of this file's first row
  --row-count N     rows in this file
  --port N          listen port (default 9000)
  --max-rows N      largest gather accepted, per request (default 131072)
  --threads N       concurrent connections, one thread each (default 8)
  --warm            fault the whole file in at startup (default)
  --no-warm         skip the warm pass and let rows arrive on demand
  --help
```

There is nothing else. In particular there is no `--table`, no `--listen` and no bind
address option of any kind. `run-node.sh start --dry-run` prints the exact `docker run` line
it would use, which is the quickest way to check that this file and the code still agree.

Four things about the daemon that the rest of this document leans on:

- **It maps the table read-only and serves rows straight out of the mapping.** It does not
  copy the table into its own memory and it never calls `mlock`. The only sizeable
  allocation is one response buffer per connection, `max_rows * 160` bytes, so 20 MiB each
  at the default.
- **It faults the mapping in itself at startup**, unless you pass `--no-warm`. This matters
  for accounting and not just for warm-up, for the reason in
  [where the page cache is charged](#where-the-page-cache-is-charged).
- **`STAT` reports `resident_pages`** from the daemon's own `mincore` of the mapping, which
  is what makes `run-node.sh health` able to tell a warm server from an evicted one.
- **It writes its log to stderr.** Only `--help` goes to stdout. Docker captures both
  streams, so `docker logs ple-row` shows it; under systemd it goes to the journal.

It also refuses to start if the file is not exactly `row_count * 160` bytes, which is the
cheap guard against a server pointed at the wrong half. A half that does not hold the rows it
claims would serve wrong embeddings and nothing downstream would notice.

### Port 9000

9000 is the daemon's `DEFAULT_PORT`, the `EXPOSE` in `server/Dockerfile`, the client's
`DEFAULT_PORT` in `client/ple_remote.py`, and the default in `bench/`. If you move it, move it
in `PLE_REMOTE_PEERS` (or `QWEN4EXP_PLE_REMOTE_PEERS`) as well, since the client dials
`host:port` pairs and defaults the port when one is not given.

### What it binds

The daemon binds `INADDR_ANY`. There is no option to narrow that, and with `--network host`
it is the host's network namespace, so the server is reachable on every address the node
holds, flannel and CNI interfaces included.

That is a statement about what the code does, not an endorsement. If you want it narrowed,
the two honest choices are a host firewall rule or a change to the server:

```sh
# allow the two peers and gx10, drop the rest, on node1
sudo nft add rule inet filter input tcp dport 9000 ip saddr \
  { 172.16.0.28, 172.16.0.33, 172.16.0.43 } accept
sudo nft add rule inet filter input tcp dport 9000 drop
```

Adding a `--bind` option to `rowserverd` would be a few lines in `listen_on()`, and it would
need a line in `docs/protocol.md` under Limits. Nothing in this directory pretends it exists.

## Prerequisites on each node

- `~/ple/half.bin` present and checksum-verified by `tools/validate_halves.sh`. The Phase 0
  gate is mandatory rather than optional, because of the shard permutation described in the
  top-level README.
- Docker, for the container path. node2 has no compiler, which does not matter: the image
  builds the binary inside a `debian:bookworm-slim` stage and ships a static binary on
  `scratch`, so no host toolchain is involved on either node.
- `python3`, which the health check and the cache-drop in `purge` use. Stdlib only.

### The table has to be readable by uid 65534

`server/Dockerfile` ends with `USER 65534:65534`, and Docker does not remap that to you. The
bind-mounted `half.bin` keeps the ownership and mode it has on the host, so the daemon opens
it as `nobody`:

```sh
chmod o+r ~/ple/half.bin
chmod o+x ~ ~/ple
```

`run-node.sh start` checks the file and every directory above it before starting anything and
tells you the exact `chmod` if it is wrong. It checks plain POSIX bits only, so ACLs or
AppArmor could still get in the way. The systemd install does not have this problem, since it
runs as `coenie`.

## Deploying with Docker

Copy the repo to each node and run the script there. It has to run on the node because the
container bind-mounts that node's own `half.bin`.

```sh
for h in node1 node2; do
  ssh "$h" 'mkdir -p ~/ple-row-server'
  rsync -a --exclude .git ~/workspace/ple-row-server/ "$h:~/ple-row-server/"
done
```

Build the image once, somewhere that is not node1, and move it across. A `docker build` pulls
layers and compiler output through the page cache, which is the single most reliable way to
evict the table and to make etcd's fsyncs slow at the same time. The build context is the
`server/` directory, since that is where the Dockerfile lives and all it copies is
`Makefile` and `rowserverd.c`.

```sh
ssh node2 '~/ple-row-server/deploy/run-node.sh build'
ssh node2 'docker save ple-row-server' | ssh node1 'docker load'
```

Then start each half. The script works out which one this node owns from its address, since
the netbooted nodes carry generated hostnames rather than `node1` and `node2`.

```sh
ssh node2 '~/ple-row-server/deploy/run-node.sh start --no-build --yes'
ssh node1 '~/ple-row-server/deploy/run-node.sh start --no-build --yes'
```

Start node2 first. If something is wrong with the image or the arguments, you find out on the
node that is not holding the cluster together.

`--yes` is the k3s acknowledgement. Both nodes are control-plane members running etcd, so
`start` stops and asks before it does anything there. Over `ssh` there is no terminal to ask
on, so without `--yes` (or `YES=1`) it fails with that message rather than going ahead
quietly. That is the point: the gate has to fire in the way the deployment is actually
driven, not only when someone happens to be sitting at a prompt.

`--dry-run` prints the exact `docker run` command and touches nothing, so it works on a box
with no Docker and on one where nothing should be started:

```sh
ssh node1 '~/ple-row-server/deploy/run-node.sh start --dry-run'
```

`start` refuses to go ahead if the table file is missing, if it is not exactly
`row_count * 160` bytes, if the container user cannot read it, if the port is already taken,
or if k3s is running and nobody has acknowledged it. It warns, without refusing, if
`MemAvailable` is below the size of the half or if `MEM_MAX` is smaller than the range being
served.

Geometry and sizes are derived from the arguments rather than hardcoded, so the documented
overrides work:

```sh
# serve a small test range out of a partial export
MEM_MAX=1g MEM_HIGH=900m MEM_LOW=800m CONTAINER=ple-row-test \
  run-node.sh start --table ~/ple/test.bin --base-row 0 --row-count 1000000 \
    --port 9100 --max-rows 4096 --threads 2 --yes
```

## Verifying

```sh
ssh node1 '~/ple-row-server/deploy/run-node.sh health node1'
ssh node2 '~/ple-row-server/deploy/run-node.sh health node2'
```

A good result looks like this:

```
172.16.0.28:9000
  ping         0.184 ms min, 0.211 ms median of 5
  range        rows [0, 160000768)  row_bytes 160
  resident     6250030 / 6250030 pages  (23.84 of 23.84 GiB, 100.0%)
  served       148213 requests
  OK
```

The health check opens one connection with `TCP_NODELAY`, sends five `PING`s and one `STAT`,
and asserts three things: that the server answers at all, that it is serving the half you
think it is, and that the table is still resident. Exit codes are 0 for healthy, 1 for
unreachable or a protocol error, 2 for a geometry mismatch, and 3 for a table that has been
partly evicted. A partly evicted table still answers every request correctly, it just pays a
major page fault per evicted row, so without this check it surfaces at the far end as a gather
that got slow for no apparent reason. `RESIDENT_MIN_PCT` sets the threshold, default 95.

The geometry assertion follows the host, not the spelling of the target. `node1`,
`172.16.0.28` and `172.16.0.28:9000` all assert rows `[0, 160000768)`. For anything else, pass
what you expect:

```sh
run-node.sh health 172.16.0.28:9100 --base-row 0 --row-count 1000000
```

Without that, the check says plainly that it is not asserting a range rather than passing on a
server that could be serving anything.

`resident_pages` is a count of 4 KiB pages. 6,250,030 of them is a whole half. If the nodes
ever move to a kernel with a different base page size, pass `PAGE_SIZE=`. A `resident 0` line
means either the warm pass has not finished yet or `mincore` is being blocked, which is worth
checking in the server log before believing the table has been evicted.

The health check is stdlib Python and can be run from anywhere that can reach the node,
including from gx10, so the same command works as a client-side probe:

```sh
watch -n 60 '~/ple-row-server/deploy/run-node.sh health node1'
```

For the local view of a node, including the numbers that show whether the memory is behaving:

```sh
ssh node1 'sudo ~/ple-row-server/deploy/run-node.sh status'
```

```
state       running (exit 0, oom-killed false, restarts 0)
memory      current 23.98 GiB  file 23.85 GiB  anon 0.12 GiB
limits      max 27917287424  high 26843545600  low 25769803776
events      low 0 high 0 max 0 oom 0 oom_kill 0
pressure    some avg10=0.00 avg60=0.00 avg300=0.00 total=0
node psi    some avg10=0.00 avg60=0.11 avg300=0.09 total=48210231
```

`anon` being small is the proof that the daemon is serving out of the mapping rather than
copying rows into its own memory. If `anon` is large, none of the reasoning below applies any
more. `file` being the whole table is the expected case but not a guaranteed one, for the
reason in the next section: pages that were already cached before the container started are
charged somewhere else.

## Deploying without Docker

`ple-rowserver.service` is the plain host install. It is the better option on node1 if you
want the full set of memory controls, because Docker has no flag for `memory.high`, and the
script's workaround (writing the cgroup file after start) needs root and does not survive a
container restart.

Build the binary somewhere with a compiler. node2 has none, so either build on node1, or lift
the static binary straight out of the image, which is the same binary:

```sh
make -C server                 # produces server/rowserverd
# or, from the image
id=$(docker create ple-row-server) && docker cp "$id:/rowserverd" ./rowserverd && docker rm "$id"
```

Install the binary, the protocol doc the unit points at, and the unit itself:

```sh
sudo install -m 0755 rowserverd /usr/local/bin/rowserverd
sudo install -D -m 0644 docs/protocol.md /usr/local/share/ple-row-server/protocol.md
sudo install -m 0644 deploy/ple-rowserver.service /etc/systemd/system/
```

The unit is named `ple-rowserver.service` and the binary it runs is `rowserverd`. The doc
install is not decoration: `Documentation=` in the unit points at that path, and a
`Documentation=` pointing at a file nobody installed is worse than none.

`/etc/default/ple-rowserver` on node1:

```sh
PLE_TABLE=/home/coenie/ple/half.bin
PLE_BASE_ROW=0
PLE_ROW_COUNT=160000768
```

and on node2:

```sh
PLE_TABLE=/home/coenie/ple/half.bin
PLE_BASE_ROW=160000768
PLE_ROW_COUNT=160000768
```

`PLE_PORT`, `PLE_MAX_ROWS` and `PLE_THREADS` have defaults in the unit (9000, 131072, 8) and
only need to be in this file if you are changing them. The three above have no defaults on
purpose: without them `rowserverd` prints its usage and exits 2, which is a better failure
than serving the wrong half.

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now ple-rowserver
systemctl show ple-rowserver -p MemoryLow -p MemoryHigh -p MemoryMax -p MemoryCurrent
journalctl -u ple-rowserver -f     # the daemon logs to stderr, so this is where it lands
```

The unit's sandbox allows `mincore` explicitly on top of `@system-service`, which does not
include it. Without that line the syscall filter returns `EPERM`, the daemon logs
`mincore failed` and `STAT` reports zero resident pages, so a perfectly warm table reads as
fully evicted. Verified with `systemd-analyze syscall-filter @system-service` on systemd 255.

`Type=exec` only tells you that the process started, not that the half is faulted in. If the
daemon learns to send `sd_notify` with `READY=1` after the warm pass, switch to `Type=notify`
and add a `TimeoutStartSec=` generous enough to cover it. Then `systemctl start` would block
until the half is actually usable, which is worth having.

## Why the limits are what they are

### The table is page cache, not anonymous memory

This is the thing that makes the rest of it safe, and it is worth being precise about which
parts of it are certain.

Certain, and the reason a cap is tolerable here at all:

- The daemon maps `half.bin` read-only, never writes to it and never `mlock`s it, so the
  23.842 GiB it occupies is clean, file-backed page cache. Clean file pages are reclaimed by
  dropping them: no writeback, no swap, no waiting.
- Under cgroup v2 that page cache is charged to a cgroup, and it counts towards
  `memory.max` exactly like anonymous memory does.
- When a charge would exceed `memory.max`, the kernel reclaims inside the cgroup before it
  considers an OOM kill. With 23.8 GiB of reclaimable file pages under the limit there is
  always something to drop, so for this workload the cap trims the table and the process
  keeps running. An OOM kill would need the anonymous part alone to exceed the cap, which at
  this workload means a leak.

So an undersized cap here is a latency bug rather than a crash. What it is not is free: the
gather pattern is uniform across the whole half by construction, so a cap below the table size
does not settle into a warm subset. It refaults continuously for as long as the server is
used, and those faults land on the same NVMe as etcd's WAL.

Not certain, and to be measured rather than asserted:

- **Whether the reclaim ordering works out the way we want.** `memory.low` biases global
  reclaim away from us and `memory.max` biases it towards us, but the actual page the kernel
  picks also depends on LRU age and where the pressure came from. The evidence is
  `memory.events` and `STAT`'s `resident_pages` over a real workload, not this document.
- **Whether the cap ever prevents anything.** Its defensible purpose is containment and
  observability: a bounded cgroup makes the row server's footprint attributable, so it cannot
  look like unbounded cache growth from the node's point of view. We have not measured a case
  where it stopped harm.
- **That Docker's `--memory-reservation` lands on `memory.low`.** That is what runc does on
  cgroup v2 as far as we know, and `run-node.sh status` prints `memory.low` for the container
  so you can confirm it instead of trusting it. If it reads 0, the protection is not there.

### Where the page cache is charged

The kernel charges a file page to the cgroup whose task first faults it in, and the charge
stays with that cgroup until the page is reclaimed. Two consequences that are easy to trip
over:

- Warming the table from a shell (`dd`, `cat`, `vmtouch`) charges it to that shell's cgroup.
  The pages would then sit outside the server's limit and outside its reclaim protection while
  still occupying the node's RAM. The daemon's own warm pass is what makes the numbers here
  mean what they say, which is why `--no-warm` is a deliberate choice rather than a default.
- If `half.bin` is still cached from `tools/transfer_halves.sh` or a previous run, those pages
  keep their old charge. `status` can then show `file` well below 23.8 GiB while `health`
  reports 100% resident, and both are telling the truth. If you want the accounting to line
  up, drop the cache first (`run-node.sh purge` does the same `posix_fadvise`) and let the
  daemon fault it in.

### `--memory 26g` / `MemoryMax=26G`

23.842 GiB of table plus 2.158 GiB of headroom. The headroom covers in-flight responses, which
are the only meaningful anonymous allocation the daemon makes: at `max_rows` of 131,072 a
single response buffer is 131072 x 160 = 20 MiB, one per connection, and the default of 8
connection threads bounds that at 160 MiB plus the id buffers.

`memory.events` tells you which kind of pressure you are looking at. `max` rising with
`oom_kill` at zero is the table being trimmed, which is expected under pressure. `oom_kill`
rising is a bug.

If you serve a range that is not a full half, set `MEM_MAX`, `MEM_HIGH` and `MEM_LOW` to
match. The script warns when `MEM_MAX` is below the range you asked it to serve, but it does
not guess a size for you.

### `--memory-swap 26g` / `MemorySwapMax=0`

Setting `--memory-swap` equal to `--memory` gives the container no swap at all. The nodes run
without swap today, so this only matters if that changes: the table never swaps regardless,
being file-backed, but a response buffer could, and a gather that lands on a swapped-out
buffer costs a disk read on the critical path. On node1 there is a second reason, which is
that k3s wants swap off anyway. Confirm with `swapon --show`.

### `--memory-reservation 24g` / `MemoryLow=24G`

This is reclaim protection, not a limit. While our usage is below it, reclaim driven from
outside the cgroup is supposed to skip us, so a large image pull, a log rotation or a build on
the node does not quietly evict the table underneath us.

24 GiB sits just above the 23.842 GiB table, so the protection covers the mapping and nothing
else. Protecting our slop as well would be protection taken from etcd for no gain.

`memory.low` is best effort by design. When the node is genuinely out of memory the kernel
breaches the protection and reclaims from us rather than OOM-killing something, which is
exactly the behaviour we want: we yield, etcd keeps its memory. **Do not replace this with
`MemoryMin`.** `MemoryMin` is the hard version, and a hard guarantee here would convert node
memory pressure into an OOM kill somewhere else on the box, which on node1 means k3s.

One thing that is easy to get wrong: `memory.low` has no effect on reclaim triggered by our
own `memory.high` or `memory.max`. That is a different path in the kernel. Protection is about
pressure from outside, limits are about pressure from inside.

### `MemoryHigh=25G`

The throttle before the wall. There is 1.15 GiB of slack above the table at this setting, so
in normal operation we never reach it; reaching it means our anonymous footprint grew, not
that the table did. Treat it as a leak alarm that reclaims and slows the process down rather
than killing it.

Docker has no flag for this. `run-node.sh start` writes it to the container's cgroup after
start if it has permission and prints the command to run manually if it does not. Two
caveats worth knowing: it needs root, and a container restart creates a fresh cgroup, so the
value is gone after Docker restarts the container on failure. The systemd unit just declares
it and does not have either problem.

### `--oom-score-adj 1000` / `OOMScoreAdjust=1000`

If the node runs out of memory in spite of everything above, the kernel OOM killer picks a
victim by score, and 1000 is the maximum bias towards being chosen. k3s and etcd run with a
strongly negative adjustment. Killing the row server costs a warm-up and some slow gathers;
killing etcd costs the cluster. The ordering should never be in question.

`--oom-kill-disable` is the opposite of this and must not be used.

### `--restart on-failure:3`, `Restart=on-failure` with `StartLimitBurst=3`

A few attempts for a transient crash, then stay down. An `always` policy is the wrong choice on
a node under memory pressure: each restart faults 23.842 GiB back in, which makes the pressure
worse, which gets the process killed again. That loop is how a single misbehaving service turns
into a node-wide outage.

If the container has stopped, check whether it was an OOM kill before restarting it:

```sh
docker inspect -f '{{.State.OOMKilled}} {{.State.ExitCode}} {{.RestartCount}}' ple-row
```

`true` there means the node had a memory problem, and restarting the row server is the last
thing you should do until you know what caused it.

### `--cpu-shares 512` / `CPUWeight=50`, and no CPU quota

Weight, not quota. `--cpus` and `CPUQuota=` are enforced by the CFS bandwidth controller over a
100 ms period, so a burst past the quota parks the process for the remainder of that period.
100 ms is longer than an entire engine step, so a quota would introduce exactly the kind of
tail-latency stall this whole project exists to avoid. Weight only takes effect when the CPU is
actually contended and never inserts a stall of its own. Half the default weight means that
when node1 is busy, etcd wins and the gather gets slower, which is the right way round.

The `mincore` walk behind `STAT` is the one part of the daemon that burns a core for a
noticeable stretch, a few milliseconds over 23.8 GiB of page tables. It is off the gather path,
but it is a reason not to poll `health` in a tight loop.

### `--network host`

Bridge networking puts NAT and, for published ports, the userland proxy in front of every
request. The gather budget is about 1 ms for a whole decode step, so a per-packet copy is not
affordable. Host networking also keeps `TCP_NODELAY` end to end rather than through a proxy
that has its own buffering.

The cost is that the daemon binds port 9000 on the host's network namespace, on every address
the node holds. See [what it binds](#what-it-binds).

### `-v ~/ple/half.bin:/ple/half.bin:ro`

Read-only, and the daemon maps it read-only. Beyond the obvious, this is what guarantees the
pages stay clean: a dirty page has to be written back before it can be reclaimed, which turns
cheap reclaim into disk I/O competing with etcd's WAL on the same device.

### `--ulimit memlock=0:0` / `LimitMEMLOCK=0`

`mlock` on the table would make it unreclaimable, and unreclaimable memory on a control-plane
node is the one failure this deployment cannot absorb: the kernel would have to find its memory
elsewhere, and elsewhere is etcd. A zero limit makes any attempt fail loudly instead of
succeeding quietly. If the daemon is ever rewritten against io_uring with registered buffers,
that also needs `RLIMIT_MEMLOCK` on older kernels, so raise it deliberately rather than
discovering it as a startup failure.

### `--read-only`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--pids-limit`

The daemon opens one file, writes to sockets and writes to stderr, so none of this should cost
anything. The image is `FROM scratch` with a single static binary in it, so there is nothing
else in there to protect either way. The systemd unit does the equivalent with
`ProtectSystem=strict`, `ProtectHome=read-only` (it needs to read `/home/coenie/ple/half.bin`),
an empty capability bounding set, and a system call filter.

### `--log-driver json-file --log-opt max-size=10m --log-opt max-file=3`

Docker's json-file driver does not rotate by default. On a k3s node a disk filling with
container logs also stops etcd from writing, so the bound is cheap insurance. The driver is
named explicitly rather than left to the daemon's default, because `max-size` and `max-file`
are rejected outright by some other drivers, and a rejected option is a container that does
not start. At steady state the daemon logs a handful of lines in total, so 30 MiB is generous;
the bound is there for a pathological case such as a client hammering a malformed header,
where one line is logged per bad request.

## What actually happens under pressure

Three scenarios, in increasing order of seriousness.

**Something else on the node wants a few GiB.** Reclaim starts, sees our cgroup is under its
`memory.low`, and skips us. It takes the memory from ordinary page cache elsewhere instead.
Nothing changes for us. `memory.events` stays at zero.

**Something wants far more than the node has spare.** Reclaim comes back for us anyway, because
`memory.low` is best effort. It drops clean table pages with no writeback and no swap.
`memory.events` shows `low` climbing, `STAT`'s `resident_pages` falls below 6,250,030, and the
health check exits 3. Gather latency rises because the evicted rows now come off NVMe on the
critical path. Nothing is killed, and the table repopulates on its own as those rows get
gathered again.

**The node is genuinely out of memory.** The OOM killer runs and picks the process with the
highest score, which we have arranged to be the row server. It dies, Docker retries it up to
three times with backoff and then leaves it down. The inference side loses its gathers and
fails loudly rather than stalling, per the protocol's no-reconnect rule. etcd is untouched.

The case that is not on this list is the row server causing any of it, which is what the memory
cap is for.

## What to watch on node1

node1 holds the k3s embedded etcd. etcd is sensitive to two things this deployment could
plausibly affect: memory pressure, and disk write latency on the device holding its WAL.

Memory pressure, as a single number:

```sh
ssh node1 'cat /proc/pressure/memory'
```

`some avg10` sitting above a few percent means the kernel is spending real time reclaiming. At
idle it should be zero or near it. The per-cgroup version in `run-node.sh status` tells you
whether that pressure is ours.

Whether we have been reclaimed:

```sh
ssh node1 'sudo ~/ple-row-server/deploy/run-node.sh status'   # memory.events, low / high / max
~/ple-row-server/deploy/run-node.sh health node1               # resident_pages
```

etcd's own view, which is the one that matters:

```sh
ssh node1 'kubectl get --raw "/readyz?verbose"' | grep etcd
ssh node1 'sudo journalctl -u k3s -p warning --since -15m'
```

k3s logs `apply request took too long` and `slow fdatasync` when etcd is struggling. If those
appear after this deployment and were not there before, the likely cause is the table being
refaulted from disk while etcd is trying to fsync its WAL, not CPU contention. Confirm with
`iostat -x 2` on the device holding both.

Two habits worth keeping:

- Do not build images on node1. Use `docker save` and `docker load`, as above.
- Change one node at a time and check etcd between them.

## Stopping and removing it

Stop the container, keeping the image and the table:

```sh
ssh node1 '~/ple-row-server/deploy/run-node.sh stop'
```

That sends `SIGTERM` first and gives the daemon 15 seconds, which is more than it needs: it
closes the listener, logs how many requests it served and exits. Then the container is removed,
so the next `start` is a clean one. The table's pages stay cached.

Remove everything this deployment put on the node, apart from the table itself:

```sh
ssh node1 '~/ple-row-server/deploy/run-node.sh purge'
```

`purge` removes the container and the image, then hands the page cache back with
`posix_fadvise(DONTNEED)`, the same call the Phase 0 transfer uses. The cache part is not
cosmetic. Removing a cgroup does not free its page cache, it reparents the charge, so without
this step the node keeps 23.842 GiB cached under the parent cgroup after the server is gone.
Those pages are reclaimable and will be given up under pressure, but until then `free` shows
the node as much fuller than it is, which makes the next person's capacity decision wrong.
`DONTNEED` only drops pages nothing has mapped, which is why it runs after the container is
gone rather than before.

If you only want the cache back and want to keep the image:

```sh
ssh node1 '~/ple-row-server/deploy/run-node.sh stop'
ssh node1 'python3 -c "
import os
fd = os.open(os.path.expanduser(\"~/ple/half.bin\"), os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)"'
```

For the systemd install:

```sh
sudo systemctl disable --now ple-rowserver
sudo systemctl reset-failed ple-rowserver
sudo rm -f /etc/systemd/system/ple-rowserver.service /etc/default/ple-rowserver
sudo rm -f /usr/local/bin/rowserverd
sudo rm -rf /usr/local/share/ple-row-server
sudo systemctl daemon-reload
python3 -c "
import os
fd = os.open(os.path.expanduser('~/ple/half.bin'), os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)"
```

The table file itself is left alone in both cases. Deleting it reclaims 23.842 GiB of disk but
costs a fresh export from gx10 (about 150 seconds per half over 2.5 GbE) and a re-run of the
checksum gate in `tools/validate_halves.sh`, which is mandatory rather than optional given the
shard permutation described in the top-level README.

## Symptoms and where to look

| symptom | first thing to check |
|---|---|
| health exits 1, unreachable | container state (`status`), then the server log: `docker logs ple-row` |
| health exits 2, geometry mismatch | the node is serving the other half; check `--base-row` against `/etc/default/ple-rowserver` or `start --dry-run` |
| health exits 3, resident below 95% | `memory.events`: `low` means node pressure took it, `max` means our own cap did |
| health prints `resident 0` on a warm server | `mincore` blocked by the syscall filter; the server log says `mincore failed` |
| container exits immediately | the log line names it: a size mismatch on the half, a port in use, or `open: Permission denied` for uid 65534 |
| gather latency up, residency fine | not memory; check the network path, and remember gx10 is currently two switch hops away |
| container keeps restarting | `docker inspect -f '{{.State.OOMKilled}}' ple-row`; if true, stop restarting it and find out what used the memory |
| `status` shows `file` well below the table | pages charged to another cgroup from an earlier warm; see [where the page cache is charged](#where-the-page-cache-is-charged) |
| etcd warnings on node1 | `iostat -x 2` on the WAL device, and whether the table is being refaulted (residency falling) |
