# Build:
#   docker build -t ple-row-server server
#
# Run (node1's half; node2 is --base-row 160000768):
#   docker run -d --name ple-row --network host --memory 26g --read-only \
#     --log-opt max-size=10m --log-opt max-file=3 \
#     -v $HOME/ple/half.bin:/ple/half.bin:ro ple-row-server \
#     --file /ple/half.bin --base-row 0 --row-count 160000768
#
# Keep the log-opts. The daemon writes to stderr and the default json-file driver
# keeps everything it is given, on nodes that are k3s control-plane members: etcd's
# WAL and a full disk do not survive each other. The daemon holds every line a peer
# can trigger to one a second per site, so 30 MB across three files is generous.
# deploy/run-node.sh passes the same two options, and the driver as well.
#
# --memory does less than it looks. Page cache is charged to whichever cgroup first
# faults a page in, so the limit governs the mapped half only when this container is
# the one that reads it in: those pages then count against the 26g and the kernel
# trims them from this cgroup before the box goes into global reclaim. Pages already
# resident from another reader (a previous run of this container, the copy that put
# half.bin here, a checksum pass) stay charged where they were faulted in and this
# container maps them without its own usage moving, limit or no limit. Either way
# the pages are file backed and never mlocked, so reclaim costs latency and nothing
# else. half.bin needs to be world readable: the daemon runs unprivileged.

FROM debian:bookworm-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev make \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Makefile rowserverd.c ./
RUN make static && strip rowserverd

# Static binary, so there is nothing else to ship and nothing in the image to exploit.
FROM scratch
COPY --from=build /src/rowserverd /rowserverd
USER 65534:65534
EXPOSE 9000
ENTRYPOINT ["/rowserverd"]
