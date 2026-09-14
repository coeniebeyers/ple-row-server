#!/usr/bin/env python3
"""
Dependency-free probe for the row servers: correctness first, then latency.

Speaks the wire protocol with nothing but the standard library, so it runs on the
inference box, which has no torch outside the serving container. That matters because
the only latency number worth having is the one measured from where the client will
actually live.

Correctness is checked against the ORIGINAL safetensors rather than against the halves,
by shelling out to ple_export_half.py for individual rows. The halves were themselves
verified against that file, so this closes the loop end to end: safetensors -> export ->
transfer -> mmap -> wire -> here.
"""
import argparse, json, socket, statistics, struct, subprocess, sys, time
import numpy as np

MAGIC_REQ, MAGIC_RESP = 0x52454C50, 0x50534552
VERSION = 1
OP_GATHER, OP_PING, OP_STAT = 1, 2, 3
HDR, STAT_BODY, ROW_BYTES = 16, 32, 160
STATUS = {0: "ok", 1: "bad magic/version", 2: "unknown op", 3: "id out of range",
          4: "count over max_rows", 5: "internal"}


class Peer:
    def __init__(self, addr):
        host, _, port = addr.partition(":")
        self.addr = addr
        self.sock = socket.create_connection((host, int(port or 9000)), timeout=30)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.req_id = 0
        st = self.stat()
        self.base, self.count = st["base_row"], st["row_count"]
        self.stat_info = st

    def _hdr(self, op, count):
        self.req_id = (self.req_id + 1) & 0xFFFFFFFF
        return struct.pack("<IBBHII", MAGIC_REQ, VERSION, op, 0, self.req_id, count)

    def _recv(self, n):
        buf = bytearray(n)
        mv, got = memoryview(buf), 0
        while got < n:
            k = self.sock.recv_into(mv[got:], n - got)
            if not k:
                raise IOError(f"{self.addr}: closed after {got} of {n} bytes")
            got += k
        return buf

    def _read_hdr(self):
        magic, req_id, status, count = struct.unpack("<IIII", self._recv(HDR))
        if magic != MAGIC_RESP:
            raise IOError(f"{self.addr}: response magic 0x{magic:08x}")
        if req_id != self.req_id:
            raise IOError(f"{self.addr}: req_id {req_id}, expected {self.req_id}")
        if status:
            raise IOError(f"{self.addr}: status {status} ({STATUS.get(status, '?')})")
        return count

    def stat(self):
        self.sock.sendall(self._hdr(OP_STAT, 0))
        self._read_hdr()
        base, rows, rb, resident, served = struct.unpack("<QQIIQ", self._recv(STAT_BODY))
        return {"base_row": base, "row_count": rows, "row_bytes": rb,
                "resident_pages": resident, "served_requests": served}

    def ping(self):
        self.sock.sendall(self._hdr(OP_PING, 0))
        self._read_hdr()

    def send_gather(self, ids):
        """ids is a uint32 ndarray. tobytes beats struct.pack, which would have to
        unpack 65,536 arguments onto the interpreter stack for one prefill request."""
        self.sock.sendall(self._hdr(OP_GATHER, ids.size)
                          + ids.astype("<u4", copy=False).tobytes())

    def read_gather(self, n):
        got = self._read_hdr()
        if got != n:
            raise IOError(f"{self.addr}: returned {got} rows for {n}")
        return self._recv(n * ROW_BYTES)


def gather(peers, ids):
    """Send every request before reading any, so the peers work concurrently.

    Partition and scatter are vectorised deliberately. A per-row Python loop here costs
    more than the network does and would be measuring the probe rather than the server.
    """
    ids = np.asarray(ids, dtype=np.uint32)
    masks = [(ids >= p.base) & (ids < p.base + p.count) for p in peers]
    if int(sum(m.sum() for m in masks)) != ids.size:
        raise ValueError("some ids belong to no peer")
    subs = [ids[m] for m in masks]
    for p, sub in zip(peers, subs):
        if sub.size:
            p.send_gather(sub)
    out = np.empty((ids.size, ROW_BYTES), dtype=np.uint8)
    for p, sub, m in zip(peers, subs, masks):
        if sub.size:
            raw = p.read_gather(sub.size)
            out[m] = np.frombuffer(raw, dtype=np.uint8).reshape(-1, ROW_BYTES)
    return out


def truth_row(exporter, model_file, row_id):
    cmd = [sys.executable, exporter, "--start", str(row_id), "--count", "1"]
    if model_file:
        cmd += ["--file", model_file]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def pct(xs, q):
    return statistics.quantiles(xs, n=1000, method="inclusive")[q - 1] if len(xs) > 2 else max(xs)


def run(peers, rows, iters, rows_total, label, seed=7):
    rnd = np.random.default_rng(seed)
    for _ in range(max(3, iters // 10)):                      # warmup, excluded
        gather(peers, rnd.integers(0, rows_total, rows, dtype=np.uint32))
    lat = []
    t_start = time.perf_counter()
    for _ in range(iters):
        ids = rnd.integers(0, rows_total, rows, dtype=np.uint32)
        t0 = time.perf_counter()
        gather(peers, ids)
        lat.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - t_start
    mb = rows * ROW_BYTES * iters / 1e6
    return {"workload": label, "rows": rows, "iters": iters,
            "p50": statistics.median(lat), "p90": pct(lat, 900), "p99": pct(lat, 990),
            "max": max(lat), "min": min(lat), "mb_s": mb / wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peers", default="172.16.0.28:9000,172.16.0.33:9000")
    ap.add_argument("--exporter", default="/home/coenie/bin/ple_export_half.py")
    ap.add_argument("--model-file")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args()

    peers = [Peer(x.strip()) for x in a.peers.split(",")]
    peers.sort(key=lambda p: p.base)
    rows_total = sum(p.count for p in peers)

    print("peers")
    for p in peers:
        s = p.stat_info
        res = s["resident_pages"] * 4096 / (1 << 30)
        print(f"  {p.addr:22s} rows [{p.base}, {p.base + p.count})  "
              f"row_bytes {s['row_bytes']}  resident {res:.2f} GiB  served {s['served_requests']}")
    cover = 0
    for p in peers:
        if p.base != cover:
            sys.exit(f"FATAL: gap or overlap at row {cover} (peer starts at {p.base})")
        cover += p.count
    print(f"  coverage contiguous: [0, {cover})  rows_total {rows_total}")

    t0 = time.perf_counter()
    for p in peers:
        p.ping()
    print(f"  ping round trip: {(time.perf_counter() - t0) * 1000 / len(peers):.3f} ms avg\n")

    if not a.skip_verify:
        half = peers[0].count
        probe_ids = [0, 1, half - 1, half, half + 1, rows_total - 1,
                     12345678, 250001200]          # last one is inside a permuted shard
        got = gather(peers, probe_ids)
        bad = 0
        for n, rid in enumerate(probe_ids):
            mine = got[n].tobytes()
            ref = truth_row(a.exporter, a.model_file, rid)
            ok = mine == ref
            bad += not ok
            print(f"  row {rid:>10}  {'MATCH' if ok else 'MISMATCH'}")
        if bad:
            sys.exit(f"FATAL: {bad} rows do not match the safetensors")
        print(f"  all {len(probe_ids)} rows match the original safetensors\n")

    results = []
    for label, rows, iters in (("decode", 512, a.iters),
                               ("prefill", 65536, max(10, a.iters // 10)),
                               ("sweep-2048", 2048, a.iters),
                               ("sweep-8192", 8192, max(20, a.iters // 4))):
        r = run(peers, rows, iters, rows_total, label)
        results.append(r)
        print(f"  {r['workload']:<12} rows={r['rows']:>6} n={r['iters']:>4}  "
              f"p50 {r['p50']:7.3f}  p90 {r['p90']:7.3f}  p99 {r['p99']:7.3f}  "
              f"max {r['max']:8.3f} ms   {r['mb_s']:7.1f} MB/s")

    if a.json:
        json.dump({"peers": [p.stat_info | {"addr": p.addr} for p in peers],
                   "results": results}, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
