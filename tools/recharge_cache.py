#!/usr/bin/env python3
"""Move a running row server's page cache onto its own cgroup, without restarting it.

Page cache is charged to whichever cgroup first faults a page in. If the table was warmed
from a shell before the daemon started, the pages belong to user.slice, the container's
memory.low protects nothing, and the kernel reclaims the table out from under a server that
still reports a healthy residency. The symptom is major faults and slow gathers with no
obvious cause.

This fixes it in place: drop a slice of the file from cache, then immediately make the
DAEMON read those same rows, so it faults them back in under its own cgroup. Working one
slice at a time keeps only a small part of the table cold at any moment, so a live client
sees a ripple rather than a stall.

Run it ON the node, so the sweep goes over loopback instead of the wire.
"""
import argparse, array, os, socket, struct, sys, time

MAGIC_REQ, MAGIC_RESP = 0x52454C50, 0x50534552
VERSION, OP_GATHER, OP_STAT = 1, 1, 3
HDR, STAT_BODY, ROW_BYTES = 16, 32, 160


class Client:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=120)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.req_id = 0

    def _recv(self, n):
        buf = bytearray(n); mv = memoryview(buf); got = 0
        while got < n:
            k = self.sock.recv_into(mv[got:], n - got)
            if not k:
                raise IOError(f"closed after {got} of {n}")
            got += k
        return buf

    def _hdr(self, op, count):
        self.req_id = (self.req_id + 1) & 0xFFFFFFFF
        return struct.pack("<IBBHII", MAGIC_REQ, VERSION, op, 0, self.req_id, count)

    def _read_hdr(self):
        magic, rid, status, count = struct.unpack("<IIII", self._recv(HDR))
        if magic != MAGIC_RESP or rid != self.req_id:
            raise IOError(f"bad response magic=0x{magic:08x} req_id={rid}")
        if status:
            raise IOError(f"server status {status}")
        return count

    def stat(self):
        self.sock.sendall(self._hdr(OP_STAT, 0)); self._read_hdr()
        base, rows, rb, resident, served = struct.unpack("<QQIIQ", self._recv(STAT_BODY))
        return base, rows, rb, resident, served

    def gather(self, first, count):
        # array.tobytes rather than struct.pack(f"<{count}I", *ids): the latter would push
        # 131,072 arguments onto the interpreter stack for one request.
        ids = array.array("I", range(first, first + count))
        if sys.byteorder != "little":
            ids.byteswap()
        self.sock.sendall(self._hdr(OP_GATHER, count) + ids.tobytes())
        n = self._read_hdr()
        self._recv(n * ROW_BYTES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.expanduser("~/ple/half.bin"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--slice-mib", type=int, default=1024,
                    help="how much of the table is cold at any one moment")
    ap.add_argument("--rows-per-request", type=int, default=131072)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    c = Client(a.host, a.port)
    base, rows, rb, resident, served = c.stat()
    if rb != ROW_BYTES:
        sys.exit(f"server serves {rb} byte rows, expected {ROW_BYTES}")
    total_pages = (rows * rb + 4095) // 4096
    print(f"server rows [{base}, {base + rows})  resident {resident}/{total_pages} "
          f"({resident * 100 // total_pages}%)  served {served}")
    if a.dry_run:
        return

    fd = os.open(a.file, os.O_RDONLY)
    rows_per_slice = (a.slice_mib * 1024 * 1024) // rb
    done, t0 = 0, time.time()
    try:
        while done < rows:
            n = min(rows_per_slice, rows - done)
            # Drop just this slice, then fault it straight back through the daemon.
            os.posix_fadvise(fd, done * rb, n * rb, os.POSIX_FADV_DONTNEED)
            sent = 0
            while sent < n:
                k = min(a.rows_per_request, n - sent)
                c.gather(base + done + sent, k)
                sent += k
            done += n
            pct = done * 100 // rows
            print(f"\r  recharged {done * rb / (1 << 30):6.2f} GiB ({pct:3d}%)", end="", flush=True)
    finally:
        os.close(fd)

    _, _, _, resident, served = c.stat()
    print(f"\ndone in {time.time() - t0:.0f}s  resident {resident}/{total_pages} "
          f"({resident * 100 // total_pages}%)  served {served}")


if __name__ == "__main__":
    main()
