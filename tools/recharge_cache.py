#!/usr/bin/env python3
"""Move a running row server's page cache onto its own cgroup, without restarting it.

Page cache is charged to whichever cgroup first faults a page in. If the table was warmed
from a shell before the daemon started, the pages belong to user.slice, the container's
memory.low protects nothing, and the kernel reclaims the table out from under a server that
still reports a healthy residency. The symptom is major faults and gathers that got about
10x slower for no visible reason.

This sends the server one RECHARGE (docs/protocol.md, op 4). The server then works through
its mapping a slice at a time, dropping each slice out of cache and immediately faulting it
back in through its own mapping, so the charge moves to the cgroup the daemon runs in. One
slice is out of cache at a time, so a live client sees a ripple rather than a stall.

Why the version this replaces did not work
------------------------------------------
It dropped the cache from out here, with posix_fadvise(DONTNEED) on its own descriptor, and
then pulled the rows back through the daemon with ordinary GATHERs. It read as if it worked
and it did almost nothing, because posix_fadvise cannot evict a page that a running process
has mapped. The daemon maps the whole half, so every resident page, which is exactly the
set whose charge needed moving, stayed where it was. The only pages that moved were the ones
already missing. The eviction has to happen inside the daemon, after it has zapped its own
page table entries for that slice, and that ordering is what op 4 exists for.

Run it on the node if you want the cgroup numbers: they come from that node's /proc. The
RECHARGE itself works from anywhere that can reach the server.
"""
import argparse
import os
import socket
import struct
import sys
import time

MAGIC_REQ, MAGIC_RESP = 0x52454C50, 0x50534552
VERSION = 1
OP_STAT, OP_RECHARGE = 3, 4
HDR, STAT_BODY, RECHARGE_BODY, ROW_BYTES = 16, 32, 40, 160
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
GIB = 1 << 30

STATUS = {
    1: "bad magic or version",
    2: "unknown op",
    3: "row id out of range",
    4: "count over max_rows",
    5: "internal error",
    6: "slice size out of range",
    7: "a recharge is already running on that server",
}


class Client:
    def __init__(self, host, port, timeout):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.req_id = 0

    def _recv(self, n):
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        while got < n:
            k = self.sock.recv_into(mv[got:], n - got)
            if not k:
                raise IOError(f"server closed after {got} of {n} bytes")
            got += k
        return buf

    def _call(self, op, arg, body_len):
        self.req_id = (self.req_id + 1) & 0xFFFFFFFF
        self.sock.sendall(struct.pack("<IBBHII", MAGIC_REQ, VERSION, op, arg, self.req_id, 0))
        magic, rid, status, count = struct.unpack("<IIII", self._recv(HDR))
        if magic != MAGIC_RESP or rid != self.req_id:
            raise IOError(f"bad response magic=0x{magic:08x} req_id={rid}")
        if status:
            raise IOError(f"server status {status} ({STATUS.get(status, 'unknown')})")
        if count:
            raise IOError(f"header-only op answered with {count} rows")
        return self._recv(body_len)

    def stat(self):
        base, rows, row_bytes, resident, served = struct.unpack("<QQIIQ", self._call(OP_STAT, 0, STAT_BODY))
        return {"base_row": base, "row_count": rows, "row_bytes": row_bytes,
                "resident_pages": resident, "served_requests": served}

    def recharge(self, slice_mib):
        body = self._call(OP_RECHARGE, slice_mib, RECHARGE_BODY)
        before, after, recharged, done, total, slice_pages, ms = struct.unpack("<QQQIIII", body)
        return {"pages_before": before, "pages_after": after, "pages_recharged": recharged,
                "slices_done": done, "slices_total": total, "slice_pages": slice_pages,
                "elapsed_ms": ms}


def mapping_cgroups(path):
    """(pid, cgroup directory) for every process that has `path` mapped.

    Matched on device and inode rather than on the path, because the daemon usually runs in a
    container and knows the file as /ple/half.bin rather than as whatever the bind mount
    points at here. Reading another user's maps needs root, so an empty list is as likely to
    be permissions as it is to be a server that is not running.
    """
    found = []
    try:
        st = os.stat(path)
    except OSError:
        return found
    want_dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}"
    want_ino = str(st.st_ino)

    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/maps") as fh:
                if not any(len(f) >= 5 and f[3] == want_dev and f[4] == want_ino
                           for f in (line.split() for line in fh)):
                    continue
            with open(f"/proc/{entry}/cgroup") as fh:
                for line in fh:
                    ident, _, rest = line.rstrip("\n").partition("::")
                    if ident == "0":
                        found.append((int(entry), "/sys/fs/cgroup" + rest))
        except OSError:
            continue
    return found


def cgroup_file_bytes(cgdir):
    """The cgroup's page cache charge, which is the number this whole exercise is about."""
    try:
        with open(os.path.join(cgdir, "memory.stat")) as fh:
            for line in fh:
                key, _, value = line.partition(" ")
                if key == "file":
                    return int(value)
    except OSError:
        return None
    return None


def default_table():
    """Reading another process's cgroup needs root, and sudo resets HOME to /root, so
    resolve the table against the invoking user rather than the effective one."""
    home = os.environ.get("HOME", "")
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        try:
            import pwd
            home = pwd.getpwnam(sudo_user).pw_dir
        except (ImportError, KeyError):
            pass
    return os.path.join(home or os.path.expanduser("~"), "ple", "half.bin")


def residency(resident, total):
    return f"{resident} / {total} pages ({100.0 * resident / total if total else 0.0:.2f}%)"


def main():
    ap = argparse.ArgumentParser(description="drive a row server's RECHARGE op")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--slice-mib", type=int, default=0,
                    help="how much of the table is out of cache at any one moment "
                         "(0 uses the server's default of 1024)")
    ap.add_argument("--file", default=default_table(),
                    help="the table on this node, used only to find the daemon's cgroup")
    ap.add_argument("--timeout", type=float, default=660.0,
                    help="socket timeout; a sweep answers only when it is finished, and the "
                         "server caps itself at 600 s")
    ap.add_argument("--dry-run", action="store_true", help="show STAT and stop")
    a = ap.parse_args()

    if not 0 <= a.slice_mib <= 0xFFFF:
        sys.exit("--slice-mib rides in a u16 header field, so it has to fit in one")

    client = Client(a.host, a.port, a.timeout)
    stat = client.stat()
    if stat["row_bytes"] != ROW_BYTES:
        sys.exit(f"server serves {stat['row_bytes']} byte rows, expected {ROW_BYTES}")
    total_pages = (stat["row_count"] * stat["row_bytes"] + PAGE_SIZE - 1) // PAGE_SIZE

    print(f"server    {a.host}:{a.port}  rows [{stat['base_row']}, "
          f"{stat['base_row'] + stat['row_count']})  served {stat['served_requests']} requests")
    print(f"before    {residency(stat['resident_pages'], total_pages)}")

    mapped = mapping_cgroups(a.file)
    cgdir, charge_before = None, None
    if not mapped:
        print(f"cgroup    not readable from here (no process maps {a.file}, or this needs root)")
    else:
        pid, cgdir = mapped[0]
        if len(mapped) > 1:
            # One row server per table per node is the deployment. More than one means a
            # leftover is still holding the mapping, and which of them these numbers
            # describe is a coin toss, so say so rather than pick quietly.
            print(f"cgroup    {len(mapped)} processes map {a.file}: "
                  f"{', '.join(str(p) for p, _ in mapped)}; reporting the first")
        charge_before = cgroup_file_bytes(cgdir)
        if charge_before is None:
            print(f"cgroup    pid {pid} {cgdir}, memory.stat not readable")
        else:
            print(f"cgroup    pid {pid} {cgdir}")
            print(f"          file cache charged there: {charge_before / GIB:.2f} GiB, all files "
                  f"(the table alone is {total_pages * PAGE_SIZE / GIB:.2f} GiB)")

    if a.dry_run:
        return 0

    t0 = time.time()
    try:
        rep = client.recharge(a.slice_mib)
    except IOError as exc:
        sys.exit(f"recharge failed: {exc}")

    print(f"recharge  {rep['slices_done']} of {rep['slices_total']} slices, "
          f"{rep['slice_pages'] * PAGE_SIZE // (1 << 20)} MiB each, "
          f"{rep['pages_recharged'] * PAGE_SIZE / GIB:.2f} GiB moved in "
          f"{rep['elapsed_ms'] / 1000.0:.1f} s (round trip {time.time() - t0:.1f} s)")
    print(f"after     {residency(rep['pages_after'], total_pages)}")

    # Residency cannot tell a working sweep from one that moved nothing: the pages are
    # resident either way, which is the whole reason this op exists. The cgroup charge is
    # the only number that answers the question, so when it is readable it decides.
    table_bytes = total_pages * PAGE_SIZE
    charge_failed = False
    if cgdir is not None:
        charge_after = cgroup_file_bytes(cgdir)
        if charge_after is not None:
            moved = "" if charge_before is None else f", was {charge_before / GIB:.2f}"
            print(f"cgroup    file cache charged to the daemon now: "
                  f"{charge_after / GIB:.2f} GiB{moved}")
            if charge_after < table_bytes // 2:
                charge_failed = True
                print(f"\nthe sweep reported success but the daemon's cgroup still holds only "
                      f"{charge_after / GIB:.2f} GiB\nof file cache against a "
                      f"{table_bytes / GIB:.2f} GiB table. The pages are resident but still "
                      f"charged\nelsewhere, so memory.low does not protect them and this has "
                      f"not fixed anything")

    if rep["slices_done"] < rep["slices_total"]:
        print("\nthe sweep ran out of time before it finished. Running it again is safe, but it "
              "starts\nfrom the beginning, so look at what made the disk this slow first")
        return 1
    short = total_pages - rep["pages_after"]
    if short:
        # deploy/run-node.sh health calls anything under 95% evicted, so this agrees with it
        # rather than inventing a second threshold.
        pct = 100.0 * rep["pages_after"] / total_pages
        print(f"\n{short} pages went again between being touched and being counted "
              f"({pct:.2f}% resident). A few is reclaim racing the sweep; a lot means the node "
              f"is short of memory")
        return 1 if pct < 95.0 or charge_failed else 0
    return 1 if charge_failed else 0


if __name__ == "__main__":
    sys.exit(main())
