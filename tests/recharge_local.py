#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assertions for tests/recharge_local.sh: does op 4 actually move the page cache.

Three subcommands:

  warm    evict a table and read it back in *this* process, which is the state the
          bug happens in: the pages are resident, and charged to whoever warmed
          them rather than to the daemon that serves them
  check   drive one RECHARGE through tools/recharge_cache.py while a gather loop
          runs against the same server, and assert on both
  report  print one line of a running daemon's cache accounting, for the harness

The row pattern comes from tests/e2e_local.py so the generator and the checker
cannot drift, and so a server that returns the right shaped bytes from the wrong
offset fails instead of passing.

What the assertions are actually watching, since a RECHARGE that did nothing at
all would still answer status 0 with a healthy looking residency:

  read_bytes    the daemon's own /proc/<pid>/io. The sweep can only move the charge
                by evicting the pages for real and reading them back off the disk
                itself, so this has to grow by about the size of the table. It is
                what catches a sweep that skipped either DONTNEED.
  cgroup file   memory.stat's file counter for the cgroup the daemon runs in. This
                is the number the whole op exists to change.
  residency     mincore, through STAT, before and after. It catches a sweep that
                dropped the table and forgot to fault it back in.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

PAGE = os.sysconf("SC_PAGE_SIZE")
MIB = 1 << 20
ROW_BYTES = 160

# A gather this wide, spread evenly over the table, covers every slice of it. That
# is deliberate: it means every single request during a sweep lands partly in the
# slice that is out of cache at that moment, so the race is exercised on all of
# them rather than on the unlucky few.
GATHER_ROWS = 512

# Below this the sweep and the gather loops barely overlapped and the concurrency
# result would not mean anything, so it is a failure rather than a pass.
MIN_OVERLAPPED_GATHERS = 20

# Each loop holds its own connection, so the daemon really is copying out of the
# mapping from several threads at the moment the sweep zaps a slice's page table
# entries. One connection would only ever have one copy in flight.
GATHER_THREADS = 3


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- accounting

def proc_io(pid: int, field: str) -> int | None:
    try:
        with open(f"/proc/{pid}/io") as fh:
            for line in fh:
                key, _, value = line.partition(": ")
                if key == field:
                    return int(value)
    except OSError:
        return None
    return None


def proc_faults(pid: int) -> tuple[int, int] | None:
    """(minor, major) faults for the whole process, from /proc/<pid>/stat."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            fields = fh.read().rpartition(") ")[2].split()
    except OSError:
        return None
    # After the comm field, field 0 is state, so min_flt and maj_flt are 7 and 9.
    return int(fields[7]), int(fields[9])


def gib(n: float) -> str:
    return f"{n / (1 << 30):.3f} GiB"


# ------------------------------------------------------------------- warming

def cmd_warm(args: argparse.Namespace) -> int:
    fd = os.open(args.file, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        # Nothing has the file mapped yet, so this eviction is the one case where a
        # drop from outside works. That is the point: it puts the table in cache
        # charged to this process and to no one else.
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        read = 0
        while True:
            chunk = os.read(fd, 8 << 20)
            if not chunk:
                break
            read += len(chunk)
    finally:
        os.close(fd)
    print(f"warm  read {read} of {size} bytes of {args.file} into this process's cgroup")
    return 0 if read == size else 1


def cmd_report(args: argparse.Namespace) -> int:
    tool = load_module(Path(args.tool), "recharge_cache_report")
    charge = tool.cgroup_file_bytes(args.cgroup) if args.cgroup else None
    print(f"      daemon pid {args.pid}: read_bytes {proc_io(args.pid, 'read_bytes')}, "
          f"faults {proc_faults(args.pid)}, cgroup file "
          f"{'unavailable' if charge is None else gib(charge)}")
    return 0


# ------------------------------------------------------------- gather loop

def straddling_rows() -> tuple[int, np.ndarray]:
    """(period, row offsets within it) for rows that cross a page boundary.

    A 160 byte row does not divide a 4096 byte page, so some rows are half in one
    page and half in the next. With these two sizes it works out at 4 rows in every
    128, and that is the pattern the whole table repeats.
    """
    period = np.lcm(ROW_BYTES, PAGE) // ROW_BYTES
    offsets = np.array([r for r in range(period)
                        if (r * ROW_BYTES) % PAGE + ROW_BYTES > PAGE], dtype=np.int64)
    return int(period), offsets


class GatherLoop(threading.Thread):
    """Gather rows through the real client until told to stop, checking every byte.

    This is the assertion that matters. A sweep zaps the daemon's own page table
    entries for a slice and then evicts it, and if any of that were visible to a
    reader it would show up here as bytes from the wrong row, a short read or a
    fault the client sees as a dead connection.
    """

    def __init__(self, table, e2e, rows_total: int, seed: int, straddling: bool = False) -> None:
        super().__init__(daemon=True)
        self.table = table
        self.e2e = e2e
        self.rows_total = rows_total
        self.seed = seed
        self.straddling = straddling
        self.stop = threading.Event()
        self.spans: list[tuple[float, float]] = []
        self.rows_checked = 0
        self.error: str | None = None

    def run(self) -> None:
        rng = np.random.default_rng(self.seed)
        stride = self.rows_total // GATHER_ROWS
        base = np.arange(GATHER_ROWS, dtype=np.int64) * stride
        period, offsets = straddling_rows()
        blocks = self.rows_total // period
        while not self.stop.is_set():
            if self.straddling:
                # Only rows that sit across a page boundary. Their two halves can be
                # on either side of whatever the sweep is doing at that instant, so
                # they are where a torn read would show up if one were possible.
                ids = (rng.integers(0, blocks, GATHER_ROWS, dtype=np.int64) * period
                       + rng.choice(offsets, GATHER_ROWS))
            else:
                ids = base + rng.integers(0, stride, GATHER_ROWS, dtype=np.int64)
            t0 = time.monotonic()
            try:
                got = self.table.gather_cpu(self.e2e.as_ids(ids)).numpy()
            except Exception as exc:
                self.error = f"a gather raised {type(exc).__name__}: {exc}"
                return
            t1 = time.monotonic()
            want = self.e2e.rows_for(ids)
            if not np.array_equal(got, want):
                bad = np.flatnonzero((got != want).any(axis=1))
                i = int(bad[0])
                served = int(got[i, :8].view(np.uint64)[0])
                self.error = (
                    f"{bad.size} of {GATHER_ROWS} rows wrong during the sweep; position "
                    f"{i} asked for id {int(ids[i])} and got id {served}"
                )
                return
            self.spans.append((t0, t1))
            self.rows_checked += GATHER_ROWS


def overlapped(spans, t0: float, t1: float) -> list[float]:
    """Latencies of the gathers that started and finished inside [t0, t1]."""
    return [b - a for a, b in spans if a >= t0 and b <= t1]


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else float("nan")


def summary(e2e) -> int:
    if e2e.FAILURES:
        print(f"\n{e2e.FAILURES} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0


# ------------------------------------------------------------------- checks

def cmd_check(args: argparse.Namespace) -> int:
    here = Path(__file__).resolve().parent
    e2e = load_module(here / "e2e_local.py", "e2e_local_shared")
    tool = load_module(Path(args.tool), "recharge_cache_tool")
    client = load_module(Path(args.client), "ple_remote_recharge")

    map_bytes = args.rows_total * ROW_BYTES
    total_pages = (map_bytes + PAGE - 1) // PAGE
    expect_slices = (map_bytes + args.slice_mib * MIB - 1) // (args.slice_mib * MIB)

    expect = e2e.expect
    ok = e2e.ok

    def connect():
        return client.PLERemoteTable(
            peers=f"{args.host}:{args.port}",
            rows_total=args.rows_total,
            max_rows=args.max_rows,
            timeout=30.0,
            connect_timeout=5.0,
        )

    tables = [connect() for _ in range(GATHER_THREADS)]
    table = tables[0]
    ok(f"connect: {GATHER_THREADS} x {table.describe()}")

    try:
        # 1. The state the op exists for: the table is resident, because a shell
        # warmed it, and none of it is charged to the daemon.
        before = table.stat()[0]
        pct_before = 100.0 * before["resident_pages"] / total_pages
        expect(
            f"the table starts resident ({before['resident_pages']} / {total_pages} pages, "
            f"{pct_before:.2f}%)",
            pct_before >= 99.0,
            f"only {pct_before:.2f}% resident before the sweep, so there is nothing to move",
        )

        charge_before = tool.cgroup_file_bytes(args.cgroup) if args.cgroup else None
        if charge_before is None:
            print("SKIP  cgroup accounting is not available here, "
                  "read_bytes carries the proof on its own")
        else:
            expect(
                f"none of it is charged to the daemon's cgroup yet ({gib(charge_before)} "
                f"of {gib(map_bytes)})",
                charge_before < map_bytes // 8,
                f"the daemon's cgroup already holds {gib(charge_before)}, so this run is "
                f"not reproducing the bug",
            )

        io_before = proc_io(args.pid, "read_bytes")
        faults_before = proc_faults(args.pid)
        expect("the daemon's /proc io and stat are readable",
               io_before is not None and faults_before is not None,
               f"read_bytes {io_before}, faults {faults_before} for pid {args.pid}")

        # 2. Gathers start before the sweep does and keep going right through it.
        # One loop asks only for rows that straddle a page boundary, the others for
        # rows spread evenly over the table so that every request covers every slice.
        loops = [GatherLoop(t, e2e, args.rows_total, seed=20260914 + i, straddling=(i == 0))
                 for i, t in enumerate(tables)]
        print(f"      {len(straddling_rows()[1])} of every {straddling_rows()[0]} rows "
              f"cross a page boundary; loop 0 asks for nothing else")
        for loop in loops:
            loop.start()
        time.sleep(0.3)
        expect(f"all {len(loops)} gather loops are running before the sweep starts",
               all(loop.spans and loop.error is None for loop in loops),
               "; ".join(loop.error or f"{len(loop.spans)} gathers in 0.3 s"
                         for loop in loops))

        # A second sweep has to be turned away rather than queued, or two of them
        # would have twice the table cold at once and fault each other's slices in.
        busy = {}

        def second_recharge() -> None:
            time.sleep(0.15)
            try:
                second = tool.Client(args.host, args.port, timeout=660.0)
                busy["report"] = second.recharge(args.slice_mib)
            except IOError as exc:
                busy["error"] = str(exc)
            busy["done"] = time.monotonic()

        rejector = threading.Thread(target=second_recharge, daemon=True)
        rejector.start()

        # 3. The sweep itself, through the tool that ships with the server.
        driver = tool.Client(args.host, args.port, timeout=660.0)
        sweep_error = None
        rep = None
        t0 = time.monotonic()
        try:
            rep = driver.recharge(args.slice_mib)
        except OSError as exc:
            # A sweep that takes the daemon down with it answers nothing, and the
            # gather loops have the more interesting half of that story, so this is
            # a failed assertion rather than a traceback.
            sweep_error = f"{type(exc).__name__}: {exc}"
        t1 = time.monotonic()
        rejector.join(30.0)

        for loop in loops:
            loop.stop.set()
        for loop in loops:
            loop.join(30.0)
        spans = sorted(span for loop in loops for span in loop.spans)
        rows_checked = sum(loop.rows_checked for loop in loops)
        errors = [loop.error for loop in loops if loop.error]

        if rep is None:
            e2e.fail("the sweep answered", sweep_error or "no report and no error")
            expect(f"no gather returned wrong bytes while the sweep was running "
                   f"({len(spans)} gathers of {GATHER_ROWS} rows got through)",
                   not errors, "; ".join(errors))
            return summary(e2e)

        print(f"      sweep: {rep['slices_done']} of {rep['slices_total']} slices of "
              f"{rep['slice_pages'] * PAGE // MIB} MiB, {rep['pages_before']} -> "
              f"{rep['pages_after']} pages resident, {rep['pages_recharged']} recharged, "
              f"{rep['elapsed_ms']} ms server side, {1000 * (t1 - t0):.0f} ms round trip")

        # 4. What the sweep says it did.
        expect(f"the sweep ran more than one slice ({rep['slices_total']} of "
               f"{args.slice_mib} MiB)",
               rep["slices_total"] == expect_slices and expect_slices > 1,
               f"reported {rep['slices_total']} slices, expected {expect_slices}")
        expect("every slice was swept",
               rep["slices_done"] == rep["slices_total"],
               f"{rep['slices_done']} of {rep['slices_total']} slices, so it stopped early")
        expect(f"the slice size the caller asked for is what ran "
               f"({rep['slice_pages'] * PAGE // MIB} MiB)",
               rep["slice_pages"] == args.slice_mib * MIB // PAGE,
               f"{rep['slice_pages']} pages per slice, expected {args.slice_mib * MIB // PAGE}")
        expect(f"it claims to have recharged the whole mapping ({rep['pages_recharged']} "
               f"of {total_pages} pages)",
               rep["pages_recharged"] == total_pages,
               f"recharged {rep['pages_recharged']} of {total_pages} pages")

        # 5. What actually happened to the cache.
        pct_after = 100.0 * rep["pages_after"] / total_pages
        expect(f"the table is still resident afterwards ({rep['pages_after']} / "
               f"{total_pages} pages, {pct_after:.2f}%)",
               pct_after >= 99.0,
               f"{pct_after:.2f}% resident after the sweep: the slices were dropped and "
               f"not faulted back in")

        io_after = proc_io(args.pid, "read_bytes")
        read_delta = (io_after or 0) - (io_before or 0)
        faults_after = proc_faults(args.pid)
        expect(f"the daemon re-read the table off the disk itself ({gib(read_delta)} of "
               f"{gib(map_bytes)})",
               read_delta >= map_bytes // 2,
               f"the daemon only read {gib(read_delta)} during a sweep of {gib(map_bytes)}, "
               f"so the pages were never really evicted and the charge cannot have moved")
        if faults_before and faults_after:
            print(f"      daemon faults over the sweep: "
                  f"{faults_after[0] - faults_before[0]} minor, "
                  f"{faults_after[1] - faults_before[1]} major")

        if charge_before is not None:
            charge_after = tool.cgroup_file_bytes(args.cgroup)
            moved = (charge_after or 0) - charge_before
            expect(f"the page cache is now charged to the daemon's cgroup "
                   f"({gib(charge_before)} -> {gib(charge_after or 0)}, "
                   f"{gib(moved)} moved)",
                   moved >= (map_bytes * 4) // 5,
                   f"only {gib(moved)} moved onto the daemon's cgroup out of "
                   f"{gib(map_bytes)}, which is the one thing this op is for")

        # 6. The concurrency result, which is the assertion a live model depends on.
        inside = overlapped(spans, t0, t1)
        expect(f"no gather returned wrong bytes while the sweep was running "
               f"({len(inside)} gathers of {GATHER_ROWS} rows inside it)",
               not errors,
               "; ".join(errors))
        expect(f"gathers really did overlap the sweep ({len(inside)} of "
               f"{len(spans)} entirely inside it)",
               len(inside) >= MIN_OVERLAPPED_GATHERS,
               f"only {len(inside)} gathers fell inside the sweep, so this proves nothing")
        outside = overlapped(spans, 0.0, t0) or overlapped(spans, t1, time.monotonic())
        if inside:
            print(f"      gather latency: median {1000 * percentile(inside, 50):.2f} ms, "
                  f"p99 {1000 * percentile(inside, 99):.2f} ms, worst "
                  f"{1000 * max(inside):.2f} ms during the sweep; median "
                  f"{1000 * percentile(outside, 50):.2f} ms outside it "
                  f"({rows_checked} rows checked in total)")

        # 7. The second sweep, which should have been refused while the first ran.
        if "error" in busy and "status 7" in busy["error"]:
            ok("a second sweep during the first is refused with status 7")
        elif "report" in busy:
            overlapping = busy.get("done", 0.0) <= t1
            expect("a second sweep during the first is refused with status 7",
                   not overlapping,
                   "a second sweep ran to completion while the first was still going, "
                   "so two of them can have the table cold at once")
            if not overlapping:
                print("      note: the second sweep landed after the first had finished, "
                      "so status 7 was not exercised")
        else:
            expect("a second sweep during the first is refused with status 7",
                   False, busy.get("error", "the second RECHARGE never answered"))

        # 8. A slice size over the cap is refused rather than clamped.
        try:
            tool.Client(args.host, args.port, timeout=30.0).recharge(8192)
            expect("a slice over the 4096 MiB cap is refused with status 6", False,
                   "the server accepted it")
        except IOError as exc:
            expect("a slice over the 4096 MiB cap is refused with status 6",
                   "status 6" in str(exc), str(exc))

        # 9. And the table still reads back correctly, all of it, after the sweep.
        chunk = args.max_rows
        wrong = 0
        first_bad = ""
        for start in range(0, args.rows_total, chunk):
            ids = np.arange(start, min(start + chunk, args.rows_total), dtype=np.int64)
            got = table.gather_cpu(e2e.as_ids(ids)).numpy()
            want = e2e.rows_for(ids)
            if not np.array_equal(got, want):
                bad = np.flatnonzero((got != want).any(axis=1))
                wrong += bad.size
                if not first_bad:
                    i = int(bad[0])
                    first_bad = (f"id {int(ids[i])} came back as id "
                                 f"{int(got[i, :8].view(np.uint64)[0])}")
        expect(f"every one of the {args.rows_total} rows still reads back correctly "
               f"after the sweep",
               wrong == 0, f"{wrong} rows wrong, first: {first_bad}")

        after = table.stat()[0]
        expect("the server still answers STAT with the same geometry",
               (after["base_row"], after["row_count"], after["row_bytes"])
               == (before["base_row"], before["row_count"], before["row_bytes"]),
               f"{after}")
    finally:
        for t in tables:
            t.close()

    return summary(e2e)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("warm", help="evict a table and read it in, in this process")
    w.add_argument("--file", required=True)
    w.set_defaults(func=cmd_warm)

    r = sub.add_parser("report", help="one line of a daemon's cache accounting")
    r.add_argument("--tool", required=True)
    r.add_argument("--pid", type=int, required=True)
    r.add_argument("--cgroup", default="")
    r.set_defaults(func=cmd_report)

    c = sub.add_parser("check", help="drive one RECHARGE and assert on it")
    c.add_argument("--client", required=True, help="path to client/ple_remote.py")
    c.add_argument("--tool", required=True, help="path to tools/recharge_cache.py")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, required=True)
    c.add_argument("--rows-total", type=int, required=True)
    c.add_argument("--max-rows", type=int, required=True)
    c.add_argument("--slice-mib", type=int, required=True)
    c.add_argument("--pid", type=int, required=True, help="the daemon, for /proc")
    c.add_argument("--cgroup", default="", help="the daemon's cgroup directory, if any")
    c.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
