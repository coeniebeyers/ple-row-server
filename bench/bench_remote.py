#!/usr/bin/env python3
"""
Latency and correctness bench for the PLE row servers.

Standalone by design: no vLLM, no model, nothing that touches the inference box.
The row servers have to be measured and proven correct before anything goes near
a forward pass, otherwise an integration that comes out slow has two suspects
instead of one.

  bench_remote.py                                    decode + prefill, table output
  bench_remote.py --mode sweep --json > results.json
  bench_remote.py --mode verify                      run on a node, against its own half.bin

Correctness is the part that matters most. Both halves were checksummed against
the safetensors in Phase 0, so a node's half.bin holds exactly the rows that
node's server is supposed to answer with, and comparing server bytes against that
file needs nothing from gx10.

Exit status is 0 when everything checked out and 1 otherwise: a verification that
mismatched, a peer that would not answer, or a command line that does not make
sense, each with a line saying which.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError as exc:
    # The client imports both at module scope, so there is no torch free way to
    # drive it and nothing to be gained from pretending otherwise here.
    raise SystemExit(f"bench_remote.py needs numpy and torch, the two the client needs: {exc}")

ROW_BYTES = 160
ROWS_TOTAL = 320_001_536
DEFAULT_MAX_ROWS = 131_072        # the server's default --max-rows, per request
DEFAULT_PORT = 9000               # rowserverd's default port
DEFAULT_PEERS = "172.16.0.28,172.16.0.33"
DEFAULT_REFERENCE = Path.home() / "ple" / "half.bin"

DECODE_ROWS = 512                 # steady state decode step at 16 rows per token
PREFILL_ROWS = 65_536             # a 4096 token prefill chunk at 16 rows per token
SWEEP_ROWS = (16, 64, 256, 512, 1024, 4096, 16_384, 65_536, 131_072)
DEFAULT_ITERS = {"decode": 200, "prefill": 50, "sweep": 50}
VERIFY_SAMPLES = 8192
ORDER_SAMPLES = 256               # the single row leg costs one round trip per id

GIB = 1 << 30
# resident_pages comes over the wire without a unit. 4 KiB is right on the peers,
# and the output labels the estimate so a reader on 64 KiB pages is not misled.
ASSUMED_PAGE = 4096


# ---------------------------------------------------------------- client binding

def load_client(path: Path):
    spec = importlib.util.spec_from_file_location("ple_remote_bench", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load client module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, "PLERemoteTable", None)
    if cls is None:
        raise SystemExit(f"{path}: no PLERemoteTable in the client module")
    return cls


def connect(cls, addrs: list[str], max_rows: int, timeout: float):
    """Open the table, and turn the client's own assertions into a one line exit.

    The client STATs every peer at construction and refuses to build unless the
    ranges tile the table with no gap and no overlap, so a server on the wrong
    half, a missing peer or a wrong row_bytes fails here rather than part way
    through a run. Repeating those checks in the bench would only be a second copy
    of them to keep in step.
    """
    try:
        return cls(
            peers=",".join(addrs),
            rows_total=ROWS_TOTAL,
            max_rows=max_rows,
            timeout=timeout,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"cannot open the table on {', '.join(addrs)}: {exc}")


def gather(table, ids: list[int]) -> np.ndarray:
    """Rows for ids as a uint8 [n, ROW_BYTES] array. Untimed, for verification.

    gather_cpu hands back a view of the client's staging buffer and the next call
    overwrites it, so anything kept past the call has to be a copy.
    """
    return table.gather_cpu(torch.tensor(ids, dtype=torch.int64)).numpy().copy()


def residency_notes(stats: list[dict]) -> list[str]:
    """A cold half is worth shouting about but is not a failure: a server that has
    just started has not faulted its whole file in yet."""
    notes = []
    for s in stats:
        owned = s["row_count"] * s["row_bytes"]
        resident = s["resident_pages"] * ASSUMED_PAGE
        if owned and resident < 0.95 * owned:
            notes.append(
                f"{s['peer']}: {resident / owned * 100:.1f}% of its half is resident, "
                "so some of these rows came off disk"
            )
    return notes


def print_peers(stats: list[dict], notes: list[str], out) -> None:
    print("peers (STAT as reported by each server)", file=out)
    for s in stats:
        owned = s["row_count"] * s["row_bytes"]
        resident = s["resident_pages"] * ASSUMED_PAGE
        frac = resident / owned * 100 if owned else 0.0
        print(
            f"  {s['peer']:<20} rows {s['base_row']:>12,} .. "
            f"{s['base_row'] + s['row_count']:>12,}  row_bytes {s['row_bytes']:>3}"
            f"  resident {resident / GIB:6.2f} of {owned / GIB:6.2f} GiB"
            f" ({frac:5.1f}%, 4 KiB pages)  served {s['served_requests']:,}",
            file=out,
        )
    print(f"  coverage {sum(s['row_count'] for s in stats):,} of {ROWS_TOTAL:,} rows", file=out)
    for msg in notes:
        print(f"  note {msg}", file=out)
    print(file=out)


# ------------------------------------------------------------------- measurement

def percentile(ordered: list[int], p: float) -> int:
    """Nearest rank over the raw samples.

    No interpolation: at 50 iterations an interpolated p99 is an average of the
    two worst samples, which is the opposite of what a tail number is for.
    """
    k = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[k - 1]


def summarise(name: str, rows: int, samples_ns: list[int]) -> dict:
    ordered = sorted(samples_ns)
    payload = rows * ROW_BYTES
    p50 = percentile(ordered, 50)
    return {
        "workload": name,
        "rows": rows,
        "iters": len(ordered),
        "bytes_per_gather": payload,
        "min_ms": ordered[0] / 1e6,
        "p50_ms": p50 / 1e6,
        "p90_ms": percentile(ordered, 90) / 1e6,
        "p99_ms": percentile(ordered, 99) / 1e6,
        "max_ms": ordered[-1] / 1e6,
        "mean_ms": sum(ordered) / len(ordered) / 1e6,
        "mb_s": payload / (p50 / 1e9) / 1e6,
        "us_per_row": p50 / 1e3 / rows,
    }


HEADER = (
    f"{'workload':<9}{'rows':>8}{'iters':>7}{'p50 ms':>9}{'p90 ms':>9}"
    f"{'p99 ms':>9}{'max ms':>9}{'MB/s':>9}{'us/row':>9}"
)


def format_run(r: dict) -> str:
    return (
        f"{r['workload']:<9}{r['rows']:>8}{r['iters']:>7}{r['p50_ms']:>9.3f}"
        f"{r['p90_ms']:>9.3f}{r['p99_ms']:>9.3f}{r['max_ms']:>9.3f}"
        f"{r['mb_s']:>9.1f}{r['us_per_row']:>9.3f}"
    )


def run_workload(table, name: str, rows: int, iters: int, warmup: int,
                 gen: torch.Generator) -> dict:
    """Time gather_cpu, which is the call the model's forward pass sits on.

    The warmup iterations are not just cache warming: the client allocates its
    pinned staging buffer on the first gather of a size, so an unwarmed first
    sample would carry that allocation.
    """
    samples: list[int] = []
    want = (rows, ROW_BYTES)
    for i in range(warmup + iters):
        ids = torch.randint(0, ROWS_TOTAL, (rows,), generator=gen, dtype=torch.int64)
        t0 = time.perf_counter_ns()
        got = table.gather_cpu(ids)
        dt = time.perf_counter_ns() - t0
        # The client already fails a peer that returns fewer rows than it asked
        # for. This is the cheap second look, off the clock, because a short
        # response would otherwise post the best numbers in the file.
        if tuple(got.shape) != want:
            raise SystemExit(f"{name}: gather returned {tuple(got.shape)}, expected {want}")
        if i >= warmup:
            samples.append(dt)
    return summarise(name, rows, samples)


# ------------------------------------------------------------------ verification

def sample_ids(rnd: random.Random, lo: int, hi: int, n: int) -> list[int]:
    """A random draw plus the cases a plausible server gets wrong.

    Duplicates and a descending run ride in the same request as the random ids,
    because the protocol promises rows back in request order with duplicates
    kept, and a server that sorts or dedupes internally still looks correct when
    it is only ever handed a sorted unique id list.
    """
    span = hi - lo
    if span < 8:
        raise SystemExit(f"reference range is {span} rows, too small to sample")
    ids = [lo, lo + 1, lo + span // 2, hi - 1]
    ids += [lo + span // 3] * 3
    ids += sorted((rnd.randrange(lo, hi) for _ in range(64)), reverse=True)
    ids += [rnd.randrange(lo, hi) for _ in range(max(0, n - len(ids)))]
    return ids


def read_reference(fd: int, base: int, ids: list[int]) -> np.ndarray:
    # Deliberately no posix_fadvise afterwards: run on a peer, the reference is
    # the same file the server holds resident, and dropping those pages is the
    # damage this project exists to avoid.
    out = np.empty((len(ids), ROW_BYTES), dtype=np.uint8)
    for k, row in enumerate(ids):
        buf = os.pread(fd, ROW_BYTES, (row - base) * ROW_BYTES)
        if len(buf) != ROW_BYTES:
            raise SystemExit(f"reference file: short read at row {row}")
        out[k] = np.frombuffer(buf, dtype=np.uint8)
    return out


def compare(ids: list[int], got: np.ndarray, want: np.ndarray,
            max_report: int = 5) -> tuple[int, list[dict]]:
    """Returns the mismatch count and the first few, first 16 bytes of each."""
    if got.shape != want.shape:
        raise SystemExit(f"gather returned {got.shape}, expected {want.shape}")
    bad = np.flatnonzero((got != want).any(axis=1))
    samples = [
        {"index": int(k), "row_id": int(ids[k]),
         "got": got[k][:16].tobytes().hex(), "want": want[k][:16].tobytes().hex()}
        for k in bad[:max_report]
    ]
    return int(bad.size), samples


def print_mismatches(total: int, samples: list[dict], out) -> None:
    for s in samples:
        print(f"  MISMATCH row {s['row_id']} at index {s['index']}: "
              f"got {s['got']} want {s['want']}", file=out)
    if total > len(samples):
        print(f"  and {total - len(samples):,} more", file=out)


def host_is_local(peer: str) -> bool:
    """True when this machine holds the address that peer answers on.

    Bind is the test: a socket can only bind an address that is on this box. It is
    how a run on a node works out that the half.bin next to it is the half its own
    server is serving.
    """
    host = peer.rsplit(":", 1)[0]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
        return True
    except OSError:
        return False


def resolve_verify_base(path: Path, file_rows: int, stats: list[dict],
                        given: int | None) -> tuple[int, str]:
    """Work out which global row the reference file starts at.

    Guessing wrong here is not a quiet error, it compares one half's server against
    the other half's file and reports every row as a mismatch, so the only
    inference allowed is one that lands on exactly one peer.
    """
    if given is not None:
        return given, "--verify-base"
    candidates = [s for s in stats if s["row_count"] == file_rows]
    if len(candidates) == 1:
        return candidates[0]["base_row"], f"the only peer serving {file_rows:,} rows"
    if len(candidates) > 1:
        local = [s for s in candidates if host_is_local(s["peer"])]
        if len(local) == 1:
            return local[0]["base_row"], f"{local[0]['peer']} is this machine"
    options = ", ".join(f"--verify-base {s['base_row']} for {s['peer']}" for s in stats)
    raise SystemExit(
        f"{path} holds {file_rows:,} rows, which does not say on its own which global "
        f"rows they are. Pass one of: {options}, or the row this file starts at."
    )


def check_reference_range(path: Path, base: int, file_rows: int, stats: list[dict]) -> None:
    """Refuse a reference that claims rows the peers do not serve.

    A mistyped base aims the sample at rows nobody owns, and the gather then fails
    on routing rather than on anything to do with the file. The client asserted at
    connect time that the peers tile their range with no gap, so the lowest base
    and the highest end are the whole of what is served.
    """
    lo = min(s["base_row"] for s in stats)
    hi = max(s["base_row"] + s["row_count"] for s in stats)
    if base < lo or base + file_rows > hi:
        raise SystemExit(
            f"{path} would cover global rows {base:,} .. {base + file_rows:,}, "
            f"but the peers serve {lo:,} .. {hi:,}. Either --verify-base {base} is "
            "wrong or this file is not one of the halves these peers are holding."
        )


def verify_against_file(table, stats: list[dict], path: Path, base: int | None,
                        n_samples: int, rnd: random.Random, out) -> dict:
    """Compare what a server answers against the same rows read from its half.

    This is the check that means something. The halves were checksummed against
    the safetensors in Phase 0, so a row read out of half.bin is the row the
    checkpoint holds, and a server that indexes its mapping wrongly or hands back
    rows in the wrong order has nowhere to hide.
    """
    size = path.stat().st_size
    if size % ROW_BYTES:
        raise SystemExit(f"{path}: {size} bytes is not a whole number of {ROW_BYTES} byte rows")
    file_rows = size // ROW_BYTES
    base, why = resolve_verify_base(path, file_rows, stats, base)
    check_reference_range(path, base, file_rows, stats)

    print(f"reference {path}", file=out)
    print(f"  covers global rows {base:,} .. {base + file_rows:,} "
          f"({size:,} bytes, base from {why})", file=out)

    ids = sample_ids(rnd, base, base + file_rows, n_samples)
    fd = os.open(path, os.O_RDONLY)
    try:
        want = read_reference(fd, base, ids)
    finally:
        os.close(fd)
    total, bad = compare(ids, gather(table, ids), want)

    hit = {s["peer"]: 0 for s in stats}
    for row in ids:
        for s in stats:
            if s["base_row"] <= row < s["base_row"] + s["row_count"]:
                hit[s["peer"]] += 1
    print(f"  checked {len(ids):,} rows, {total:,} mismatched", file=out)
    print("  ids landed on: " + ", ".join(f"{a} {n:,}" for a, n in hit.items()), file=out)
    print_mismatches(total, bad, out)
    if len(hit) > 1 and min(hit.values()) == 0:
        print("  the reference only spans one peer's range, so only that peer was checked",
              file=out)
    print(file=out)
    return {
        "reference": str(path),
        "reference_base_row": base,
        "reference_rows": file_rows,
        "checked": len(ids),
        "mismatches": total,
        "mismatch_samples": bad,
        "ids_per_peer": hit,
        "ok": total == 0,
    }


def order_ids(stats: list[dict], n: int, rnd: random.Random) -> list[int]:
    """One request holding every shape a batching bug hides behind."""
    ids: list[int] = []
    for s in stats:
        top = s["base_row"] + s["row_count"] - 1
        ids += [s["base_row"], top, s["base_row"]]
    ids += sorted((rnd.randrange(ROWS_TOTAL) for _ in range(min(64, n))), reverse=True)
    ids += [ids[0]] * 3
    ids += [rnd.randrange(ROWS_TOTAL) for _ in range(max(0, n - len(ids)))]
    return ids


def verify_order(table, stats: list[dict], n_samples: int, rnd: random.Random, out) -> dict:
    """Check a batched gather against the same rows fetched one at a time.

    Both legs come from the same servers, so this cannot catch a server that
    indexes its own half wrongly; that is what the reference file is for. What it
    does catch is a server that sorts or dedupes a request internally, or a client
    that scatters the peers' responses back in the wrong order, neither of which
    shows up when the request is a sorted unique id list. Every id spans the full
    peer set, so it also exercises the routing across the split boundary.
    """
    ids = order_ids(stats, n_samples, rnd)
    batched = gather(table, ids)
    repeat = gather(table, ids)
    one_at_a_time = np.empty_like(batched)
    for k, row in enumerate(ids):
        one_at_a_time[k] = gather(table, [row])[0]

    total, bad = compare(ids, batched, one_at_a_time)
    stable = bool(np.array_equal(batched, repeat))
    dupes = len(ids) - len(set(ids))

    print("order and duplicates (no reference needed)", file=out)
    print(f"  {len(ids):,} ids in one request, {dupes:,} of them repeats, "
          f"boundary rows and a descending run included", file=out)
    print(f"  batched against one row per request: {total:,} mismatched", file=out)
    print_mismatches(total, bad, out)
    print(f"  the same request twice: {'identical' if stable else 'DIFFERENT BYTES'}", file=out)
    print(file=out)
    return {
        "checked": len(ids),
        "duplicates": dupes,
        "mismatches": total,
        "mismatch_samples": bad,
        "repeat_stable": stable,
        "ok": total == 0 and stable,
    }


# --------------------------------------------------------------------------- cli

def parse_peers(spec: str) -> list[str]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(part if ":" in part else f"{part}:{DEFAULT_PORT}")
    if not out:
        raise SystemExit("no peers given")
    return out


def build_plan(a) -> list[tuple[str, int, int]]:
    if a.mode == "verify":
        return []
    if a.mode == "sweep":
        try:
            counts = [int(x) for x in a.sweep.split(",")] if a.sweep else list(SWEEP_ROWS)
        except ValueError:
            raise SystemExit(f"--sweep wants comma separated row counts, got {a.sweep!r}")
        iters = DEFAULT_ITERS["sweep"] if a.iters is None else a.iters
        return [("sweep", n, iters) for n in counts]
    plan = []
    if a.mode in ("all", "decode"):
        plan.append(("decode",
                     DECODE_ROWS if a.rows is None else a.rows,
                     DEFAULT_ITERS["decode"] if a.iters is None else a.iters))
    if a.mode in ("all", "prefill"):
        plan.append(("prefill",
                     PREFILL_ROWS if a.rows is None else a.rows,
                     DEFAULT_ITERS["prefill"] if a.iters is None else a.iters))
    return plan


def parse_args():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peers", default=DEFAULT_PEERS,
                    help=f"comma separated host[:port], default {DEFAULT_PEERS} on {DEFAULT_PORT}")
    ap.add_argument("--client", type=Path, default=here.parent / "client" / "ple_remote.py",
                    help="row client module, default ../client/ple_remote.py")
    ap.add_argument("--mode", default="all",
                    choices=("all", "decode", "prefill", "sweep", "verify"))
    ap.add_argument("--rows", type=int, help="row count for --mode decode or --mode prefill")
    ap.add_argument("--sweep", help="comma separated row counts for --mode sweep")
    ap.add_argument("--iters", type=int, help="timed iterations per workload")
    ap.add_argument("--warmup", type=int, default=10, help="untimed iterations, excluded")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                    help=f"the servers' --max-rows, default {DEFAULT_MAX_ROWS}")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="socket timeout in seconds once connected, default 10")
    ap.add_argument("--json", action="store_true",
                    help="machine readable report on stdout, everything else on stderr")
    ap.add_argument("--verify-against", type=Path,
                    help=f"flat half file to check the returned rows against, "
                         f"default {DEFAULT_REFERENCE} in --mode verify")
    ap.add_argument("--verify-base", type=int,
                    help="global row id of the first row in that file, default taken from STAT")
    ap.add_argument("--verify-order", action="store_true",
                    help="also check a batched gather against one row per request")
    ap.add_argument("--verify-samples", type=int, default=VERIFY_SAMPLES,
                    help=f"rows to check per verification, default {VERIFY_SAMPLES}")
    return ap.parse_args()


def resolve_reference(a) -> Path | None:
    """Which half file to check against, if any.

    The default is only reached in verify mode, so a timing run never goes looking
    for a file nobody asked about.
    """
    reference = a.verify_against
    if reference is None and a.mode == "verify":
        if DEFAULT_REFERENCE.exists():
            reference = DEFAULT_REFERENCE
        elif not a.verify_order:
            raise SystemExit(
                f"--mode verify needs a half to check against. {DEFAULT_REFERENCE} is not "
                "here, so pass --verify-against PATH (run this on a node, where the half "
                "is), or --verify-order for the check that needs no reference."
            )
    if reference is not None and not reference.exists():
        raise SystemExit(f"reference not found: {reference}")
    return reference


def main() -> int:
    a = parse_args()
    out = sys.stderr if a.json else sys.stdout
    addrs = parse_peers(a.peers)
    if not a.client.exists():
        raise SystemExit(f"client not found: {a.client}")
    if a.iters is not None and a.iters < 1:
        raise SystemExit("--iters has to be at least 1")
    if a.warmup < 0:
        raise SystemExit("--warmup cannot be negative")
    if a.max_rows < 1:
        raise SystemExit("--max-rows has to be at least 1")
    if a.rows is not None and a.mode not in ("decode", "prefill"):
        raise SystemExit("--rows applies to --mode decode or --mode prefill")

    reference = resolve_reference(a)
    run_order = a.verify_order or a.mode == "verify"
    if a.verify_base is not None and reference is None:
        raise SystemExit("--verify-base says where a reference file starts, so it needs "
                         "--verify-against")

    plan = build_plan(a)
    # Ids are drawn uniformly, so a peer's share of one request is binomial, not
    # exactly n over k. A ceiling of peers x max_rows needs a dead even split to
    # fit and is rejected on almost every draw, since the larger of two shares is
    # above half the request unless the split is exact. The only count that cannot
    # overshoot is the one where a single peer taking every id is still legal.
    ceiling = a.max_rows
    for _, rows, _ in plan:
        if rows < 1:
            raise SystemExit(f"{rows} rows per gather makes no sense")
        if rows > ceiling:
            raise SystemExit(
                f"{rows:,} rows per gather can put more than max_rows ({ceiling:,}) on one "
                "peer, which the server rejects. Lower the row count, or raise --max-rows "
                "to match a server started with a bigger cap."
            )
    if a.verify_samples < 1 or a.verify_samples > ceiling:
        raise SystemExit(f"--verify-samples has to be between 1 and max_rows ({ceiling:,})")

    cls = load_client(a.client)
    rnd = random.Random(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    report = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "bench_host": socket.gethostname(),
        "peers_dialed": addrs,
        "client": {"module": str(a.client), "class": cls.__name__, "entry_point": "gather_cpu"},
        "seed": a.seed,
        "warmup": a.warmup,
        "max_rows": a.max_rows,
        "runs": [],
    }

    table = connect(cls, addrs, a.max_rows, a.timeout)
    try:
        stats = table.stat()
        notes = residency_notes(stats)
        report["peers"] = stats
        report["notes"] = notes
        ok = True

        print(f"bench host {report['bench_host']}  client {cls.__name__}.gather_cpu"
              f"  seed {a.seed}  warmup {a.warmup}", file=out)
        print_peers(stats, notes, out)

        if reference is not None:
            res = verify_against_file(table, stats, reference, a.verify_base,
                                      a.verify_samples, rnd, out)
            report["verify_file"] = res
            ok = ok and res["ok"]
        if run_order:
            res = verify_order(table, stats, min(a.verify_samples, ORDER_SAMPLES), rnd, out)
            report["verify_order"] = res
            ok = ok and res["ok"]

        if plan:
            print(HEADER, file=out)
            for name, rows, iters in plan:
                res = run_workload(table, name, rows, iters, a.warmup, gen)
                report["runs"].append(res)
                print(format_run(res), file=out)
                out.flush()
            print(file=out)
    finally:
        table.close()

    report["ok"] = ok
    if a.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
