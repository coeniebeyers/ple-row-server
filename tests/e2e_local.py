#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic halves and client-side assertions for tests/e2e_local.sh.

Two subcommands, sharing one row pattern so the generator and the checker can
never drift apart:

  gen    write a half file of rows [base, base + count)
  check  drive real row servers through client/ple_remote.py and assert

Row content is a function of the global row id, so a server that serves the
right shaped bytes from the wrong offset fails the comparison instead of
passing it.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROW_BYTES = 160
WORDS = ROW_BYTES // 8

FAILURES = 0


def rows_for(ids) -> np.ndarray:
    """Rows for global ids, as [n, ROW_BYTES] uint8.

    Word 0 is the id itself, so a mismatch report says which row arrived. The
    rest is splitmix64 keyed by the id, which keeps neighbouring rows and rows
    a power of two apart from looking alike.
    """
    ids = np.asarray(ids, dtype=np.uint64).reshape(-1)
    words = np.empty((ids.size, WORDS), dtype=np.uint64)
    with np.errstate(over="ignore"):
        words[:, 0] = ids
        for k in range(1, WORDS):
            z = ids + np.uint64(k) * np.uint64(0x9E3779B97F4A7C15)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            words[:, k] = z ^ (z >> np.uint64(31))
    # Host byte order on both sides of the comparison, and the server only ever
    # copies these bytes around, so nothing here depends on which order that is.
    return words.view(np.uint8).reshape(ids.size, ROW_BYTES)


def as_ids(values) -> torch.Tensor:
    """Ids the way a caller of the table hands them over: flat int64 on the CPU."""
    return torch.as_tensor(np.asarray(values, dtype=np.int64).reshape(-1))


def cmd_gen(args: argparse.Namespace) -> int:
    rows = rows_for(np.arange(args.base, args.base + args.count, dtype=np.uint64))
    Path(args.out).write_bytes(rows.tobytes())
    print(f"gen  {args.out}: rows [{args.base}, {args.base + args.count}), {rows.nbytes} bytes")
    return 0


# ----------------------------------------------------------------- assertions

def ok(label: str) -> None:
    print(f"ok    {label}")


def fail(label: str, detail: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"FAIL  {label}: {detail}")


def expect(label: str, cond: bool, detail: str = "") -> bool:
    if cond:
        ok(label)
        return True
    fail(label, detail or "condition was false")
    return False


def expect_rows(label: str, got: np.ndarray, ids) -> None:
    want = rows_for(ids)
    if got.shape != want.shape:
        fail(label, f"shape {got.shape}, expected {want.shape}")
        return
    bad = np.flatnonzero((got != want).any(axis=1))
    if bad.size == 0:
        ok(f"{label} ({want.shape[0]} rows)")
        return
    i = int(bad[0])
    served = int(got[i, :8].view(np.uint64)[0])
    fail(
        label,
        f"{bad.size} of {want.shape[0]} rows wrong; position {i} asked for id "
        f"{int(np.asarray(ids, dtype=np.uint64).reshape(-1)[i])} and got id {served}",
    )


def load_client(path: Path):
    spec = importlib.util.spec_from_file_location("ple_remote_e2e", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load client module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def raw_gather(table, peer, ids):
    """One GATHER straight at one peer, through the client's own peer object.

    gather_cpu refuses an out of range id and an over sized request before it
    puts anything on the wire, which is the right thing for production and the
    reason the server side of those two rules needs this path to be tested at
    all.
    """
    ids = np.asarray(ids, dtype=np.uint32).reshape(-1)
    peer.reserve(ids.size)
    peer.req_ids[: ids.size] = ids
    peer.send_gather(table._next_req_id(), ids.size)
    return peer.read_rows(ids.size)


def raw_gather_status(table, peer, ids):
    """Return the status code the server answered a deliberately bad request with.

    A non-zero status leaves the client's pending request id set, because the
    client has no recovery path by design: a real gather marks the whole table
    failed. The server is still in sync, having answered header only, so the
    test clears that one field and carries on using the connection.
    """
    ids = np.asarray(ids, dtype=np.uint32).reshape(-1)
    peer.reserve(ids.size)
    peer.req_ids[: ids.size] = ids
    peer.send_gather(table._next_req_id(), ids.size)
    try:
        peer.read_rows(ids.size)
    except RuntimeError as exc:
        peer.pending = None
        text = str(exc)
        for code in range(1, 6):
            if f"status {code} " in text:
                return code, text
        return -1, text
    return 0, "server accepted a request it should have rejected"


def cmd_check(args: argparse.Namespace) -> int:
    client = load_client(Path(args.client))
    table_cls = client.PLERemoteTable

    half = args.rows_total // 2
    top = args.rows_total - 1

    table = table_cls(
        peers=args.peers,
        rows_total=args.rows_total,
        max_rows=args.max_rows,
        timeout=15.0,
        connect_timeout=5.0,
    )
    ok(f"connect: {table.describe()}")

    try:
        # 1. STAT geometry, straight off the wire rather than from what the
        # client cached at connect time.
        stats = table.stat()
        want = [(0, half), (half, args.rows_total - half)]
        for stat, (base, count) in zip(stats, want):
            expect(
                f"stat {stat['peer']}: base_row {base}, row_count {count}, row_bytes {ROW_BYTES}",
                (stat["base_row"], stat["row_count"], stat["row_bytes"]) == (base, count, ROW_BYTES),
                f"got base_row {stat['base_row']}, row_count {stat['row_count']}, "
                f"row_bytes {stat['row_bytes']}",
            )
        expect("stat reports both peers", len(stats) == 2, f"got {len(stats)}")

        table.ping()
        ok("ping both peers")

        # 2. The split itself: the last row of the first half, the first row of
        # the second, and their neighbours, in one request that has to be cut in
        # two and stitched back together.
        boundary = [half - 2, half - 1, half, half + 1, 0, top, half, half - 1]
        expect_rows("boundary rows across the split", table.gather_cpu(as_ids(boundary)).numpy(), boundary)

        # 3. Order and duplicates. Shuffled, both halves, every id repeated.
        rng = np.random.default_rng(20260914)
        order = rng.integers(0, args.rows_total, 512, dtype=np.int64)
        order = np.repeat(order, 2)
        rng.shuffle(order)
        got = table.gather_cpu(as_ids(order)).numpy()
        expect_rows("request order preserved over a shuffled request", got, order)
        dup = int(order.size - np.unique(order).size)
        expect(
            f"duplicates come back duplicated ({dup} of {order.size} ids are repeats)",
            dup > 0 and np.array_equal(got, rows_for(order)),
            "the duplicate rows did not match",
        )

        # 4. A request at exactly max_rows, entirely inside one half.
        edge = np.arange(half - args.max_rows, half, dtype=np.int64)
        expect_rows(f"a gather of exactly max_rows ({args.max_rows}) rows", table.gather_cpu(as_ids(edge)).numpy(), edge)

        # 5. Every row of the table, over both peers, through the load time path.
        slab = table.read_rows_contiguous(0, args.rows_total)
        expect_rows("read_rows_contiguous over the whole table", slab.numpy(), np.arange(args.rows_total))

        # 6. Out of range, at the server. Both peers, both kinds: an id that
        # belongs to the other peer and an id past the end of the table.
        lo, hi = table.peers
        for peer, ids, label in (
            (lo, [half], "an id belonging to the other peer"),
            (hi, [half - 1], "an id belonging to the other peer"),
            (hi, [args.rows_total + 7], "an id past the end of the table"),
            (lo, [0xFFFFFFFF], "the largest u32 there is"),
        ):
            status, text = raw_gather_status(table, peer, ids)
            expect(f"{peer}: {label} ({ids[0]}) is rejected with status 3", status == 3, text)

        # 7. Over max_rows, at the server. The ids are all in range, so count is
        # the only thing wrong with the request.
        over = args.max_rows + 1
        ids = np.arange(over, dtype=np.uint32) % np.uint32(half)
        status, text = raw_gather_status(table, lo, ids)
        expect(f"{lo}: a request of {over} rows is rejected with status 4", status == 4, text)

        # 8. And the connection survived all of that, both at the low level and
        # through the client's own path.
        after = [half - 1, 3, 3, 0]
        expect_rows("the connection still works after a rejected request", table.gather_cpu(as_ids(after)).numpy(), after)
        expect_rows(
            f"{lo}: direct gather after a rejected request",
            np.asarray(raw_gather(table, lo, [1, 2, 1])),
            [1, 2, 1],
        )

        served = table.stat()
        expect(
            "both peers counted the requests they answered",
            all(s["served_requests"] > 0 for s in served),
            ", ".join(f"{s['peer']}={s['served_requests']}" for s in served),
        )
        print(
            "      served: "
            + ", ".join(f"{s['peer']} {s['served_requests']} requests, "
                        f"{s['resident_pages']} resident pages" for s in served)
        )
    finally:
        table.close()

    if FAILURES:
        print(f"\n{FAILURES} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="write one synthetic half")
    g.add_argument("--out", required=True)
    g.add_argument("--base", type=int, required=True)
    g.add_argument("--count", type=int, required=True)
    g.set_defaults(func=cmd_gen)

    c = sub.add_parser("check", help="drive the servers through the real client")
    c.add_argument("--client", required=True, help="path to client/ple_remote.py")
    c.add_argument("--peers", required=True, help="host:port,host:port")
    c.add_argument("--rows-total", type=int, required=True)
    c.add_argument("--max-rows", type=int, required=True, help="what the servers were started with")
    c.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
