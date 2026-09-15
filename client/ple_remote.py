#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Network-backed PLE n-gram table: a drop-in replacement for PLEMmapTable.

Same surface as the disk reader in the Qwen3.8-Flash-Next patch
(``gather`` / ``gather_cpu`` / ``read_rows_contiguous`` / ``describe``), the
only difference being where the rows come from: each peer holds a contiguous
range of the table in its own page cache and answers row-gather requests over
TCP (see docs/protocol.md).

Torch and numpy are the only imports that matter, so this can be driven from a
benchmark script on a box with no vLLM on it.

A gather runs inside every forward pass, so a transport fault mid-gather (a
timeout, a reset, a peer closing on us) replaces the affected connections and
reissues the request, under one wall-clock deadline for the whole gather.
Protocol errors are never retried. See PLERemoteTable._exchange for the rules.

Run the module directly for a self-check of ordering and duplicate handling
against two loopback servers.
"""

from __future__ import annotations

import os
import select
import socket
import struct
import threading
import time

import numpy as np
import torch

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # standalone benchmarking, no vLLM on the box
    import logging

    logger = logging.getLogger(__name__)

MAGIC_REQ = 0x52454C50  # "PLER"
MAGIC_RESP = 0x50534552  # "RESP"
VERSION = 1
OP_GATHER, OP_PING, OP_STAT = 1, 2, 3
HEADER = 16
STAT_BODY = 32

# Checkpoint geometry. row_bytes is ple_embed_dim / ngram_heads = 2560 / 16,
# which is not the config's head_dim. Both are only defaults: what the peers
# report through STAT has to agree with them or the table refuses to load.
TABLE_ROWS = 320_001_536
ROW_BYTES = 160

DEFAULT_PEERS = "node1:9000,node2:9000"
DEFAULT_PORT = 9000
DEFAULT_MAX_ROWS = 131072
# The socket timeout is not a latency budget: a full max_rows reply is 21 MB,
# under a second of 2.5 GbE, and nothing legitimate comes near either number.
# It is what decides that a peer has gone, and on 2026-09-15 a disturbance of
# about a minute that both peers sat out healthy tripped 10 s and took the
# engine down with it. 30 s is the server's own socket timeout.
DEFAULT_TIMEOUT = 30.0
DEFAULT_CONNECT_TIMEOUT = 5.0
# Startup only. The model load costs nine to thirteen minutes, so waiting a
# minute for a peer that is not up yet is free next to failing and loading again.
DEFAULT_CONNECT_DEADLINE = 60.0
# One whole gather, reissues and reconnects included: the longest a forward
# pass can be held, and so the longest disturbance survived. Sized so the
# measured one, about 60 s, fits after the first socket timeout has been spent.
DEFAULT_GATHER_DEADLINE = 120.0
GATHER_ATTEMPTS = 3

_STATUS = {
    1: "bad magic or version",
    2: "unknown op",
    3: "row id outside that server's range",
    4: "row count above the server's max_rows",
    5: "server internal error",
}


def _env(name: str, default: str) -> str:
    # The vLLM patch namespaces its knobs QWEN4EXP_PLE_*; the short form keeps
    # standalone benchmarking readable.
    for prefix in ("QWEN4EXP_PLE_REMOTE_", "PLE_REMOTE_"):
        value = os.environ.get(prefix + name)
        if value:
            return value
    return default


class _Peer:
    """One row server: its socket, the range it owns, and its reusable buffers."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.addr: tuple = (host, port)
        self.sock: socket.socket | None = None
        self.timeout = 0.0
        self._armed = 0.0
        self.deadline: float | None = None
        self.answered = False
        self.base_row = 0
        self.row_count = 0
        self.row_bytes = 0
        self.resident_pages = 0
        self.served_requests = 0
        self.cap = 0
        self.pending: int | None = None
        self.req = bytearray()
        self.req_mv = memoryview(self.req)
        self.req_ids = np.empty(0, dtype=np.uint32)
        self.rows = np.empty((0, 0), dtype=np.uint8)
        self.rows_mv = memoryview(bytearray())
        # Header-only ops need a buffer before STAT has told us row_bytes.
        self._hdr_out = bytearray(HEADER)
        self._hdr_in = bytearray(HEADER)
        self._hdr_in_mv = memoryview(self._hdr_in)

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"

    def connect(self, connect_timeout: float, timeout: float) -> None:
        sock = socket.create_connection(self.addr, timeout=connect_timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(timeout)
        self.sock = sock
        self.timeout = self._armed = timeout
        # Reconnects go to the address the first connect resolved, so one made
        # under a gather deadline never sits in the resolver, which no timeout
        # here covers.
        self.addr = sock.getpeername()[:2]

    def close(self, abort: bool = False) -> None:
        """Drop the connection; abort makes it an RST rather than a FIN.

        For a socket whose framing is in doubt: the peer stops sending at once,
        and nothing it already sent can turn up as the answer to a later request.
        """
        sock, self.sock, self.pending = self.sock, None, None
        if sock is None:
            return
        try:
            if abort:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        finally:
            sock.close()

    def _arm(self) -> None:
        """Clamp the socket timeout to what is left of the gather deadline, if one is set.

        Before every recv and send, not once per request: the socket timeout
        restarts on every byte, so alone it bounds the silence between bytes and
        not the call. The cached value keeps the fcntl off the hot path.
        """
        budget = self.timeout
        if self.deadline is not None:
            budget = min(budget, self.deadline - time.monotonic())
            if budget <= 0:
                raise TimeoutError(f"PLE remote {self}: gather deadline passed")
        if budget != self._armed:
            self.sock.settimeout(budget)
            self._armed = budget

    def _recv(self, mv: memoryview, want: int) -> int:
        self._arm()
        try:
            return self.sock.recv_into(mv, want)
        except TimeoutError:
            raise TimeoutError(f"PLE remote {self}: no data for {self._armed:.1f}s") from None

    def reserve(self, rows: int) -> None:
        """Grow the request and response buffers to hold `rows` rows."""
        if rows <= self.cap:
            return
        cap = max(rows, 4096)
        self.req = bytearray(HEADER + 4 * cap)
        self.req_mv = memoryview(self.req)
        self.req_ids = np.frombuffer(self.req, dtype=np.uint32, offset=HEADER)
        self.rows = np.empty((cap, self.row_bytes), dtype=np.uint8)
        self.rows_mv = memoryview(self.rows).cast("B")
        self.cap = cap

    def fail(self, what: str, exc: type[Exception] = RuntimeError) -> None:
        raise exc(f"PLE remote {self}: {what}")

    def send_gather(self, req_id: int, count: int) -> None:
        """Send a GATHER for the ids already written into req_ids[:count]."""
        struct.pack_into("<IBBHII", self.req, 0, MAGIC_REQ, VERSION, OP_GATHER, 0, req_id, count)
        self._arm()
        self.sock.sendall(self.req_mv[: HEADER + 4 * count])
        self.pending = req_id

    def send_bare(self, op: int, req_id: int) -> None:
        struct.pack_into("<IBBHII", self._hdr_out, 0, MAGIC_REQ, VERSION, op, 0, req_id, 0)
        self._arm()
        self.sock.sendall(self._hdr_out)
        self.pending = req_id

    def read_rows(self, count: int) -> np.ndarray:
        """Read the response for the outstanding GATHER, in request order."""
        self.begin_rows(count)
        self.finish_rows()
        return self.rows[:count]

    def begin_rows(self, count: int) -> None:
        """Consume the response header and set up the body read, without blocking on it.

        Separating the header from the body is what lets a caller with several peers
        drain them together. Draining one peer to completion first leaves the others
        stalled on a full receive window once a response outgrows the socket buffer,
        which on a prefill-sized gather costs more than the transfer itself.
        """
        got = self._read_header()
        if got != count:
            self.fail(f"returned {got} rows for a request of {count}")
        self._rd_want = count * self.row_bytes
        self._rd_mv = self.rows_mv[: self._rd_want]
        self._rd_got = 0

    def pump_rows(self) -> bool:
        """One recv into the pending body. True once it is complete."""
        n = self._recv(self._rd_mv[self._rd_got:], self._rd_want - self._rd_got)
        if not n:
            self.fail(f"connection closed after {self._rd_got} of {self._rd_want} bytes", ConnectionError)
        self._rd_got += n
        self.answered = self._rd_got >= self._rd_want
        return self.answered

    def finish_rows(self) -> None:
        while self._rd_got < self._rd_want:
            self.pump_rows()

    def read_ack(self) -> None:
        if self._read_header() != 0:
            self.fail("header-only response carried row data")

    def query_stat(self, req_id: int) -> dict:
        self.send_bare(OP_STAT, req_id)
        if self._read_header() != 0:
            self.fail("STAT response carried row data")
        body = bytearray(STAT_BODY)
        self._recv_exact(memoryview(body))
        base, rows, row_bytes, resident_pages, served = struct.unpack("<QQIIQ", body)
        return {
            "peer": str(self),
            "base_row": base,
            "row_count": rows,
            "row_bytes": row_bytes,
            "resident_pages": resident_pages,
            "served_requests": served,
        }

    def _read_header(self) -> int:
        self._recv_exact(self._hdr_in_mv)
        magic, req_id, status, count = struct.unpack_from("<IIII", self._hdr_in)
        if magic != MAGIC_RESP:
            self.fail(f"response magic 0x{magic:08x}, expected 0x{MAGIC_RESP:08x}")
        if req_id != self.pending:
            self.fail(f"response for request {req_id}, expected {self.pending} (stream out of sync)")
        # Cleared before the status check because a rejected request still leaves the
        # connection in sync: the server sends a complete header and no row data. Raising
        # with pending still set would make the next request look like a desync.
        self.pending = None
        if status:
            # Status 5 is sent only from the server's out-of-memory path at
            # connection setup, after which it closes, so a fresh connection
            # is worth a try. Every other status is a bug or a misconfiguration.
            self.fail(f"status {status} ({_STATUS.get(status, 'unknown status')})",
                      ConnectionError if status == 5 else RuntimeError)
        return count

    def _recv_exact(self, mv: memoryview) -> None:
        want = len(mv)
        got = 0
        while got < want:
            n = self._recv(mv[got:], want - got)
            if not n:
                self.fail(f"connection closed after {got} of {want} bytes", ConnectionError)
            got += n


class PLERemoteTable:
    """One PLE n-gram table, served out of peer machines' memory.

    The table is split into contiguous row ranges, one per peer. Which peer owns
    what is learned from STAT at connect time rather than configured here, so a
    server started on the wrong half is caught instead of quietly serving the
    wrong embeddings.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        layer_idx: int = 0,
        *,
        threads: int | None = None,
        peers: str | None = None,
        rows_total: int | None = None,
        row_bytes: int = ROW_BYTES,
        max_rows: int | None = None,
        timeout: float | None = None,
        connect_timeout: float | None = None,
        connect_deadline: float | None = None,
        gather_deadline: float | None = None,
    ) -> None:
        # model_dir and threads exist so this can stand in for PLEMmapTable
        # unchanged. Nothing here reads the checkpoint, and the concurrency is
        # the peers themselves rather than a local pool.
        self.model_dir = model_dir
        self.layer_idx = int(layer_idx)
        self.threads = 0 if threads is None else int(threads)
        self.dtype = torch.float8_e4m3fn
        self.row_bytes = int(row_bytes)
        self.rows_total = int(rows_total if rows_total is not None else _env("ROWS", str(TABLE_ROWS)))
        self.max_rows = int(max_rows if max_rows is not None else _env("MAX_ROWS", str(DEFAULT_MAX_ROWS)))
        self.timeout = float(timeout if timeout is not None else _env("TIMEOUT", str(DEFAULT_TIMEOUT)))
        self.connect_timeout = float(
            connect_timeout if connect_timeout is not None else _env("CONNECT_TIMEOUT", str(DEFAULT_CONNECT_TIMEOUT))
        )
        self.connect_deadline = float(
            connect_deadline if connect_deadline is not None else _env("CONNECT_DEADLINE", str(DEFAULT_CONNECT_DEADLINE))
        )
        self.gather_deadline = float(
            gather_deadline if gather_deadline is not None else _env("GATHER_DEADLINE", str(DEFAULT_GATHER_DEADLINE))
        )
        self.peers = _parse_peers(peers if peers is not None else _env("PEERS", DEFAULT_PEERS))

        self._lock = threading.Lock()
        self._req_id = 0
        self._failed: str | None = None
        self._staging: torch.Tensor | None = None
        self._staging_np: np.ndarray | None = None
        self._sel: list[np.ndarray] = [np.empty(0, dtype=np.intp)] * len(self.peers)
        self._take = np.empty(0, dtype=np.int64)
        self._mask_a = np.empty(0, dtype=bool)
        self._mask_b = np.empty(0, dtype=bool)

        try:
            deadline = time.monotonic() + self.connect_deadline
            for peer in self.peers:
                self._connect(peer, deadline)
                self._apply_stat(peer, peer.query_stat(self._next_req_id()))
            self._check_split()
        except Exception:
            for peer in self.peers:
                peer.close()
            raise

        # Shards are a checkpoint concept the client never sees: it addresses
        # global rows and the peers own ranges. Both attributes stay, pointed at
        # the remote split, so a caller reading them gets a true number.
        # shard_rows is the first peer's range, which is every peer's range for
        # any even split.
        self.num_shards = len(self.peers)
        self.shard_rows = self.peers[0].row_count
        logger.info("PLE remote table: %s", self.describe())

    # ------------------------------------------------------------ connect
    def _connect(self, peer: _Peer, deadline: float) -> None:
        """Connect peer, retrying with backoff until deadline, and say so in the log."""
        pause = 0.5
        while True:
            left = deadline - time.monotonic()
            try:
                peer.connect(min(self.connect_timeout, max(left, 0.001)), self.timeout)
                return
            except OSError as exc:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"PLE remote {peer}: still unreachable at the deadline ({exc})") from exc
                logger.warning("PLE remote %s: connect failed (%s), retrying for up to %.0fs more", peer, exc, left)
                time.sleep(min(pause, left))
                pause = min(pause * 2, 5.0)

    def _check_range(self, peer: _Peer, stat: dict) -> None:
        """A peer whose range moved would serve the right shaped bytes for the wrong ids."""
        if (stat["base_row"], stat["row_count"], stat["row_bytes"]) != (
            peer.base_row,
            peer.row_count,
            peer.row_bytes,
        ):
            raise RuntimeError(
                f"PLE remote: {peer} changed its range under us: "
                f"{peer.base_row:,}+{peer.row_count:,} -> {stat['base_row']:,}+{stat['row_count']:,}"
            )
        peer.resident_pages = stat["resident_pages"]
        peer.served_requests = stat["served_requests"]

    def _apply_stat(self, peer: _Peer, stat: dict) -> None:
        peer.base_row = stat["base_row"]
        peer.row_count = stat["row_count"]
        peer.row_bytes = stat["row_bytes"]
        peer.resident_pages = stat["resident_pages"]
        peer.served_requests = stat["served_requests"]
        if peer.row_bytes != self.row_bytes:
            peer.fail(f"serves {peer.row_bytes} byte rows, table is {self.row_bytes}")
        if peer.row_count <= 0:
            peer.fail("serves an empty row range")

    def _check_split(self) -> None:
        """Assert the peers tile [0, rows_total) exactly.

        A gap means some ids have nowhere to go; an overlap means two servers
        claim the same row and the answer depends on which one is asked. Both
        are silent wrong-embedding bugs downstream, so neither is tolerated.
        """
        self.peers.sort(key=lambda p: p.base_row)
        expect = 0
        for peer in self.peers:
            if peer.base_row != expect:
                kind = "gap" if peer.base_row > expect else "overlap"
                raise RuntimeError(
                    f"PLE remote: {kind} in the table split: {peer} starts at row "
                    f"{peer.base_row:,}, expected {expect:,}"
                )
            expect += peer.row_count
        if expect != self.rows_total:
            raise RuntimeError(
                f"PLE remote: peers cover {expect:,} rows, table has {self.rows_total:,} "
                f"(peers: {', '.join(f'{p}@{p.base_row:,}+{p.row_count:,}' for p in self.peers)})"
            )

    def _next_req_id(self) -> int:
        self._req_id = (self._req_id + 1) & 0xFFFFFFFF
        return self._req_id

    def _check_usable(self) -> None:
        if self._failed:
            raise RuntimeError(f"PLE remote table is not usable ({self._failed}); rebuild it to reconnect")

    # ------------------------------------------------------------- gather
    def _staging_for(self, n: int) -> torch.Tensor:
        if self._staging is None or self._staging.shape[0] < n:
            cap = max(n, 4096)
            try:
                buf = torch.empty((cap, self.row_bytes), dtype=torch.uint8, pin_memory=True)
            except Exception:
                # No CUDA context (CPU test) or pinned pool exhausted: pageable
                # staging still works, just without async H2D overlap.
                buf = torch.empty((cap, self.row_bytes), dtype=torch.uint8)
            self._staging = buf
            self._staging_np = buf.numpy()
        return self._staging[:n]

    def _workspace_for(self, n: int) -> None:
        if self._take.shape[0] < n:
            cap = max(n, 4096)
            self._take = np.empty(cap, dtype=np.int64)
            self._mask_a = np.empty(cap, dtype=bool)
            self._mask_b = np.empty(cap, dtype=bool)

    def _partition(self, ids: np.ndarray, n: int) -> int:
        """Index positions per peer, in caller order. Returns the ids routed.

        Runs to completion before any socket traffic: a bad id has to fail
        before half the requests are on the wire, or the responses left behind
        would desync every later gather.
        """
        lo = self._mask_a[:n]
        hi = self._mask_b[:n]
        routed = 0
        for i, peer in enumerate(self.peers):
            np.greater_equal(ids, peer.base_row, out=lo)
            np.less(ids, peer.base_row + peer.row_count, out=hi)
            np.logical_and(lo, hi, out=lo)
            sel = np.flatnonzero(lo)
            if sel.size > self.max_rows:
                raise ValueError(
                    f"PLE remote: {sel.size:,} rows for {peer} exceeds max_rows "
                    f"{self.max_rows:,}; raise it on the servers and in PLE_REMOTE_MAX_ROWS"
                )
            self._sel[i] = sel
            routed += sel.size
        return routed

    def _raise_unroutable(self, ids: np.ndarray, n: int) -> None:
        covered = np.zeros(n, dtype=bool)
        for sel in self._sel:
            covered[sel] = True
        missing = np.flatnonzero(~covered)
        first = int(missing[0])
        raise IndexError(
            f"PLE remote: {missing.size} row id(s) outside the served table "
            f"[0, {self.rows_total:,}): id {int(ids[first])} at position {first}"
        )

    def gather_cpu(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather rows for flat int64 CPU ids -> uint8 [n, row_bytes] staging.

        Every request goes out before any response is read, which is what makes
        the peers work at the same time instead of one after the other.
        """
        assert ids.device.type == "cpu"
        n = ids.numel()
        out = self._staging_for(n)
        if n == 0:
            return out
        self._check_usable()
        flat = ids.detach().reshape(-1)
        if flat.dtype != torch.int64:
            flat = flat.to(torch.int64)
        ids_np = flat.numpy()
        with self._lock:
            self._workspace_for(n)
            if self._partition(ids_np, n) != n:
                self._raise_unroutable(ids_np, n)
            try:
                take = self._take
                active = []
                for peer, sel in zip(self.peers, self._sel):
                    k = sel.size
                    if not k:
                        continue
                    peer.reserve(k)
                    np.take(ids_np, sel, out=take[:k])
                    np.copyto(peer.req_ids[:k], take[:k], casting="unsafe")
                    active.append((peer, sel))
                self._exchange(active)
                rows = self._staging_np[:n]
                for peer, sel in active:
                    rows[sel] = peer.rows[: sel.size]
            except Exception as exc:
                # A protocol error is a bug or a misconfiguration, and a
                # transport fault that outlasted the retry budget has left at
                # least one socket out of sync. Either way the whole table goes.
                self._failed = f"{type(exc).__name__}: {exc}"
                raise
        return out

    def _exchange(self, active: list[tuple[_Peer, np.ndarray]]) -> None:
        """Send every peer its request and read every answer, reissuing on a transport fault.

        A timeout, a reset or a peer closing mid-message gets the unanswered
        requests sent again on fresh connections, up to GATHER_ATTEMPTS times.
        Which peers are reissued is decided by which have not finished reading
        their answer, not by which one raised: with two responses in flight
        the other socket is just as much in doubt, and a socket in doubt is
        closed with an RST and never read again, because bytes left on it
        would be taken for the reply to whatever is sent next. Reconnecting is
        part of the next attempt, so a fault there costs an attempt and not the
        gather, and the reconnected peer is asked for STAT again so one that
        came back with a different range is refused.

        The deadline is enforced inside the reads, not between attempts: every
        blocking call in here gets at most what is left of it, the select in
        _drain, each recv and send through _Peer._arm, and the connect and the
        pause before it in _connect. So a gather returns or raises within
        gather_deadline of starting, plus the scheduler's latency in waking
        the one call that was pending when it ran out. Attempts alone could
        not promise that, because the socket timeout restarts on every byte
        and a peer trickling one byte per timeout would never trip it.
        """
        start = time.monotonic()
        deadline = start + self.gather_deadline
        for peer, _ in active:
            peer.deadline = deadline
        todo = active
        try:
            for attempt in range(1, GATHER_ATTEMPTS + 1):
                try:
                    for peer, _ in todo:
                        peer.answered = False
                        if peer.sock is None:
                            self._connect(peer, deadline)
                            self._check_range(peer, peer.query_stat(self._next_req_id()))
                    for peer, sel in todo:
                        peer.send_gather(self._next_req_id(), sel.size)
                    for peer, sel in todo:
                        peer.begin_rows(sel.size)
                    if len(todo) > 1:
                        self._drain([peer for peer, _ in todo], deadline)
                    else:
                        todo[0][0].finish_rows()
                    if attempt > 1:
                        logger.warning("PLE remote: gather recovered on attempt %d, %.1fs in", attempt, time.monotonic() - start)
                    return
                except OSError as exc:
                    todo = [(peer, sel) for peer, sel in todo if not peer.answered]
                    for peer, _ in todo:
                        peer.close(abort=True)
                    if attempt == GATHER_ATTEMPTS or time.monotonic() >= deadline:
                        raise
                    logger.warning(
                        "PLE remote: gather attempt %d failed, %s: %s; reconnecting %s",
                        attempt, type(exc).__name__, exc, ", ".join(str(peer) for peer, _ in todo),
                    )
        finally:
            for peer, _ in active:
                peer.deadline = None

    def _drain(self, peers: list[_Peer], deadline: float) -> None:
        """Read every peer's body as its bytes arrive, rather than one peer at a time."""
        waiting = {peer.sock: peer for peer in peers}
        while waiting:
            wait = min(self.timeout, deadline - time.monotonic())
            ready = select.select(list(waiting), [], [], wait)[0] if wait > 0 else []
            if not ready:
                names = ", ".join(str(peer) for peer in waiting.values())
                raise TimeoutError(
                    f"PLE remote: no data from {names} for {wait:.1f}s" if wait > 0
                    else f"PLE remote: gather deadline passed with {names} unanswered"
                )
            for sock in ready:
                if waiting[sock].pump_rows():
                    del waiting[sock]

    def gather(self, ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Gather rows for ids (any device) into a tensor of self.dtype.

        Returns [n, row_bytes] on ids.device (or ``out`` if given).
        """
        device = ids.device
        cpu_ids = ids.detach().to("cpu", dtype=torch.int64)  # sync point
        rows = self.gather_cpu(cpu_ids)
        if out is None:
            out = torch.empty((rows.shape[0], self.row_bytes), dtype=torch.uint8, device=device)
        out[: rows.shape[0]].copy_(rows, non_blocking=True)
        return out[: rows.shape[0]].view(self.dtype)

    def read_rows_contiguous(self, start: int, count: int, chunk_rows: int = 1 << 20) -> torch.Tensor:
        """Read rows [start, start+count) into one uint8 tensor [count, row_bytes].

        Load-time path only. It pulls the range through the same request path as
        a gather, so a whole half over 2.5 GbE is minutes rather than seconds.
        """
        if count <= 0:
            return torch.empty((0, self.row_bytes), dtype=torch.uint8)
        try:
            buf = torch.empty((count, self.row_bytes), dtype=torch.uint8, pin_memory=True)
        except Exception:
            buf = torch.empty((count, self.row_bytes), dtype=torch.uint8)
        step = max(1, min(int(chunk_rows), self.max_rows))
        done = 0
        while done < count:
            n = min(step, count - done)
            ids = torch.arange(start + done, start + done + n, dtype=torch.int64)
            buf[done : done + n].copy_(self.gather_cpu(ids))
            done += n
        return buf

    # --------------------------------------------------------------- info
    def ping(self) -> None:
        """Round trip every peer. Worth doing once before the first decode step."""
        with self._lock:
            self._check_usable()
            try:
                for peer in self.peers:
                    peer.send_bare(OP_PING, self._next_req_id())
                for peer in self.peers:
                    peer.read_ack()
            except Exception as exc:
                self._failed = f"{type(exc).__name__}: {exc}"
                raise

    def stat(self) -> list[dict]:
        """Fresh STAT from every peer, for health checks.

        Shares the gather sockets, so keep it off the hot path.
        """
        with self._lock:
            self._check_usable()
            try:
                stats = [peer.query_stat(self._next_req_id()) for peer in self.peers]
            except Exception as exc:
                self._failed = f"{type(exc).__name__}: {exc}"
                raise
            for peer, stat in zip(self.peers, stats):
                self._check_range(peer, stat)
        return stats

    def close(self) -> None:
        with self._lock:
            for peer in self.peers:
                peer.close()
            self._failed = self._failed or "closed"

    def describe(self) -> str:
        where = ", ".join(f"{p}[{p.base_row:,}+{p.row_count:,}]" for p in self.peers)
        return (
            f"PLERemoteTable(layer={self.layer_idx}, peers={len(self.peers)}, "
            f"rows={self.rows_total:,}, row_bytes={self.row_bytes}, dtype={self.dtype}, {where})"
        )


def _parse_peers(spec: str) -> list[_Peer]:
    peers: list[_Peer] = []
    seen: set[tuple[str, int]] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        host, sep, port = item.rpartition(":")
        if not sep:
            host, port = item, str(DEFAULT_PORT)
        key = (host, int(port))
        if key in seen:
            raise ValueError(f"PLE remote: peer {item} listed twice in {spec!r}")
        seen.add(key)
        peers.append(_Peer(host, int(port)))
    if not peers:
        raise ValueError(f"PLE remote: no peers in {spec!r}")
    return peers


__all__ = ["PLERemoteTable"]


# ---------------------------------------------------------------- self-check
# Everything below runs only under __main__. The loopback server here is a
# stand-in for the real one so the client can be checked without a deployment.


def _synthetic_table(rows: int, row_bytes: int) -> np.ndarray:
    r = np.arange(rows, dtype=np.uint32).reshape(-1, 1)
    c = np.arange(row_bytes, dtype=np.uint32).reshape(1, -1)
    return ((r * 31 + c * 7) & 0xFF).astype(np.uint8)


class _LoopbackServer(threading.Thread):
    def __init__(self, rows: np.ndarray, base: int) -> None:
        super().__init__(daemon=True)
        self.rows = rows
        self.base = base
        self.row_bytes = rows.shape[1]
        self.served = 0
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]

    def run(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    @staticmethod
    def _head(req_id: int, status: int, count: int) -> bytes:
        return struct.pack("<IIII", MAGIC_RESP, req_id, status, count)

    def _serve(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with conn:
            while True:
                head = self._recv(conn, HEADER)
                if head is None:
                    return
                magic, version, op, _pad, req_id, count = struct.unpack("<IBBHII", head)
                if magic != MAGIC_REQ or version != VERSION:
                    conn.sendall(self._head(req_id, 1, 0))
                elif op == OP_STAT:
                    conn.sendall(
                        self._head(req_id, 0, 0)
                        + struct.pack("<QQIIQ", self.base, self.rows.shape[0], self.row_bytes, 0, self.served)
                    )
                elif op == OP_PING:
                    conn.sendall(self._head(req_id, 0, 0))
                elif op != OP_GATHER:
                    conn.sendall(self._head(req_id, 2, 0))
                else:
                    ids = np.frombuffer(self._recv(conn, 4 * count), dtype=np.uint32).astype(np.int64) - self.base
                    if count and (ids.min() < 0 or ids.max() >= self.rows.shape[0]):
                        conn.sendall(self._head(req_id, 3, 0))
                    else:
                        self.served += 1
                        conn.sendall(self._head(req_id, 0, count) + self.rows[ids].tobytes())

    @staticmethod
    def _recv(conn: socket.socket, want: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < want:
            chunk = conn.recv(want - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)


def _expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception as exc:
        print(f"ok   {label}: {type(exc).__name__}: {exc}")
        return
    raise AssertionError(f"{label}: expected a failure, got none")


def _self_check() -> None:
    rows_total, row_bytes = 4096, ROW_BYTES
    half = rows_total // 2
    table = _synthetic_table(rows_total, row_bytes)
    left = _LoopbackServer(table[:half], 0)
    right = _LoopbackServer(table[half:], half)
    left.start()
    right.start()
    peers = f"127.0.0.1:{left.port},127.0.0.1:{right.port}"

    t = PLERemoteTable(peers=peers, rows_total=rows_total)
    print(f"ok   connect: {t.describe()}")
    assert (t.rows_total, t.row_bytes, t.num_shards) == (rows_total, row_bytes, 2)

    # Out of order, duplicated, and crossing the peer boundary in both directions.
    pattern = [0, rows_total - 1, half, half - 1, 0, 0, half, 1, rows_total - 1, half - 1, half - 1]
    got = t.gather_cpu(torch.tensor(pattern, dtype=torch.int64)).numpy()
    assert np.array_equal(got, table[pattern]), "gather_cpu returned rows out of order"
    print(f"ok   ordering and duplicates over {len(pattern)} ids spanning both peers")

    rng = np.random.default_rng(7)
    heavy = rng.integers(0, rows_total, 4096)
    heavy[1::3] = heavy[0]  # a third of the request is the same row
    got = t.gather_cpu(torch.from_numpy(heavy)).numpy()
    assert np.array_equal(got, table[heavy]), "heavy-duplicate gather mismatched"
    print(f"ok   {heavy.size} ids with {heavy.size - np.unique(heavy).size} duplicates")

    assert t.gather_cpu(torch.empty(0, dtype=torch.int64)).shape == (0, row_bytes)
    print("ok   empty gather")

    ids = torch.tensor([half - 1, half, 3, 3], dtype=torch.int64)
    out = torch.empty((4, row_bytes), dtype=torch.uint8)
    ret = t.gather(ids, out=out)
    assert ret.dtype == torch.float8_e4m3fn and ret.data_ptr() == out.data_ptr()
    assert np.array_equal(out.numpy(), table[ids.numpy()])
    print("ok   gather() honours out= and returns float8_e4m3fn")

    slab = t.read_rows_contiguous(half - 8, 16)
    assert np.array_equal(slab.numpy(), table[half - 8 : half + 8])
    print("ok   read_rows_contiguous across the peer boundary")

    # Both of these have to be caught before anything is sent, otherwise the
    # responses left behind would desync the next gather.
    _expect_failure("id past the end of the table", lambda: t.gather_cpu(torch.tensor([0, rows_total])))
    t.max_rows = 4
    _expect_failure("request over max_rows", lambda: t.gather_cpu(torch.arange(0, 64)))
    t.max_rows = DEFAULT_MAX_ROWS
    assert np.array_equal(t.gather_cpu(torch.tensor([5, 6])).numpy(), table[[5, 6]])
    print("ok   sockets still in sync after rejected requests")

    stats = t.stat()
    assert [s["base_row"] for s in stats] == [0, half]
    print(f"ok   stat: {stats[0]['served_requests']} + {stats[1]['served_requests']} requests served")
    t.ping()
    print("ok   ping")
    t.close()

    _expect_failure(
        "one peer short",
        lambda: PLERemoteTable(peers=f"127.0.0.1:{left.port}", rows_total=rows_total),
    )
    _expect_failure(
        "peers do not add up to rows_total",
        lambda: PLERemoteTable(peers=peers, rows_total=rows_total + 1),
    )
    overlap = _LoopbackServer(table[:half], 0)
    overlap.start()
    _expect_failure(
        "two peers claim the same rows",
        lambda: PLERemoteTable(peers=f"127.0.0.1:{left.port},127.0.0.1:{overlap.port}", rows_total=rows_total),
    )
    gap = _LoopbackServer(table[half + 1 :], half + 1)
    gap.start()
    _expect_failure(
        "gap between the peers",
        lambda: PLERemoteTable(peers=f"127.0.0.1:{left.port},127.0.0.1:{gap.port}", rows_total=rows_total),
    )
    print("self-check passed")


if __name__ == "__main__":
    _self_check()
