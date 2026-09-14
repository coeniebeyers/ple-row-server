/*
 * rowserverd: serve one contiguous slice of the Qwen3.8-Flash-Next PLE n-gram
 * table out of this machine's page cache. Wire protocol in docs/protocol.md.
 *
 * The slice is mmap'd read-only, file-backed and shared, and that is the whole
 * trick. File pages are reclaimable, so if the box ever comes under genuine
 * memory pressure the kernel takes rows back and gathers get slower, rather than
 * the OOM killer taking out the k3s etcd member that lives on node1. Nothing here
 * mlocks, and the table never occupies anonymous memory.
 *
 * A decode step gathers about 512 rows inside a ~1 ms budget, so the serving path
 * does no allocation, no parsing and one write per request.
 */

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define REQ_MAGIC   0x52454C50u   /* "PLER" */
#define RESP_MAGIC  0x50534552u   /* "RESP" */
#define PROTO_VER   1u
#define HDR_BYTES   16u
#define STAT_BYTES  32u
#define ROW_BYTES   160u          /* ple_embed_dim / ngram_heads = 2560 / 16, not config head_dim */

enum { OP_GATHER = 1, OP_PING = 2, OP_STAT = 3 };

enum {
    ST_OK = 0,
    ST_BAD_HEADER = 1,
    ST_BAD_OP = 2,
    ST_RANGE = 3,
    ST_TOO_MANY = 4,
    ST_INTERNAL = 5
};

enum { RD_OK = 0, RD_EOF = 1, RD_ERR = 2, RD_TIMEOUT = 3 };

#define DEFAULT_PORT      9000        /* free on both nodes: Phase 0 streamed the halves over it */
#define DEFAULT_MAX_ROWS  131072u     /* 4096 token prefill chunk x 16 heads, with room */
#define DEFAULT_THREADS   8
#define MAX_MAX_ROWS      (1u << 21)  /* 320 MiB of response buffer per connection is already absurd */
#define MAX_THREADS       256
#define WARM_CHUNK        (64u << 20)
#define MINCORE_PAGES     (1u << 15)  /* 32 KiB of vector per call */

/* Long enough that no legitimate gather comes near it: a full max_rows reply is
 * 21 MB, which is under a second of 2.5 GbE. */
#define SOCK_TIMEOUT_SECS 30

/* The pending connection accept() could not take stays queued, so a retry without
 * a pause fails the same way as fast as the core allows. */
#define ACCEPT_BACKOFF_NS 100000000L

/* An id array we are not going to use still has to come off the socket, or the next
 * header lands in the middle of it. Bounded, because a wildly wrong count is a broken
 * client and not a reason to read for a minute. */
#define MAX_DRAIN_BYTES   (64u << 20)

struct server {
    uint8_t *map;
    size_t map_len;
    size_t page_size;
    uint64_t base_row;      /* as STAT reports the range */
    uint64_t row_count;
    uint32_t base32;        /* as the gather loop checks it, ids being u32 on the wire */
    uint32_t rows32;
    uint32_t max_rows;
    int threads;
};

struct conn {
    int fd;
    const struct server *srv;
};

static volatile sig_atomic_t stopping;
static atomic_int live_conns;
static atomic_ullong served_requests;

/* ------------------------------------------------------------------ logging */

static void log_line(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

static void log_line(const char *fmt, ...)
{
    char line[512];
    struct timespec ts;
    struct tm tm;
    va_list ap;
    size_t n;
    int m;
    ssize_t w;

    clock_gettime(CLOCK_REALTIME, &ts);
    gmtime_r(&ts.tv_sec, &tm);
    n = strftime(line, sizeof line, "%Y-%m-%dT%H:%M:%SZ ", &tm);

    va_start(ap, fmt);
    m = vsnprintf(line + n, sizeof line - n - 1, fmt, ap);
    va_end(ap);
    if (m < 0)
        return;
    n += (size_t)m < sizeof line - n - 1 ? (size_t)m : sizeof line - n - 1;
    line[n++] = '\n';

    /* One write from any thread: stderr's own buffering would interleave lines. */
    w = write(STDERR_FILENO, line, n);
    (void)w;
}

/* Anything a peer or a stuck resource can make this daemon print goes through
 * log_capped, one line a second per call site. These nodes are k3s control-plane
 * members and stderr ends up on the disk etcd writes its WAL to, so a flood there
 * is worse than the event that caused it. Startup and shutdown lines, which happen
 * once, use log_line directly. */
struct ratelimit {
    pthread_mutex_t lock;
    struct timespec last;
    uint64_t dropped;
    int primed;
};

#define RATELIMIT_INIT { PTHREAD_MUTEX_INITIALIZER, { 0, 0 }, 0, 0 }

/* Lines dropped since the last one that got through, or -1 to drop this one too. */
static int64_t ratelimit_take(struct ratelimit *rl)
{
    struct timespec now;
    int64_t since_ms, dropped;

    clock_gettime(CLOCK_MONOTONIC, &now);
    pthread_mutex_lock(&rl->lock);
    since_ms = (int64_t)(now.tv_sec - rl->last.tv_sec) * 1000 +
               (now.tv_nsec - rl->last.tv_nsec) / 1000000;
    if (rl->primed && since_ms < 1000) {
        rl->dropped++;
        pthread_mutex_unlock(&rl->lock);
        return -1;
    }
    dropped = (int64_t)rl->dropped;
    rl->dropped = 0;
    rl->last = now;
    rl->primed = 1;
    pthread_mutex_unlock(&rl->lock);
    return dropped;
}

static void log_capped(struct ratelimit *rl, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void log_capped(struct ratelimit *rl, const char *fmt, ...)
{
    char line[384];
    int64_t dropped = ratelimit_take(rl);
    va_list ap;
    int m;

    if (dropped < 0)
        return;

    va_start(ap, fmt);
    m = vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    if (m < 0)
        return;
    if (dropped > 0)
        log_line("%s (and %" PRId64 " more not logged)", line, dropped);
    else
        log_line("%s", line);
}

/* The static gives every call site its own budget without a limiter to declare. */
#define LOG_CAPPED(fmt, ...) do {                               \
        static struct ratelimit rl_ = RATELIMIT_INIT;           \
        log_capped(&rl_, fmt, ##__VA_ARGS__);                   \
    } while (0)

/* ------------------------------------------------------------- byte order */

/* The wire is little-endian whatever the host is. Shifts say so explicitly and
 * compile down to a plain load on the machines this runs on. */
static inline uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline void wr_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static inline void wr_u64(uint8_t *p, uint64_t v)
{
    wr_u32(p, (uint32_t)v);
    wr_u32(p + 4, (uint32_t)(v >> 32));
}

/* --------------------------------------------------------------- socket io */

static int read_full(int fd, void *buf, size_t n)
{
    uint8_t *p = buf;
    size_t got = 0;

    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r > 0) {
            got += (size_t)r;
            continue;
        }
        if (r == 0)
            return got == 0 ? RD_EOF : RD_ERR;
        if (errno == EINTR)
            continue;
        /* SO_RCVTIMEO fired. Nothing read yet means the peer is simply idle and it
         * is up to the caller whether that is allowed; part way in it is a stall. */
        if ((errno == EAGAIN || errno == EWOULDBLOCK) && got == 0)
            return RD_TIMEOUT;
        return RD_ERR;
    }
    return RD_OK;
}

static int write_full(int fd, const void *buf, size_t n)
{
    const uint8_t *p = buf;
    size_t sent = 0;

    while (sent < n) {
        ssize_t w = write(fd, p + sent, n - sent);
        if (w > 0) {
            sent += (size_t)w;
            continue;
        }
        if (w < 0 && errno == EINTR)
            continue;
        return -1;
    }
    return 0;
}

static int drain(int fd, uint8_t *scratch, size_t scratch_len, uint64_t bytes)
{
    if (bytes > MAX_DRAIN_BYTES)
        return -1;
    while (bytes > 0) {
        size_t n = bytes < scratch_len ? (size_t)bytes : scratch_len;
        if (read_full(fd, scratch, n) != RD_OK)
            return -1;
        bytes -= n;
    }
    return 0;
}

/* ------------------------------------------------------------------- table */

static uint64_t resident_pages(const struct server *s)
{
    static unsigned char vec[MINCORE_PAGES];
    static pthread_mutex_t vec_lock = PTHREAD_MUTEX_INITIALIZER;
    const size_t span = (size_t)MINCORE_PAGES * s->page_size;
    uint64_t resident = 0;
    size_t off;

    /* mincore walks page tables, so this is milliseconds over 23.8 GiB. STAT is a
     * health check and never sits on the gather path, so that is fine. */
    pthread_mutex_lock(&vec_lock);
    for (off = 0; off < s->map_len; off += span) {
        size_t len = s->map_len - off < span ? s->map_len - off : span;
        size_t pages = (len + s->page_size - 1) / s->page_size;
        size_t i;

        if (mincore(s->map + off, len, vec) != 0) {
            log_line("mincore failed at offset %zu: %s", off, strerror(errno));
            pthread_mutex_unlock(&vec_lock);
            return 0;
        }
        for (i = 0; i < pages; i++)
            resident += vec[i] & 1u;
    }
    pthread_mutex_unlock(&vec_lock);
    return resident;
}

static void warm_map(const struct server *s)
{
    static volatile uint64_t sink;   /* the touch loop has no other observable effect */
    struct timespec t0, t1;
    double secs;
    size_t off;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (s->map_len > 0)
        madvise(s->map, s->map_len < WARM_CHUNK ? s->map_len : WARM_CHUNK, MADV_WILLNEED);

    for (off = 0; off < s->map_len; off += WARM_CHUNK) {
        size_t len = s->map_len - off < WARM_CHUNK ? s->map_len - off : WARM_CHUNK;
        size_t next = off + WARM_CHUNK;
        uint64_t acc = 0;
        size_t p;

        /* Advise one chunk ahead so the readahead for the next chunk is already in
         * flight while this one is being faulted in. */
        if (next < s->map_len) {
            size_t nlen = s->map_len - next < WARM_CHUNK ? s->map_len - next : WARM_CHUNK;
            madvise(s->map + next, nlen, MADV_WILLNEED);
        }
        for (p = 0; p < len; p += s->page_size)
            acc += s->map[off + p];
        sink += acc;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    log_line("warm: touched %.2f GiB in %.1f s (%.0f MB/s)",
             (double)s->map_len / 1073741824.0, secs,
             secs > 0 ? (double)s->map_len / secs / 1e6 : 0.0);
}

static void report_residency(const struct server *s)
{
    uint64_t pages = resident_pages(s);
    uint64_t total = (s->map_len + s->page_size - 1) / s->page_size;

    log_line("resident: %" PRIu64 " / %" PRIu64 " pages (%.2f GiB, %.1f%%)",
             pages, total, (double)pages * (double)s->page_size / 1073741824.0,
             total ? 100.0 * (double)pages / (double)total : 0.0);
}

/* ------------------------------------------------------------------ serving */

static int gather_rows(const struct server *s, const uint8_t *ids, uint32_t count, uint8_t *dst)
{
    const uint8_t *table = s->map;
    uint32_t base = s->base32;
    uint32_t rows = s->rows32;
    uint32_t i;

    for (i = 0; i < count; i++) {
        uint32_t id = rd_u32(ids + 4u * (size_t)i);
        uint32_t local = id - base;

        /* Ids below base wrap past rows, so one unsigned compare covers both ends. */
        if (local >= rows)
            return ST_RANGE;
        memcpy(dst + (size_t)i * ROW_BYTES, table + (size_t)local * ROW_BYTES, ROW_BYTES);
    }
    return ST_OK;
}

static int send_response(int fd, uint8_t *buf, uint32_t req_id, uint32_t status,
                         uint32_t count, size_t body)
{
    wr_u32(buf, RESP_MAGIC);
    wr_u32(buf + 4, req_id);
    wr_u32(buf + 8, status);
    wr_u32(buf + 12, count);
    return write_full(fd, buf, HDR_BYTES + body);
}

static void fill_stat(const struct server *s, uint8_t *body)
{
    uint64_t pages = resident_pages(s);

    wr_u64(body, s->base_row);
    wr_u64(body + 8, s->row_count);
    wr_u32(body + 16, ROW_BYTES);
    wr_u32(body + 20, pages > UINT32_MAX ? UINT32_MAX : (uint32_t)pages);
    wr_u64(body + 24, atomic_load_explicit(&served_requests, memory_order_relaxed));
}

static void *serve_conn(void *arg)
{
    struct conn c = *(struct conn *)arg;
    const struct server *s = c.srv;
    /* Both buffers are sized for max_rows once, here, so the request loop allocates
     * nothing. resp holds the header and the rows contiguously: one write per reply. */
    uint8_t *resp = malloc(HDR_BYTES + (size_t)s->max_rows * ROW_BYTES);
    uint8_t *ids = malloc((size_t)s->max_rows * 4);

    free(arg);
    if (!resp || !ids) {
        /* A response echoes the req_id of a request, so this waits for the one the
         * client is about to send and answers that, rather than closing on it. */
        uint8_t hdr[HDR_BYTES];

        LOG_CAPPED("connection buffers: out of memory");
        if (read_full(c.fd, hdr, HDR_BYTES) == RD_OK) {
            uint32_t req_id = rd_u32(hdr + 8);

            send_response(c.fd, hdr, req_id, ST_INTERNAL, 0, 0);
        }
        goto done;
    }

    for (;;) {
        uint8_t hdr[HDR_BYTES];
        uint32_t magic, req_id, count;
        uint8_t version, op;
        int r = read_full(c.fd, hdr, HDR_BYTES);

        if (r == RD_EOF)
            break;
        /* Between requests the client is allowed to be quiet for as long as it
         * likes: it holds this connection for the life of the model and has no
         * reconnect path. SO_KEEPALIVE is what reaps a peer that went away. */
        if (r == RD_TIMEOUT)
            continue;
        if (r != RD_OK) {
            LOG_CAPPED("connection dropped part way through a request header");
            break;
        }

        magic = rd_u32(hdr);
        version = hdr[4];
        op = hdr[5];
        req_id = rd_u32(hdr + 8);
        count = rd_u32(hdr + 12);

        if (magic != REQ_MAGIC || version != PROTO_VER) {
            /* Framing is gone and there is no way to find the next header, so this
             * one is answered and then the connection goes. */
            LOG_CAPPED("bad header: magic %08x version %u", magic, version);
            send_response(c.fd, resp, req_id, ST_BAD_HEADER, 0, 0);
            break;
        }
        /* Counted on arrival rather than on the reply, so a STAT includes itself and
         * a health check polling it sees the number move. */
        atomic_fetch_add_explicit(&served_requests, 1, memory_order_relaxed);

        if (op == OP_GATHER) {
            int st;

            if (count > s->max_rows) {
                LOG_CAPPED("request of %u rows over max_rows %u", count, s->max_rows);
                if (drain(c.fd, ids, (size_t)s->max_rows * 4, (uint64_t)count * 4) != 0) {
                    send_response(c.fd, resp, req_id, ST_TOO_MANY, 0, 0);
                    break;
                }
                if (send_response(c.fd, resp, req_id, ST_TOO_MANY, 0, 0) != 0)
                    break;
                continue;
            }
            if (count > 0 && read_full(c.fd, ids, (size_t)count * 4) != RD_OK) {
                LOG_CAPPED("short read on %u row ids", count);
                break;
            }
            st = gather_rows(s, ids, count, resp + HDR_BYTES);
            if (st != ST_OK) {
                if (send_response(c.fd, resp, req_id, (uint32_t)st, 0, 0) != 0)
                    break;
                continue;
            }
            if (send_response(c.fd, resp, req_id, ST_OK, count, (size_t)count * ROW_BYTES) != 0)
                break;
            continue;
        }

        /* PING and STAT carry no ids, but draining whatever count claims keeps a
         * confused client's stream in sync instead of failing the next request.
         * Past the drain limit there is no way back to a header boundary, so this
         * answers the way GATHER does and then goes. */
        if (count > 0 && drain(c.fd, ids, (size_t)s->max_rows * 4, (uint64_t)count * 4) != 0) {
            send_response(c.fd, resp, req_id, ST_TOO_MANY, 0, 0);
            break;
        }

        if (op == OP_PING) {
            if (send_response(c.fd, resp, req_id, ST_OK, 0, 0) != 0)
                break;
        } else if (op == OP_STAT) {
            fill_stat(s, resp + HDR_BYTES);
            if (send_response(c.fd, resp, req_id, ST_OK, 0, STAT_BYTES) != 0)
                break;
        } else {
            LOG_CAPPED("unknown op %u", op);
            if (send_response(c.fd, resp, req_id, ST_BAD_OP, 0, 0) != 0)
                break;
        }
    }

done:
    free(resp);
    free(ids);
    close(c.fd);
    atomic_fetch_sub(&live_conns, 1);
    return NULL;
}

/* --------------------------------------------------------------- lifecycle */

static void on_signal(int sig)
{
    (void)sig;
    stopping = 1;
}

static int listen_on(int port)
{
    struct sockaddr_in addr;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;

    if (fd < 0) {
        log_line("socket: %s", strerror(errno));
        return -1;
    }
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one) != 0)
        log_line("SO_REUSEADDR: %s", strerror(errno));

    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);

    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) != 0) {
        log_line("bind port %d: %s", port, strerror(errno));
        close(fd);
        return -1;
    }
    if (listen(fd, 16) != 0) {
        log_line("listen: %s", strerror(errno));
        close(fd);
        return -1;
    }
    return fd;
}

/* Nagle would hold a 16 byte header back waiting for the id array and cost up to
 * 40 ms, which is most of an engine step. The timeouts are for a client that stops
 * part way through a request: there are only --threads slots and a stalled one
 * would otherwise own its slot for good. Keepalive covers the other half of that,
 * a peer that went away without ever sending a FIN. */
static void tune_conn(int fd)
{
    struct timeval tv = { SOCK_TIMEOUT_SECS, 0 };
    int one = 1;

    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one) != 0)
        LOG_CAPPED("TCP_NODELAY: %s", strerror(errno));
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) != 0)
        LOG_CAPPED("SO_RCVTIMEO: %s", strerror(errno));
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv) != 0)
        LOG_CAPPED("SO_SNDTIMEO: %s", strerror(errno));
    if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one) != 0)
        LOG_CAPPED("SO_KEEPALIVE: %s", strerror(errno));
}

static int parse_u64(const char *text, uint64_t *out)
{
    char *end;
    unsigned long long v;

    errno = 0;
    v = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0')
        return -1;
    *out = (uint64_t)v;
    return 0;
}

static void usage(FILE *out)
{
    fprintf(out,
        "usage: rowserverd --file PATH --base-row N --row-count N [options]\n"
        "\n"
        "  --file PATH       flat file of row_count x %u byte rows\n"
        "  --base-row N      global id of this file's first row\n"
        "  --row-count N     rows in this file\n"
        "  --port N          listen port (default %d)\n"
        "  --max-rows N      largest gather accepted, per request (default %u)\n"
        "  --threads N       concurrent connections, one thread each (default %d)\n"
        "  --warm            fault the whole file in at startup (default)\n"
        "  --no-warm         skip the warm pass and let rows arrive on demand\n"
        "  --help\n"
        "\n"
        "The two halves of the current table are:\n"
        "  node1  --base-row 0          --row-count 160000768\n"
        "  node2  --base-row 160000768  --row-count 160000768\n",
        ROW_BYTES, DEFAULT_PORT, DEFAULT_MAX_ROWS, DEFAULT_THREADS);
}

int main(int argc, char **argv)
{
    static const struct option opts[] = {
        { "file",      required_argument, NULL, 'f' },
        { "base-row",  required_argument, NULL, 'b' },
        { "row-count", required_argument, NULL, 'r' },
        { "port",      required_argument, NULL, 'p' },
        { "max-rows",  required_argument, NULL, 'm' },
        { "threads",   required_argument, NULL, 't' },
        { "warm",      no_argument,       NULL, 'w' },
        { "no-warm",   no_argument,       NULL, 'W' },
        { "help",      no_argument,       NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };
    static struct server s;   /* connection threads keep a pointer to this */
    struct sigaction sa;
    pthread_attr_t attr;
    const char *path = NULL;
    uint64_t base_row = 0, row_count = 0, n;
    int have_base = 0, have_count = 0, warm = 1;
    int port = DEFAULT_PORT, lfd, fd, opt;
    struct stat st;

    s.max_rows = DEFAULT_MAX_ROWS;
    s.threads = DEFAULT_THREADS;

    while ((opt = getopt_long(argc, argv, "f:b:r:p:m:t:h", opts, NULL)) != -1) {
        switch (opt) {
        case 'f':
            path = optarg;
            break;
        case 'b':
            if (parse_u64(optarg, &base_row) != 0) {
                log_line("bad --base-row: %s", optarg);
                return 2;
            }
            have_base = 1;
            break;
        case 'r':
            if (parse_u64(optarg, &row_count) != 0) {
                log_line("bad --row-count: %s", optarg);
                return 2;
            }
            have_count = 1;
            break;
        case 'p':
            if (parse_u64(optarg, &n) != 0 || n < 1 || n > 65535) {
                log_line("bad --port: %s", optarg);
                return 2;
            }
            port = (int)n;
            break;
        case 'm':
            if (parse_u64(optarg, &n) != 0 || n < 1 || n > MAX_MAX_ROWS) {
                log_line("bad --max-rows: %s (1 to %u)", optarg, MAX_MAX_ROWS);
                return 2;
            }
            s.max_rows = (uint32_t)n;
            break;
        case 't':
            if (parse_u64(optarg, &n) != 0 || n < 1 || n > MAX_THREADS) {
                log_line("bad --threads: %s (1 to %d)", optarg, MAX_THREADS);
                return 2;
            }
            s.threads = (int)n;
            break;
        case 'w':
            warm = 1;
            break;
        case 'W':
            warm = 0;
            break;
        case 'h':
            usage(stdout);
            return 0;
        default:
            usage(stderr);
            return 2;
        }
    }

    if (!path || !have_base || !have_count) {
        usage(stderr);
        return 2;
    }
    if (row_count == 0) {
        log_line("--row-count must be positive");
        return 2;
    }
    /* Each term is bounded on its own first: base_row + row_count on two u64 wraps
     * for a large enough base_row and lets a typo through as a valid range, and the
     * gather loop would then serve the wrong rows with status 0. */
    if (base_row > UINT32_MAX || row_count > UINT32_MAX ||
        base_row + row_count > UINT32_MAX) {
        log_line("row ids are u32 on the wire: base-row %" PRIu64 " plus row-count %"
                 PRIu64 " does not fit in %u", base_row, row_count, UINT32_MAX);
        return 2;
    }

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        log_line("open %s: %s", path, strerror(errno));
        return 1;
    }
    if (fstat(fd, &st) != 0) {
        log_line("stat %s: %s", path, strerror(errno));
        close(fd);
        return 1;
    }

    /* The cheap guard against a server pointed at the wrong half: a half that does
     * not have exactly the rows it claims would just serve wrong embeddings, and
     * nothing downstream would notice. */
    if ((uint64_t)st.st_size != row_count * ROW_BYTES) {
        log_line("%s is %" PRIu64 " bytes, expected %" PRIu64 " for %" PRIu64
                 " rows of %u", path, (uint64_t)st.st_size,
                 row_count * ROW_BYTES, row_count, ROW_BYTES);
        close(fd);
        return 1;
    }

    s.map_len = (size_t)st.st_size;
    s.map = mmap(NULL, s.map_len, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (s.map == MAP_FAILED) {
        log_line("mmap %s: %s", path, strerror(errno));
        return 1;
    }
    s.page_size = (size_t)sysconf(_SC_PAGESIZE);
    s.base_row = base_row;
    s.row_count = row_count;
    s.base32 = (uint32_t)base_row;
    s.rows32 = (uint32_t)row_count;

    log_line("rowserverd: %s rows [%" PRIu64 ", %" PRIu64 ") %" PRIu64
             " bytes, port %d, max_rows %u, threads %d",
             path, base_row, base_row + row_count, (uint64_t)s.map_len,
             port, s.max_rows, s.threads);

    if (warm)
        warm_map(&s);
    /* Gathers are scattered across the whole half by construction, so readahead
     * around a faulted row is bandwidth spent on rows nobody asked for. */
    madvise(s.map, s.map_len, MADV_RANDOM);
    report_residency(&s);

    signal(SIGPIPE, SIG_IGN);
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;   /* no SA_RESTART, so accept() returns EINTR and the loop ends */
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);

    lfd = listen_on(port);
    if (lfd < 0) {
        munmap(s.map, s.map_len);
        return 1;
    }

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&attr, 512 * 1024);   /* the big buffers are on the heap */

    while (!stopping) {
        struct conn *c;
        pthread_t tid;
        int cfd = accept(lfd, NULL, NULL);
        int rc;

        if (cfd < 0) {
            if (errno == EINTR || errno == ECONNABORTED)
                continue;
            if (errno == EMFILE || errno == ENFILE) {
                /* Out of descriptors, a condition that clears itself once some
                 * connection ends. Wait for that rather than spinning on the
                 * connection still sitting in the queue. */
                struct timespec backoff = { 0, ACCEPT_BACKOFF_NS };

                LOG_CAPPED("accept: %s", strerror(errno));
                nanosleep(&backoff, NULL);
                continue;
            }
            log_line("accept: %s", strerror(errno));
            break;
        }
        tune_conn(cfd);

        if (atomic_fetch_add(&live_conns, 1) >= s.threads) {
            atomic_fetch_sub(&live_conns, 1);
            LOG_CAPPED("refusing connection: %d already open", s.threads);
            close(cfd);
            continue;
        }
        c = malloc(sizeof *c);
        if (!c) {
            atomic_fetch_sub(&live_conns, 1);
            close(cfd);
            continue;
        }
        c->fd = cfd;
        c->srv = &s;
        rc = pthread_create(&tid, &attr, serve_conn, c);
        if (rc != 0) {
            LOG_CAPPED("pthread_create: %s", strerror(rc));
            atomic_fetch_sub(&live_conns, 1);
            free(c);
            close(cfd);
        }
    }

    /* Connection threads hold nothing but read-only mappings, so there is no state
     * worth draining on the way out. The mapping is left in place rather than
     * unmapped: a thread could still be mid-copy out of it, and exit tears it down. */
    close(lfd);
    pthread_attr_destroy(&attr);
    log_line("shutting down after %llu requests",
             (unsigned long long)atomic_load(&served_requests));
    return 0;
}
