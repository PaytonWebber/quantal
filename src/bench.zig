//! Benchmark harness: recall and throughput against exact FP32 ground truth.
//!
//! Usage:
//!   qj-bench synthetic [--n 10000] [--dim 128] [--queries 1000]
//!   qj-bench fvecs <base.fvecs> [--query-file q.fvecs] [--max-base N] [--queries N]
//!   qj-bench glove <vectors.txt> [--max-base N] [--queries N]
//!
//! Common flags: --ef <ef_construction> (default 200), --no-normalize,
//! --seed <u64>.
//!
//! File datasets are L2-normalized by default so inner product equals cosine
//! similarity; ground truth is always recomputed with exact FP32 inner
//! products on the data as indexed. When no query file is given, the last
//! `--queries` vectors are held out of the index and used as queries.

const std = @import("std");
const qj = @import("quantajump");

const supported_dims = [_]usize{ 25, 50, 64, 100, 128, 200, 300, 768, 1536 };
const max_edges = 16;
const default_stage1_widths = [_]usize{ 16, 32, 64, 128 };
const top_k = 10;

const Mode = enum { synthetic, fvecs, glove };

const Config = struct {
    mode: Mode,
    path: ?[]const u8 = null,
    query_path: ?[]const u8 = null,
    n: usize = 10_000,
    dim: usize = 128,
    n_queries: usize = 1_000,
    max_base: usize = 100_000,
    normalize: bool = true,
    ef_construction: usize = 200,
    seed: u64 = 42,
    stage1_widths: []const usize = &default_stage1_widths,
    symmetric: bool = true,
    profile: bool = false,
    rerank_mode: qj.index.RerankStore = .sq8,
    tq_plus: bool = false,
    threads: usize = 1,
    /// turbovec/paper-style report: recall1@k for k in 1..64.
    recall_curve: bool = false,
};

const Dataset = struct {
    base: []f32,
    queries: []f32,
    n: usize,
    n_queries: usize,
    dim: usize,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    const config = try parseArgs(allocator, init.minimal.args);
    var dataset = try loadDataset(allocator, init.io, config);
    defer allocator.free(dataset.base);
    defer allocator.free(dataset.queries);

    if (config.normalize and config.mode != .synthetic) {
        normalizeAll(dataset.base, dataset.dim);
        normalizeAll(dataset.queries, dataset.dim);
    }

    std.debug.print(
        "dataset: n={} queries={} dim={} | ef_construction={} max_edges={} seed={}\n\n",
        .{ dataset.n, dataset.n_queries, dataset.dim, config.ef_construction, max_edges, config.seed },
    );

    inline for (supported_dims) |dim| {
        if (dim == dataset.dim) {
            return runBench(dim, allocator, init.io, &dataset, config);
        }
    }
    fatal("unsupported dimension {d}; compiled for: {any}", .{ dataset.dim, supported_dims });
}

const Stopwatch = struct {
    io: std.Io,
    started: std.Io.Timestamp,

    fn begin(io: std.Io) Stopwatch {
        return .{ .io = io, .started = std.Io.Timestamp.now(io, .awake) };
    }

    fn read(self: *const Stopwatch) u64 {
        const now = std.Io.Timestamp.now(self.io, .awake);
        return @intCast(self.started.durationTo(now).nanoseconds);
    }

    fn reset(self: *Stopwatch) void {
        self.started = std.Io.Timestamp.now(self.io, .awake);
    }
};

fn runBench(comptime dim: usize, allocator: std.mem.Allocator, io: std.Io, dataset: *const Dataset, config: Config) !void {
    const Idx = qj.Index(dim, max_edges);

    var timer = Stopwatch.begin(io);
    var index = try Idx.init(allocator, config.ef_construction, config.seed);
    defer index.deinit(allocator);
    index.rerank_store = config.rerank_mode;
    index.tq_plus = config.tq_plus;
    if (config.threads > 1 and !config.tq_plus) {
        const ids = try allocator.alloc(u64, dataset.n);
        defer allocator.free(ids);
        for (ids, 0..) |*id, i| id.* = i;
        try index.addBatch(allocator, ids, dataset.base[0 .. dataset.n * dim], config.threads);
    } else {
        for (0..dataset.n) |i| {
            try index.add(allocator, i, dataset.base[i * dim ..][0..dim]);
        }
        try index.freeze(allocator);
    }
    const build_ns = timer.read();
    std.debug.print("build: {d:.2}s ({d:.0} vectors/s)\n", .{
        @as(f64, @floatFromInt(build_ns)) / 1e9,
        @as(f64, @floatFromInt(dataset.n)) * 1e9 / @as(f64, @floatFromInt(build_ns)),
    });
    printMemory(Idx, &index, dataset.n);

    // Exact FP32 inner-product ground truth.
    timer.reset();
    const truth = try allocator.alloc([top_k]u64, dataset.n_queries);
    defer allocator.free(truth);
    for (0..dataset.n_queries) |q| {
        truth[q] = exactTopK(dataset.base, dataset.queries[q * dim ..][0..dim], dim);
    }
    std.debug.print("ground truth (exact fp32): {d:.2}s\n\n", .{
        @as(f64, @floatFromInt(timer.read())) / 1e9,
    });

    if (config.profile) {
        return profilePhases(Idx, io, &index, dataset, config);
    }
    if (config.recall_curve) {
        return recallCurve(Idx, allocator, io, &index, dataset, config, truth);
    }

    std.debug.print("{s:>5} {s:>10} {s:>10} {s:>11} {s:>9} {s:>11}\n", .{
        "m", "recall1@1", "recall1@10", "recall10@10", "QPS", "us/query",
    });
    for (config.stage1_widths) |m| {
        var ctx = try Idx.SearchContext.init(allocator, &index, m);
        defer ctx.deinit(allocator);
        ctx.symmetric = config.symmetric;

        var out: [top_k]qj.SearchResult = undefined;
        var hits_1: usize = 0;
        var hits_10: usize = 0;
        var overlap: usize = 0;

        timer.reset();
        for (0..dataset.n_queries) |q| {
            const count = index.search(&ctx, dataset.queries[q * dim ..][0..dim], &out);
            const expected = &truth[q];
            if (count > 0 and out[0].id == expected[0]) hits_1 += 1;
            for (out[0..count]) |result| {
                if (result.id == expected[0]) hits_10 += 1;
                if (std.mem.indexOfScalar(u64, expected, result.id) != null) overlap += 1;
            }
        }
        const search_ns = timer.read();

        const nq: f64 = @floatFromInt(dataset.n_queries);
        std.debug.print("{d:>5} {d:>10.3} {d:>10.3} {d:>11.3} {d:>9.0} {d:>11.1}\n", .{
            m,
            @as(f64, @floatFromInt(hits_1)) / nq,
            @as(f64, @floatFromInt(hits_10)) / nq,
            @as(f64, @floatFromInt(overlap)) / (nq * top_k),
            nq * 1e9 / @as(f64, @floatFromInt(search_ns)),
            @as(f64, @floatFromInt(search_ns)) / (nq * 1e3),
        });
    }
}

/// turbovec/paper-style report: how often the exact top-1 appears within the
/// approximate top-k, for k in {1,2,4,8,16,32,64} (their Fig. 5 metric).
fn recallCurve(
    comptime Idx: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    index: *const Idx,
    dataset: *const Dataset,
    config: Config,
    truth: []const [top_k]u64,
) !void {
    const dim = Idx.dimension;
    const ks = [_]usize{ 1, 2, 4, 8, 16, 32, 64 };

    std.debug.print("{s:>5}", .{"m"});
    for (ks) |k| std.debug.print("    1@{d:<3}", .{k});
    std.debug.print(" {s:>9} {s:>11}\n", .{ "QPS", "us/query" });

    const results = try allocator.alloc(qj.SearchResult, dataset.n_queries * 64);
    defer allocator.free(results);
    const counts = try allocator.alloc(usize, dataset.n_queries);
    defer allocator.free(counts);

    for (config.stage1_widths) |m| {
        var timer = Stopwatch.begin(io);
        try index.searchBatch(
            allocator,
            dataset.queries[0 .. dataset.n_queries * dim],
            64,
            m,
            config.threads,
            config.symmetric,
            results,
            counts,
        );
        const search_ns = timer.read();

        var hits: [ks.len]usize = @splat(0);
        for (0..dataset.n_queries) |q| {
            const want = truth[q][0];
            for (results[q * 64 ..][0..counts[q]], 0..) |result, rank| {
                if (result.id == want) {
                    for (ks, 0..) |k, ki| {
                        if (rank < k) hits[ki] += 1;
                    }
                    break;
                }
            }
        }

        const nq: f64 = @floatFromInt(dataset.n_queries);
        std.debug.print("{d:>5}", .{m});
        for (hits) |h| std.debug.print("   {d:.4}", .{@as(f64, @floatFromInt(h)) / nq});
        std.debug.print(" {d:>9.0} {d:>11.1}\n", .{
            nq * 1e9 / @as(f64, @floatFromInt(search_ns)),
            @as(f64, @floatFromInt(search_ns)) / (nq * 1e3),
        });
    }
}

/// Times each query phase separately by driving the pipeline manually.
fn profilePhases(comptime Idx: type, io: std.Io, index: *const Idx, dataset: *const Dataset, config: Config) !void {
    const dim = Idx.dimension;
    const allocator = std.heap.smp_allocator;

    std.debug.print("{s:>5} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10}\n", .{
        "m", "rotate us", "1bit us", "route us", "score us", "total us",
    });
    for (config.stage1_widths) |m| {
        var ctx = try Idx.SearchContext.init(allocator, index, m);
        defer ctx.deinit(allocator);
        ctx.symmetric = config.symmetric;

        var rotate_ns: u64 = 0;
        var prep_ns: u64 = 0;
        var route_ns: u64 = 0;
        var score_ns: u64 = 0;
        var out: [top_k]qj.SearchResult = undefined;
        var sink: f64 = 0;

        for (0..dataset.n_queries) |q| {
            const query = dataset.queries[q * dim ..][0..dim];
            var timer = Stopwatch.begin(io);

            index.rotation.apply(query, ctx.rotated_query);
            rotate_ns += timer.read();

            timer.reset();
            const query_bits = Idx.Graph.BitVec.fromF32(ctx.rotated_query);
            if (ctx.symmetric) {
                qj.turboquant.quantizeQuery(ctx.rotated_query, ctx.decoded_query);
            } else {
                @memcpy(ctx.decoded_query, ctx.rotated_query);
            }
            const consts = qj.turboquant.buildScoreLut(ctx.decoded_query, ctx.score_lut);
            prep_ns += timer.read();

            timer.reset();
            const candidates = index.routing.route(&query_bits, ctx.m, &ctx.scratch);
            route_ns += timer.read();

            timer.reset();
            var count: usize = 0;
            if (index.hasRerankStore()) {
                const pool_size = @min(ctx.rerank_factor * out.len, ctx.rerank_pool.len);
                var pool = qj.heap.TopK.fromBuffer(ctx.rerank_pool[0..pool_size]);
                for (candidates) |candidate| {
                    const payload = &index.payloads.items[@intCast(candidate.id)];
                    pool.offer(.{ .id = candidate.id, .score = payload.scoreLut(ctx.score_lut, consts.a, consts.b) });
                }
                const pooled = pool.sortDescending();
                var topk = qj.heap.TopK.fromBuffer(&out);
                for (ctx.rerank_pool[0..pooled]) |entry| {
                    const internal: usize = @intCast(entry.id);
                    topk.offer(.{
                        .id = index.payloads.items[internal].id,
                        .score = index.rerankScore(internal, ctx.rotated_query),
                    });
                }
                count = topk.sortDescending();
            } else {
                var topk = qj.heap.TopK.fromBuffer(&out);
                for (candidates) |candidate| {
                    const payload = &index.payloads.items[@intCast(candidate.id)];
                    topk.offer(.{ .id = payload.id, .score = payload.scoreLut(ctx.score_lut, consts.a, consts.b) });
                }
                count = topk.sortDescending();
            }
            score_ns += timer.read();
            for (out[0..count]) |r| sink += r.score;
        }

        const nq: f64 = @floatFromInt(dataset.n_queries);
        std.debug.print("{d:>5} {d:>10.1} {d:>10.1} {d:>10.1} {d:>10.1} {d:>10.1}\n", .{
            m,
            @as(f64, @floatFromInt(rotate_ns)) / (nq * 1e3),
            @as(f64, @floatFromInt(prep_ns)) / (nq * 1e3),
            @as(f64, @floatFromInt(route_ns)) / (nq * 1e3),
            @as(f64, @floatFromInt(score_ns)) / (nq * 1e3),
            @as(f64, @floatFromInt(rotate_ns + prep_ns + route_ns + score_ns)) / (nq * 1e3),
        });
        std.mem.doNotOptimizeAway(sink);
    }
}

/// Exact top-k base indices by FP32 inner product (insertion sort, k tiny).
fn exactTopK(base: []const f32, query: []const f32, dim: usize) [top_k]u64 {
    var ids: [top_k]u64 = @splat(std.math.maxInt(u64));
    var scores: [top_k]f32 = @splat(-std.math.inf(f32));
    const n = base.len / dim;
    for (0..n) |i| {
        const row = base[i * dim ..][0..dim];
        var score: f32 = 0;
        for (row, query) |x, y| score += x * y;
        if (score <= scores[top_k - 1]) continue;
        var pos: usize = top_k - 1;
        while (pos > 0 and score > scores[pos - 1]) : (pos -= 1) {
            scores[pos] = scores[pos - 1];
            ids[pos] = ids[pos - 1];
        }
        scores[pos] = score;
        ids[pos] = i;
    }
    return ids;
}

fn printMemory(comptime Idx: type, index: *const Idx, n: usize) void {
    const payload_bytes = n * @sizeOf(Idx.Payload);
    var graph_bytes: usize = 0;
    for (index.routing.layers.items) |*layer| {
        graph_bytes += layer.nodes.items.len * @sizeOf(Idx.Graph.Node);
        graph_bytes += layer.bit_vectors.items.len * @sizeOf(Idx.Graph.BitVec);
    }
    const rerank_bytes = index.originals.items.len * @sizeOf(f32) +
        index.sq8_codes.items.len + index.sq8_scales.items.len * @sizeOf(f32);
    const raw_bytes = n * Idx.dimension * @sizeOf(f32);
    std.debug.print(
        "memory: payloads {d:.1} MiB + graph {d:.1} MiB + rerank store ({s}) {d:.1} MiB vs raw fp32 {d:.1} MiB\n",
        .{
            @as(f64, @floatFromInt(payload_bytes)) / (1 << 20),
            @as(f64, @floatFromInt(graph_bytes)) / (1 << 20),
            @tagName(index.rerank_store),
            @as(f64, @floatFromInt(rerank_bytes)) / (1 << 20),
            @as(f64, @floatFromInt(raw_bytes)) / (1 << 20),
        },
    );
}

fn normalizeAll(data: []f32, dim: usize) void {
    var i: usize = 0;
    while (i < data.len) : (i += dim) {
        const row = data[i..][0..dim];
        var norm_sq: f32 = 0;
        for (row) |x| norm_sq += x * x;
        if (norm_sq > 0) {
            const inv = 1.0 / @sqrt(norm_sq);
            for (row) |*x| x.* *= inv;
        }
    }
}

fn loadDataset(allocator: std.mem.Allocator, io: std.Io, config: Config) !Dataset {
    switch (config.mode) {
        .synthetic => return generateSynthetic(allocator, config),
        .fvecs => {
            const base = try loadFvecs(allocator, io, config.path.?, config.max_base + config.n_queries);
            if (config.query_path) |qpath| {
                const queries = try loadFvecs(allocator, io, qpath, config.n_queries);
                if (queries.dim != base.dim) fatal("query dim {} != base dim {}", .{ queries.dim, base.dim });
                return .{
                    .base = base.data,
                    .queries = queries.data,
                    .n = base.count,
                    .n_queries = queries.count,
                    .dim = base.dim,
                };
            }
            return holdOutQueries(allocator, base.data, base.count, base.dim, config.n_queries);
        },
        .glove => {
            const base = try loadGlove(allocator, io, config.path.?, config.max_base + config.n_queries);
            return holdOutQueries(allocator, base.data, base.count, base.dim, config.n_queries);
        },
    }
}

fn generateSynthetic(allocator: std.mem.Allocator, config: Config) !Dataset {
    var prng = std.Random.DefaultPrng.init(config.seed +% 1);
    const rand = prng.random();
    const base = try allocator.alloc(f32, config.n * config.dim);
    const queries = try allocator.alloc(f32, config.n_queries * config.dim);
    for (base) |*x| x.* = rand.floatNorm(f32);
    for (queries) |*x| x.* = rand.floatNorm(f32);
    return .{
        .base = base,
        .queries = queries,
        .n = config.n,
        .n_queries = config.n_queries,
        .dim = config.dim,
    };
}

/// Splits the last n_queries vectors off as the query set.
fn holdOutQueries(allocator: std.mem.Allocator, data: []f32, count: usize, dim: usize, n_queries: usize) !Dataset {
    if (count <= n_queries) fatal("dataset has {} vectors, need more than {} for held-out queries", .{ count, n_queries });
    const n = count - n_queries;
    const queries = try allocator.dupe(f32, data[n * dim .. count * dim]);
    const base = try allocator.realloc(data, n * dim);
    return .{ .base = base, .queries = queries, .n = n, .n_queries = n_queries, .dim = dim };
}

const RawVectors = struct {
    data: []f32,
    count: usize,
    dim: usize,
};

/// TEXMEX .fvecs: repeated records of [i32 dim][dim f32], little-endian.
fn loadFvecs(allocator: std.mem.Allocator, io: std.Io, path: []const u8, max_vectors: usize) !RawVectors {
    const bytes = try readFile(allocator, io, path);
    defer allocator.free(bytes);
    if (bytes.len < 4) fatal("{s}: not an fvecs file", .{path});

    const dim: usize = @intCast(std.mem.readInt(u32, bytes[0..4], .little));
    if (dim == 0 or dim > 1 << 16) fatal("{s}: implausible fvecs dim {}", .{ path, dim });
    const record = 4 + dim * 4;

    const available = bytes.len / record;
    const count = @min(available, max_vectors);
    const data = try allocator.alloc(f32, count * dim);
    errdefer allocator.free(data);

    for (0..count) |i| {
        const rec = bytes[i * record ..][0..record];
        const rec_dim = std.mem.readInt(u32, rec[0..4], .little);
        if (rec_dim != dim) fatal("{s}: inconsistent dim at record {}", .{ path, i });
        for (0..dim) |j| {
            data[i * dim + j] = @bitCast(std.mem.readInt(u32, rec[4 + j * 4 ..][0..4], .little));
        }
    }
    return .{ .data = data, .count = count, .dim = dim };
}

/// GloVe text format: one "token v0 v1 ... vd-1" line per vector.
fn loadGlove(allocator: std.mem.Allocator, io: std.Io, path: []const u8, max_vectors: usize) !RawVectors {
    const bytes = try readFile(allocator, io, path);
    defer allocator.free(bytes);

    var lines = std.mem.tokenizeScalar(u8, bytes, '\n');
    const first = lines.next() orelse fatal("{s}: empty file", .{path});
    var dim: usize = 0;
    var probe = std.mem.tokenizeScalar(u8, first, ' ');
    _ = probe.next(); // token
    while (probe.next()) |_| dim += 1;
    if (dim == 0) fatal("{s}: no coordinates on first line", .{path});

    var data = try std.ArrayList(f32).initCapacity(allocator, @min(max_vectors, 400_000) * dim);
    errdefer data.deinit(allocator);

    lines.reset();
    var count: usize = 0;
    while (lines.next()) |line| {
        if (count == max_vectors) break;
        var fields = std.mem.tokenizeScalar(u8, line, ' ');
        _ = fields.next(); // token
        var parsed: usize = 0;
        while (fields.next()) |field| : (parsed += 1) {
            try data.append(allocator, std.fmt.parseFloat(f32, field) catch {
                fatal("{s}: bad float '{s}' at line {}", .{ path, field, count + 1 });
            });
        }
        if (parsed != dim) fatal("{s}: line {} has {} coords, expected {}", .{ path, count + 1, parsed, dim });
        count += 1;
    }
    return .{ .data = try data.toOwnedSlice(allocator), .count = count, .dim = dim };
}

fn readFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| {
        fatal("cannot read {s}: {s}", .{ path, @errorName(err) });
    };
}

fn parseArgs(allocator: std.mem.Allocator, args: std.process.Args) !Config {
    var list: std.ArrayList([]const u8) = .empty;
    defer list.deinit(allocator);
    var it = args.iterate();
    while (it.next()) |arg| try list.append(allocator, arg);
    const argv = list.items;

    if (argv.len < 2) usage();
    const mode = std.meta.stringToEnum(Mode, argv[1]) orelse usage();

    var config = Config{ .mode = mode };
    var i: usize = 2;
    if (mode != .synthetic) {
        if (argv.len < 3) usage();
        config.path = argv[2];
        i = 3;
    }
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--no-normalize")) {
            config.normalize = false;
        } else if (std.mem.eql(u8, arg, "--asymmetric")) {
            config.symmetric = false;
        } else if (std.mem.eql(u8, arg, "--no-exact")) {
            config.rerank_mode = .none;
        } else if (std.mem.eql(u8, arg, "--rerank")) {
            config.rerank_mode = std.meta.stringToEnum(qj.index.RerankStore, nextArg(argv, &i)) orelse usage();
        } else if (std.mem.eql(u8, arg, "--tq-plus")) {
            config.tq_plus = true;
        } else if (std.mem.eql(u8, arg, "--threads")) {
            config.threads = try std.fmt.parseInt(usize, nextArg(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--recall-curve")) {
            config.recall_curve = true;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            config.profile = true;
        } else if (std.mem.eql(u8, arg, "--query-file")) {
            config.query_path = nextArg(argv, &i);
        } else if (std.mem.eql(u8, arg, "--n")) {
            config.n = try std.fmt.parseInt(usize, nextArg(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--dim")) {
            config.dim = try std.fmt.parseInt(usize, nextArg(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--queries")) {
            config.n_queries = try std.fmt.parseInt(usize, nextArg(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--max-base")) {
            config.max_base = try std.fmt.parseInt(usize, nextArg(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--ef")) {
            config.ef_construction = try std.fmt.parseInt(usize, nextArg(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            config.seed = try std.fmt.parseInt(u64, nextArg(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--m")) {
            var widths: std.ArrayList(usize) = .empty;
            var fields = std.mem.tokenizeScalar(u8, nextArg(argv, &i), ',');
            while (fields.next()) |field| {
                try widths.append(allocator, try std.fmt.parseInt(usize, field, 10));
            }
            if (widths.items.len == 0) usage();
            config.stage1_widths = try widths.toOwnedSlice(allocator);
        } else {
            usage();
        }
    }
    return config;
}

fn nextArg(argv: []const []const u8, i: *usize) []const u8 {
    i.* += 1;
    if (i.* >= argv.len) usage();
    return argv[i.*];
}

fn usage() noreturn {
    std.debug.print(
        \\usage:
        \\  qj-bench synthetic [--n 10000] [--dim 128] [--queries 1000]
        \\  qj-bench fvecs <base.fvecs> [--query-file q.fvecs] [--max-base 100000] [--queries 1000]
        \\  qj-bench glove <vectors.txt> [--max-base 100000] [--queries 1000]
        \\
        \\common flags: --ef <n> (default 200), --seed <n>, --no-normalize,
        \\              --m <w1,w2,...> stage-1 beam widths (default 16,32,64,128),
        \\              --rerank <sq8|fp32|none> (default sq8), --recall-curve
        \\
    , .{});
    std.process.exit(1);
}

fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("error: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}
