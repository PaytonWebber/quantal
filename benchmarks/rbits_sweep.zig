//! routing_bits sweep on the full pipeline at d=1536 (DBpedia OpenAI3).
//!
//! Unlike routing_experiment.zig (routing recall only), this runs the whole
//! Index — rotate, route, 3-bit LUT, sq8 exact rerank — so it reports final
//! recall AND latency/throughput as routing_bits varies above and below the
//! data dimension. Answers: at high dim, does shortening the routing code
//! (a JL reduction) buy throughput cheaply, and does lengthening it help?
//!
//!   zig build rbits-sweep -Doptimize=ReleaseFast -- <base.fvecs> <query.fvecs> [nqueries]

const std = @import("std");
const qj = @import("quantajump");

const dim = 1536;
const max_edges = 16;
const ef_construction = 200;
const top_k = 10;
const m_values = [_]usize{ 128, 512 };
const rb_values = [_]usize{ 256, 512, 1024, 1536, 2048, 3072 };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args: std.ArrayList([]const u8) = .empty;
    defer args.deinit(allocator);
    var it = init.minimal.args.iterate();
    while (it.next()) |a| try args.append(allocator, a);
    if (args.items.len < 3) {
        std.debug.print("usage: rbits-sweep <base.fvecs> <query.fvecs> [nqueries]\n", .{});
        std.process.exit(1);
    }
    const nq_cap = if (args.items.len > 3) try std.fmt.parseInt(usize, args.items[3], 10) else 1000;

    const base = try loadFvecs(allocator, io, args.items[1], std.math.maxInt(usize));
    defer allocator.free(base);
    const queries_all = try loadFvecs(allocator, io, args.items[2], nq_cap);
    defer allocator.free(queries_all);
    normalize(base);
    normalize(queries_all);
    const n = base.len / dim;
    const nq = queries_all.len / dim;

    std.debug.print("DBpedia d={d}: n={d}, queries={d}, max_edges={d}\n", .{ dim, n, nq, max_edges });

    // Exact FP32 top-10 ground truth (computed once).
    var timer = Stopwatch.begin(io);
    const gt = try allocator.alloc([top_k]u64, nq);
    defer allocator.free(gt);
    for (0..nq) |q| gt[q] = exactTopK(base, queries_all[q * dim ..][0..dim]);
    std.debug.print("ground truth: {d:.1}s\n\n", .{timer.readSeconds(io)});

    std.debug.print("{s:>6} {s:>9} {s:>8} {s:>4} {s:>10} {s:>10} {s:>10} {s:>10}\n", .{
        "rbits", "code_B", "build_s", "m", "recall1@1", "recall@10", "QPS(MT)", "us/q(MT)",
    });
    inline for (rb_values) |rb| {
        try runRb(rb, allocator, io, base, queries_all, gt, n, nq);
    }
}

fn runRb(
    comptime rb: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    base: []const f32,
    queries: []const f32,
    gt: []const [top_k]u64,
    n: usize,
    nq: usize,
) !void {
    const Idx = qj.Index(dim, max_edges, rb);
    var index = try Idx.init(allocator, ef_construction, 42);
    defer index.deinit(allocator);

    const ids = try allocator.alloc(u64, n);
    defer allocator.free(ids);
    for (ids, 0..) |*id, i| id.* = i;

    var timer = Stopwatch.begin(io);
    try index.addBatch(allocator, ids, base[0 .. n * dim], 12);
    const build_s = timer.readSeconds(io);
    const code_bytes = Idx.Graph.BitVec.words_count * 8;

    const results = try allocator.alloc(qj.SearchResult, nq * top_k);
    defer allocator.free(results);
    const counts = try allocator.alloc(usize, nq);
    defer allocator.free(counts);

    for (m_values) |m| {
        timer = Stopwatch.begin(io);
        try index.searchBatch(allocator, queries[0 .. nq * dim], top_k, m, 12, true, results, counts);
        const elapsed = timer.readSeconds(io);

        var hits1: usize = 0;
        var overlap: usize = 0;
        for (0..nq) |q| {
            const want = gt[q][0];
            if (counts[q] > 0 and results[q * top_k].id == want) hits1 += 1;
            for (results[q * top_k ..][0..counts[q]]) |r| {
                if (std.mem.indexOfScalar(u64, &gt[q], r.id) != null) overlap += 1;
            }
        }
        const nqf: f64 = @floatFromInt(nq);
        std.debug.print("{d:>6} {d:>9} {d:>8.1} {d:>4} {d:>10.4} {d:>10.4} {d:>10.0} {d:>10.1}\n", .{
            rb,                                              code_bytes,
            build_s,                                         m,
            @as(f64, @floatFromInt(hits1)) / nqf,           @as(f64, @floatFromInt(overlap)) / (nqf * top_k),
            nqf / elapsed,                                   elapsed / nqf * 1e6,
        });
    }
}

fn exactTopK(base: []const f32, query: []const f32) [top_k]u64 {
    var ids: [top_k]u64 = @splat(std.math.maxInt(u64));
    var scores: [top_k]f32 = @splat(-std.math.inf(f32));
    const n = base.len / dim;
    for (0..n) |i| {
        const s = qj.rotation.dot(base[i * dim ..][0..dim], query);
        if (s <= scores[top_k - 1]) continue;
        var pos: usize = top_k - 1;
        while (pos > 0 and s > scores[pos - 1]) : (pos -= 1) {
            scores[pos] = scores[pos - 1];
            ids[pos] = ids[pos - 1];
        }
        scores[pos] = s;
        ids[pos] = i;
    }
    return ids;
}

fn normalize(data: []f32) void {
    var i: usize = 0;
    while (i < data.len) : (i += dim) {
        const row = data[i..][0..dim];
        var ns: f32 = 0;
        for (row) |x| ns += x * x;
        if (ns > 0) {
            const inv = 1.0 / @sqrt(ns);
            for (row) |*x| x.* *= inv;
        }
    }
}

const Stopwatch = struct {
    started: std.Io.Timestamp,
    fn begin(io: std.Io) Stopwatch {
        return .{ .started = std.Io.Timestamp.now(io, .awake) };
    }
    fn readSeconds(self: *const Stopwatch, io: std.Io) f64 {
        const now = std.Io.Timestamp.now(io, .awake);
        return @as(f64, @floatFromInt(@as(i128, self.started.durationTo(now).nanoseconds))) / 1e9;
    }
};

fn loadFvecs(allocator: std.mem.Allocator, io: std.Io, path: []const u8, max_vectors: usize) ![]f32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    const record = 4 + dim * 4;
    const count = @min(bytes.len / record, max_vectors);
    const out = try allocator.alloc(f32, count * dim);
    for (0..count) |i| {
        const rec = bytes[i * record ..][0..record];
        for (0..dim) |j| out[i * dim + j] = @bitCast(std.mem.readInt(u32, rec[4 + j * 4 ..][0..4], .little));
    }
    return out;
}
