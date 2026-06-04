//! Isolated routing-quality experiment / dimension-crossover sweep: how does
//! increasing the routing-code bit count B (decoupled from the data dimension
//! via random projection) affect recall, and where does the B=dim default
//! start failing as the data dimension drops?
//!
//! Measures *routing recall* only — the fraction of the exact top-10 that
//! land in the m-candidate set the graph returns, BEFORE any rerank. That
//! isolates the routing stage from quantization/rerank.
//!
//!   zig build routing-exp -Doptimize=ReleaseFast -- <dim> <trn.fvecs> <tst.fvecs> <gt.ivecs>

const std = @import("std");
const qj = @import("quantajump");

const max_edges = 32;
const ef_construction = 200;
const m_values = [_]usize{ 128, 512 };
const top_k = 10;
// GloVe family — angular, same source, dimension is the only variable.
const supported_dims = [_]usize{ 25, 50, 100, 200 };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args: std.ArrayList([]const u8) = .empty;
    defer args.deinit(allocator);
    var it = init.minimal.args.iterate();
    while (it.next()) |a| try args.append(allocator, a);
    if (args.items.len < 5) {
        std.debug.print("usage: routing-exp <dim> <trn.fvecs> <tst.fvecs> <gt.ivecs>\n", .{});
        std.process.exit(1);
    }
    const dim = try std.fmt.parseInt(usize, args.items[1], 10);

    inline for (supported_dims) |data_dim| {
        if (data_dim == dim) return runForDim(data_dim, allocator, io, args.items);
    }
    std.debug.print("unsupported dim {d}; compiled for {any}\n", .{ dim, supported_dims });
    std.process.exit(1);
}

fn runForDim(comptime data_dim: usize, allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    const train = try loadFvecs(allocator, io, args[2], data_dim);
    defer allocator.free(train);
    const queries = try loadFvecs(allocator, io, args[3], data_dim);
    defer allocator.free(queries);
    const gt = try loadIvecs(allocator, io, args[4], top_k);
    defer allocator.free(gt);

    const n = train.len / data_dim;
    const nq = queries.len / data_dim;
    std.debug.print("\nd={d}: train {d}, queries {d}, routing recall@{d} (candidates before rerank)\n", .{ data_dim, n, nq, top_k });
    std.debug.print("{s:>6}", .{"bits"});
    for (m_values) |m| std.debug.print("   rr@10 m={d:<4}", .{m});
    std.debug.print("   build_s\n", .{});

    // B=dim is the (proxy for the) current default; larger B is the fix.
    const bit_counts = [_]usize{ data_dim, 256, 512, 1024 };
    inline for (bit_counts) |bits| {
        try runConfig(data_dim, bits, allocator, io, train, queries, gt, n, nq);
    }
}

fn runConfig(
    comptime data_dim: usize,
    comptime bits: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    train: []const f32,
    queries: []const f32,
    gt: []const i32,
    n: usize,
    nq: usize,
) !void {
    const Graph = qj.RoutingGraph(bits, max_edges);
    const BitVec = qj.BitVector(bits);

    // B x data_dim i.i.d. Gaussian projection (SimHash). Fixed seed so the
    // only variable across configs is B.
    const proj = try allocator.alloc(f32, bits * data_dim);
    defer allocator.free(proj);
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();
    for (proj) |*v| v.* = rand.floatNorm(f32);

    const projected = try allocator.alloc(f32, bits);
    defer allocator.free(projected);

    var graph = Graph.init(ef_construction);
    defer graph.deinit(allocator);

    var timer = try Stopwatch.begin(io);
    for (0..n) |i| {
        const code = projectCode(data_dim, BitVec, bits, proj, train[i * data_dim ..][0..data_dim], projected);
        try graph.insert(allocator, i, code);
    }
    const build_s = timer.readSeconds(io);

    var scratch = qj.graph.TraversalScratch{};
    defer scratch.deinit(allocator);
    try scratch.ensureCapacity(allocator, n, m_values[m_values.len - 1]);

    std.debug.print("{d:>6}", .{bits});
    for (m_values) |m| {
        var hits: usize = 0;
        for (0..nq) |q| {
            const code = projectCode(data_dim, BitVec, bits, proj, queries[q * data_dim ..][0..data_dim], projected);
            const candidates = graph.route(&code, m, &scratch);
            for (gt[q * top_k ..][0..top_k]) |true_id| {
                for (candidates) |c| {
                    if (c.id == @as(u64, @intCast(true_id))) {
                        hits += 1;
                        break;
                    }
                }
            }
        }
        const rr = @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(nq * top_k));
        std.debug.print("   {d:>10.4}", .{rr});
    }
    std.debug.print("   {d:>7.1}\n", .{build_s});
}

fn projectCode(comptime data_dim: usize, comptime BitVec: type, comptime bits: usize, proj: []const f32, x: []const f32, scratch: []f32) BitVec {
    for (0..bits) |b| {
        scratch[b] = qj.rotation.dot(proj[b * data_dim ..][0..data_dim], x);
    }
    return BitVec.fromF32(scratch);
}

const Stopwatch = struct {
    started: std.Io.Timestamp,
    fn begin(io: std.Io) !Stopwatch {
        return .{ .started = std.Io.Timestamp.now(io, .awake) };
    }
    fn readSeconds(self: *const Stopwatch, io: std.Io) f64 {
        const now = std.Io.Timestamp.now(io, .awake);
        return @as(f64, @floatFromInt(@as(i128, self.started.durationTo(now).nanoseconds))) / 1e9;
    }
};

fn loadFvecs(allocator: std.mem.Allocator, io: std.Io, path: []const u8, dim: usize) ![]f32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    const record = 4 + dim * 4;
    const count = bytes.len / record;
    const out = try allocator.alloc(f32, count * dim);
    for (0..count) |i| {
        const rec = bytes[i * record ..][0..record];
        for (0..dim) |j| out[i * dim + j] = @bitCast(std.mem.readInt(u32, rec[4 + j * 4 ..][0..4], .little));
    }
    return out;
}

fn loadIvecs(allocator: std.mem.Allocator, io: std.Io, path: []const u8, dim: usize) ![]i32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    const record = 4 + dim * 4;
    const count = bytes.len / record;
    const out = try allocator.alloc(i32, count * dim);
    for (0..count) |i| {
        const rec = bytes[i * record ..][0..record];
        for (0..dim) |j| out[i * dim + j] = @bitCast(std.mem.readInt(u32, rec[4 + j * 4 ..][0..4], .little));
    }
    return out;
}
