//! qj-route: interactive query console over a prebuilt .tq index.
//!
//!   qj-route build --glove <vectors.txt> --out <index.tq>
//!       [--max-vectors N] [--ef 200] [--seed 42]
//!   qj-route --index <index.tq> [--dimensions N] [--m 128] [--symmetric]
//!
//! Queries are embedded with the index's own vocabulary: each known token
//! contributes its decoded payload vector (rotated space), the mean is
//! normalized, and the two-stage search runs on it. Per-query stage timings
//! and traversal counters are printed alongside the top matches.

const std = @import("std");
const qj = @import("quantajump");

const supported_dims = [_]usize{ 25, 50, 64, 100, 128, 200, 300, 768, 1536 };
const max_edges = 16;
const top_k = 10;

const BuildConfig = struct {
    glove_path: []const u8,
    out_path: []const u8,
    max_vectors: usize = std.math.maxInt(usize),
    ef_construction: usize = 200,
    seed: u64 = 42,
};

const QueryConfig = struct {
    index_path: []const u8,
    dimensions: ?usize = null,
    m: usize = 128,
    symmetric: bool = false,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args: std.ArrayList([]const u8) = .empty;
    defer args.deinit(allocator);
    var it = init.minimal.args.iterate();
    while (it.next()) |arg| try args.append(allocator, arg);
    if (args.items.len < 2) usage();

    if (std.mem.eql(u8, args.items[1], "build")) {
        const config = parseBuildArgs(args.items);
        return runBuildDispatch(allocator, io, config);
    }
    const config = parseQueryArgs(args.items);
    return runQueryDispatch(allocator, io, config);
}

// --- build mode ---

fn runBuildDispatch(allocator: std.mem.Allocator, io: std.Io, config: BuildConfig) !void {
    const data = try loadGlove(allocator, io, config.glove_path, config.max_vectors);
    std.debug.print("[qj-route] parsed {s} vectors of dim {d} from {s}\n", .{
        fmtComma(data.count), data.dim, config.glove_path,
    });
    normalizeAll(data.vectors, data.dim);

    inline for (supported_dims) |dim| {
        if (dim == data.dim) return runBuild(dim, allocator, io, config, data);
    }
    fatal("unsupported dimension {d}; compiled for: {any}", .{ data.dim, supported_dims });
}

fn runBuild(comptime dim: usize, allocator: std.mem.Allocator, io: std.Io, config: BuildConfig, data: GloveData) !void {
    const Idx = qj.Index(dim, max_edges);
    var index = try Idx.init(allocator, config.ef_construction, config.seed);
    defer index.deinit(allocator);

    var timer = Stopwatch.begin(io);
    for (0..data.count) |i| {
        try index.add(allocator, i, data.vectors[i * dim ..][0..dim]);
        if ((i + 1) % 50_000 == 0) {
            std.debug.print("[qj-route] indexed {s} / {s}\n", .{ fmtComma(i + 1), fmtComma(data.count) });
        }
    }
    std.debug.print("[qj-route] built index in {d:.1}s\n", .{nsToS(timer.read())});

    timer.reset();
    try qj.storage.save(Idx, allocator, io, config.out_path, &index, data.labels);
    std.debug.print("[qj-route] wrote {s} in {d:.1}s\n", .{ config.out_path, nsToS(timer.read()) });
}

// --- query mode ---

fn runQueryDispatch(allocator: std.mem.Allocator, io: std.Io, config: QueryConfig) !void {
    const header = qj.storage.readHeader(io, config.index_path) catch |err| {
        fatal("cannot read {s}: {s}", .{ config.index_path, @errorName(err) });
    };
    if (config.dimensions) |d| {
        if (d != header.dim) fatal("--dimensions {d} does not match index dim {d}", .{ d, header.dim });
    }
    if (header.max_edges != max_edges) {
        fatal("index built with max_edges {d}; this binary supports {d}", .{ header.max_edges, max_edges });
    }
    inline for (supported_dims) |dim| {
        if (dim == header.dim) return runQuery(dim, allocator, io, config);
    }
    fatal("unsupported dimension {d}; compiled for: {any}", .{ header.dim, supported_dims });
}

fn runQuery(comptime dim: usize, allocator: std.mem.Allocator, io: std.Io, config: QueryConfig) !void {
    const Idx = qj.Index(dim, max_edges);

    var timer = Stopwatch.begin(io);
    var loaded = try qj.storage.load(Idx, allocator, io, config.index_path);
    defer loaded.deinit(allocator);
    const index = &loaded.index;
    const load_s = nsToS(timer.read());

    var vocabulary: std.StringHashMapUnmanaged(u32) = .empty;
    defer vocabulary.deinit(allocator);
    try vocabulary.ensureTotalCapacity(allocator, @intCast(loaded.labels.len));
    for (loaded.labels, 0..) |label, i| {
        vocabulary.putAssumeCapacity(label, @intCast(i));
    }

    const payload_bytes = index.len() * @sizeOf(Idx.Payload);
    var graph_bytes: usize = 0;
    for (index.routing.layers.items) |*layer| {
        graph_bytes += layer.nodes.items.len * @sizeOf(Idx.Graph.Node);
        graph_bytes += layer.bit_vectors.items.len * @sizeOf(Idx.Graph.BitVec);
    }
    std.debug.print(
        \\[qj-route] loaded {s} vectors (dim {d}) in {d:.1}s from {s}
        \\[qj-route] resident index: {d:.1} MiB ({d:.1} MiB 3-bit payloads + {d:.1} MiB 1-bit graph mesh)
        \\[qj-route] stage-1 beam width m={d}, scoring: {s}; type a query, or "exit"
        \\
        \\
    , .{
        fmtComma(index.len()),         dim,
        load_s,                        config.index_path,
        mib(payload_bytes + graph_bytes + loaded.label_blob.len), mib(payload_bytes),
        mib(graph_bytes),              config.m,
        if (config.symmetric) "symmetric 3-bit" else "asymmetric (fp32 query)",
    });

    var ctx = try Idx.SearchContext.init(allocator, index, config.m);
    defer ctx.deinit(allocator);
    ctx.symmetric = config.symmetric;

    // Preallocated query-embedding buffers: the whole prompt loop runs
    // without touching the allocator.
    const query_vec = try allocator.alloc(f32, dim);
    defer allocator.free(query_vec);
    const decode_buf = try allocator.alloc(f32, dim);
    defer allocator.free(decode_buf);

    var stdin_buf: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);

    while (true) {
        std.debug.print("qj-query> ", .{});
        // takeDelimiter consumes the newline; its Exclusive sibling does
        // not, which would loop on the leftover delimiter forever.
        const line = (try stdin_reader.interface.takeDelimiter('\n')) orelse break;
        const trimmed = std.mem.trim(u8, line, " \t\r\"");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) break;

        runOneQuery(Idx, io, index, &vocabulary, loaded.labels, &ctx, trimmed, query_vec, decode_buf);
    }
    std.debug.print("\n", .{});
}

fn runOneQuery(
    comptime Idx: type,
    io: std.Io,
    index: *const Idx,
    vocabulary: *const std.StringHashMapUnmanaged(u32),
    labels: []const []const u8,
    ctx: *Idx.SearchContext,
    raw_query: []const u8,
    query_vec: []f32,
    decode_buf: []f32,
) void {
    var timer = Stopwatch.begin(io);

    // Embed: mean of the decoded payload vectors of the known tokens.
    // GloVe vocabularies are frequency-sorted, so the vector index doubles
    // as a frequency rank: when any contentful (rare) token is present,
    // tokens inside the stop-word band would drown it out and are skipped.
    const stop_word_rank = 200;
    var token_buf: [64]u8 = undefined;
    var token_ids: [64]u32 = undefined;
    var tokens_total: usize = 0;
    var tokens_known: usize = 0;
    var has_contentful = false;
    var tokenizer = std.mem.tokenizeAny(u8, raw_query, " \t,.;:!?'\"()-");
    while (tokenizer.next()) |token| {
        if (token.len > token_buf.len or tokens_known == token_ids.len) continue;
        tokens_total += 1;
        const lowered = std.ascii.lowerString(&token_buf, token);
        const idx = vocabulary.get(lowered) orelse continue;
        token_ids[tokens_known] = idx;
        tokens_known += 1;
        if (idx >= stop_word_rank) has_contentful = true;
    }

    @memset(query_vec, 0);
    var tokens_found: usize = 0;
    for (token_ids[0..tokens_known]) |idx| {
        if (has_contentful and idx < stop_word_rank) continue;
        tokens_found += 1;
        index.payloads.items[idx].decode(decode_buf);
        for (query_vec, decode_buf) |*acc, v| acc.* += v;
    }
    if (tokens_found == 0) {
        std.debug.print("no query token found in the index vocabulary ({d} tried)\n\n", .{tokens_total});
        return;
    }
    var norm_sq: f32 = 0;
    for (query_vec) |v| norm_sq += v * v;
    const inv_norm = 1.0 / @sqrt(@max(norm_sq, 1e-20));
    for (query_vec) |*v| v.* *= inv_norm;
    const embed_ns = timer.read();

    timer.reset();
    // Over-fetch so the query's own tokens can be dropped from the display
    // while still showing top_k neighbors.
    var out: [top_k * 2]qj.SearchResult = undefined;
    const fetched = index.searchRotated(ctx, query_vec, &out);
    const search_ns = timer.read();
    const stats = ctx.scratch.stats;

    var shown: [top_k]qj.SearchResult = undefined;
    var count: usize = 0;
    outer: for (out[0..fetched]) |result| {
        if (count == top_k) break;
        for (token_ids[0..tokens_known]) |idx| {
            if (result.id == idx) continue :outer;
        }
        shown[count] = result;
        count += 1;
    }

    std.debug.print(
        "[embed]        {d}/{d} tokens in vocabulary -> query vector in {d:.1}\u{00b5}s\n",
        .{ tokens_found, tokens_total, nsToUs(embed_ns) },
    );
    std.debug.print(
        "[1-bit graph]  traversed {d} layers, evaluated {s} nodes -> {d} candidates\n",
        .{ stats.layers_traversed, fmtComma(stats.nodes_evaluated), ctx.m },
    );
    std.debug.print(
        "[3-bit rerank] rescored {d} candidates via LUT kernel -> top {d}\n",
        .{ @min(ctx.m, index.len()), count },
    );

    if (count > 0) {
        std.debug.print("\ntop match: id {d} (score {d:.4}) - \"{s}\"\n", .{
            shown[0].id, shown[0].score, labels[@intCast(shown[0].id)],
        });
        for (shown[1..count], 2..) |result, rank| {
            std.debug.print("   {d:>2}. id {d} (score {d:.4}) - \"{s}\"\n", .{
                rank, result.id, result.score, labels[@intCast(result.id)],
            });
        }
    }
    std.debug.print(
        "\ntotal latency: {d:.1}\u{00b5}s (embed {d:.1} + search {d:.1}) | allocated heap: 0 bytes\n\n",
        .{ nsToUs(embed_ns + search_ns), nsToUs(embed_ns), nsToUs(search_ns) },
    );
}

// --- glove loading (build mode only) ---

const GloveData = struct {
    vectors: []f32,
    labels: []const []const u8,
    count: usize,
    dim: usize,
};

fn loadGlove(allocator: std.mem.Allocator, io: std.Io, path: []const u8, max_vectors: usize) !GloveData {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| {
        fatal("cannot read {s}: {s}", .{ path, @errorName(err) });
    };
    // Labels keep pointing into this buffer; it stays alive for the process.

    var lines = std.mem.tokenizeScalar(u8, bytes, '\n');
    const first = lines.next() orelse fatal("{s}: empty file", .{path});
    var dim: usize = 0;
    var probe = std.mem.tokenizeScalar(u8, first, ' ');
    _ = probe.next();
    while (probe.next()) |_| dim += 1;
    if (dim == 0) fatal("{s}: no coordinates on first line", .{path});

    var vectors: std.ArrayList(f32) = .empty;
    var labels: std.ArrayList([]const u8) = .empty;
    lines.reset();
    var count: usize = 0;
    while (lines.next()) |line| {
        if (count == max_vectors) break;
        var fields = std.mem.tokenizeScalar(u8, line, ' ');
        const token = fields.next() orelse continue;
        var parsed: usize = 0;
        while (fields.next()) |field| : (parsed += 1) {
            try vectors.append(allocator, std.fmt.parseFloat(f32, field) catch {
                fatal("{s}: bad float '{s}' at line {d}", .{ path, field, count + 1 });
            });
        }
        if (parsed != dim) fatal("{s}: line {d} has {d} coords, expected {d}", .{ path, count + 1, parsed, dim });
        try labels.append(allocator, token);
        count += 1;
    }
    return .{
        .vectors = try vectors.toOwnedSlice(allocator),
        .labels = try labels.toOwnedSlice(allocator),
        .count = count,
        .dim = dim,
    };
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

// --- plumbing ---

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

fn nsToUs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e3;
}

fn nsToS(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn mib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 20);
}

/// Formats with thousands separators into a static buffer (single-threaded
/// CLI; at most a few uses per print statement would alias, so callers keep
/// one call per format argument list... rotated through 4 slots to be safe).
fn fmtComma(value: usize) []const u8 {
    const S = struct {
        var bufs: [4][32]u8 = undefined;
        var next: usize = 0;
    };
    const buf = &S.bufs[S.next % S.bufs.len];
    S.next += 1;

    var digits_buf: [24]u8 = undefined;
    const digits = std.fmt.bufPrint(&digits_buf, "{d}", .{value}) catch unreachable;
    var out_len: usize = 0;
    for (digits, 0..) |c, i| {
        if (i > 0 and (digits.len - i) % 3 == 0) {
            buf[out_len] = ',';
            out_len += 1;
        }
        buf[out_len] = c;
        out_len += 1;
    }
    return buf[0..out_len];
}

fn parseBuildArgs(argv: []const []const u8) BuildConfig {
    var glove_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var config = BuildConfig{ .glove_path = undefined, .out_path = undefined };
    var i: usize = 2;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--glove")) {
            glove_path = nextArg(argv, &i);
        } else if (std.mem.eql(u8, arg, "--out")) {
            out_path = nextArg(argv, &i);
        } else if (std.mem.eql(u8, arg, "--max-vectors")) {
            config.max_vectors = parseInt(usize, nextArg(argv, &i));
        } else if (std.mem.eql(u8, arg, "--ef")) {
            config.ef_construction = parseInt(usize, nextArg(argv, &i));
        } else if (std.mem.eql(u8, arg, "--seed")) {
            config.seed = parseInt(u64, nextArg(argv, &i));
        } else {
            usage();
        }
    }
    config.glove_path = glove_path orelse usage();
    config.out_path = out_path orelse usage();
    return config;
}

fn parseQueryArgs(argv: []const []const u8) QueryConfig {
    var index_path: ?[]const u8 = null;
    var config = QueryConfig{ .index_path = undefined };
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--index")) {
            index_path = nextArg(argv, &i);
        } else if (std.mem.eql(u8, arg, "--dimensions")) {
            config.dimensions = parseInt(usize, nextArg(argv, &i));
        } else if (std.mem.eql(u8, arg, "--m")) {
            config.m = parseInt(usize, nextArg(argv, &i));
        } else if (std.mem.eql(u8, arg, "--symmetric")) {
            config.symmetric = true;
        } else {
            usage();
        }
    }
    config.index_path = index_path orelse usage();
    return config;
}

fn parseInt(comptime T: type, text: []const u8) T {
    return std.fmt.parseInt(T, text, 10) catch usage();
}

fn nextArg(argv: []const []const u8, i: *usize) []const u8 {
    i.* += 1;
    if (i.* >= argv.len) usage();
    return argv[i.*];
}

fn usage() noreturn {
    std.debug.print(
        \\usage:
        \\  qj-route build --glove <vectors.txt> --out <index.tq>
        \\      [--max-vectors N] [--ef 200] [--seed 42]
        \\  qj-route --index <index.tq> [--dimensions N] [--m 128] [--symmetric]
        \\
    , .{});
    std.process.exit(1);
}

fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("error: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}
