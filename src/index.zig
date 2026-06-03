//! The two-stage cascading index.
//!
//! Ingest: rotate -> encode a 3-bit TurboQuant payload (flat array) and a
//! 1-bit sign vector (routing graph node).
//! Search: rotate the query, route through the 1-bit graph to M candidates
//! (Stage 1), rerank them with unbiased 3-bit inner products (Stage 2).
//! The search path takes a preallocated SearchContext and no allocator, so
//! queries are allocation-free by construction.

const std = @import("std");
const bitvec = @import("bitvec.zig");
const turboquant = @import("turboquant.zig");
const graph_mod = @import("graph.zig");
const heap_mod = @import("heap.zig");
const rotation_mod = @import("rotation.zig");

pub const SearchResult = heap_mod.SearchResult;

pub fn Index(comptime dim: usize, comptime max_edges: usize) type {
    return struct {
        const Self = @This();
        pub const Payload = turboquant.TurboQuantPayload(dim);
        pub const Graph = graph_mod.RoutingGraph(dim, max_edges);
        pub const Rotation = rotation_mod.RandomRotation(dim);
        pub const dimension = dim;

        rotation: Rotation,
        payloads: std.ArrayList(Payload) = .empty,
        routing: Graph,
        rotate_buf: []f32,

        pub fn init(allocator: std.mem.Allocator, ef_construction: usize, seed: u64) !Self {
            var rot = try Rotation.init(allocator, seed);
            errdefer rot.deinit(allocator);
            const rotate_buf = try allocator.alloc(f32, dim);
            return .{
                .rotation = rot,
                .routing = Graph.init(ef_construction),
                .rotate_buf = rotate_buf,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.rotation.deinit(allocator);
            self.payloads.deinit(allocator);
            self.routing.deinit(allocator);
            allocator.free(self.rotate_buf);
            self.* = undefined;
        }

        pub fn len(self: *const Self) usize {
            return self.payloads.items.len;
        }

        pub fn add(self: *Self, allocator: std.mem.Allocator, id: u64, coords: []const f32) !void {
            std.debug.assert(coords.len == dim);
            self.rotation.apply(coords, self.rotate_buf);

            const internal_id: u64 = self.payloads.items.len;
            try self.payloads.append(allocator, Payload.encode(id, self.rotate_buf));
            errdefer _ = self.payloads.pop();

            const bits = Graph.BitVec.fromF32(self.rotate_buf);
            try self.routing.insert(allocator, internal_id, bits);
        }

        /// Preallocated query state. Created once (or whenever the index has
        /// grown past its capacity) and reused across searches.
        pub const SearchContext = struct {
            scratch: graph_mod.TraversalScratch = .{},
            rotated_query: []f32,
            decoded_query: []f32,
            score_lut: []f32,
            m: usize,
            /// true (default): stage 2 scores against the query's own 3-bit
            /// roundtrip. false: scores against the full-precision rotated
            /// query, matching the paper's asymmetric estimator (Thm. 2) —
            /// more accurate, same storage.
            symmetric: bool = true,

            pub fn init(allocator: std.mem.Allocator, index: *const Self, m: usize) !SearchContext {
                std.debug.assert(m > 0);
                var ctx = SearchContext{
                    .rotated_query = try allocator.alloc(f32, dim),
                    .decoded_query = try allocator.alloc(f32, dim),
                    .score_lut = try allocator.alloc(f32, dim * 8),
                    .m = m,
                };
                errdefer allocator.free(ctx.rotated_query);
                errdefer allocator.free(ctx.decoded_query);
                errdefer allocator.free(ctx.score_lut);
                try ctx.scratch.ensureCapacity(allocator, @max(index.len(), 1), m);
                return ctx;
            }

            pub fn deinit(self: *SearchContext, allocator: std.mem.Allocator) void {
                self.scratch.deinit(allocator);
                allocator.free(self.rotated_query);
                allocator.free(self.decoded_query);
                allocator.free(self.score_lut);
                self.* = undefined;
            }
        };

        /// Fills `out` with up to out.len results sorted by descending
        /// inner-product score; returns the count. Allocation-free.
        pub fn search(self: *const Self, ctx: *SearchContext, query: []const f32, out: []SearchResult) usize {
            std.debug.assert(query.len == dim);
            self.rotation.apply(query, ctx.rotated_query);
            return self.searchRotated(ctx, ctx.rotated_query, out);
        }

        /// Same as `search`, for a query already in rotated space (e.g.
        /// assembled from decoded payloads, which live there).
        pub fn searchRotated(self: *const Self, ctx: *SearchContext, rotated: []const f32, out: []SearchResult) usize {
            std.debug.assert(rotated.len == dim);
            if (self.len() == 0 or out.len == 0) return 0;
            std.debug.assert(ctx.scratch.visited.bit_length >= self.len());

            const query_bits = Graph.BitVec.fromF32(rotated);
            if (ctx.symmetric) {
                turboquant.quantizeQuery(rotated, ctx.decoded_query);
            } else {
                @memcpy(ctx.decoded_query, rotated);
            }
            const query_sum = turboquant.buildScoreLut(ctx.decoded_query, ctx.score_lut);

            const candidates = self.routing.route(&query_bits, ctx.m, &ctx.scratch);

            var topk = heap_mod.TopK.fromBuffer(out);
            for (candidates) |candidate| {
                const payload = &self.payloads.items[@intCast(candidate.id)];
                topk.offer(.{
                    .id = payload.id,
                    .score = payload.scoreLut(ctx.score_lut, query_sum),
                });
            }
            return topk.sortDescending();
        }
    };
}

test "empty index returns no results" {
    const Idx = Index(32, 4);
    var index = try Idx.init(std.testing.allocator, 16, 1);
    defer index.deinit(std.testing.allocator);

    var ctx = try Idx.SearchContext.init(std.testing.allocator, &index, 8);
    defer ctx.deinit(std.testing.allocator);

    const query: [32]f32 = @splat(1.0);
    var out: [4]SearchResult = undefined;
    try std.testing.expectEqual(@as(usize, 0), index.search(&ctx, &query, &out));
}

test "clustered recall: queries land in their own cluster" {
    const dim = 32;
    const clusters = 10;
    const per_cluster = 50;
    const Idx = Index(dim, 8);

    const allocator = std.testing.allocator;
    var index = try Idx.init(allocator, 48, 42);
    defer index.deinit(allocator);

    var prng = std.Random.DefaultPrng.init(77);
    const rand = prng.random();

    var centers: [clusters][dim]f32 = undefined;
    for (&centers) |*center| {
        var norm_sq: f32 = 0;
        for (center) |*c| {
            c.* = rand.floatNorm(f32);
            norm_sq += c.* * c.*;
        }
        const inv_norm = 1.0 / @sqrt(norm_sq);
        for (center) |*c| c.* *= inv_norm;
    }

    for (centers, 0..) |center, cluster| {
        for (0..per_cluster) |j| {
            var point: [dim]f32 = undefined;
            for (&point, center) |*p, c| p.* = c + 0.05 * rand.floatNorm(f32);
            const id: u64 = cluster * 1000 + j;
            try index.add(allocator, id, &point);
        }
    }

    var ctx = try Idx.SearchContext.init(allocator, &index, 64);
    defer ctx.deinit(allocator);

    var correct: usize = 0;
    var total: usize = 0;
    for (centers, 0..) |center, cluster| {
        var out: [10]SearchResult = undefined;
        const count = index.search(&ctx, &center, &out);
        try std.testing.expectEqual(@as(usize, 10), count);
        for (out[0..count]) |result| {
            total += 1;
            if (result.id / 1000 == cluster) correct += 1;
        }
    }
    // Well-separated clusters with sigma=0.05 noise: expect near-perfect
    // cluster purity in the top-10.
    try std.testing.expect(correct * 10 >= total * 8);
}

test "results are sorted by descending score" {
    const dim = 32;
    const Idx = Index(dim, 4);
    const allocator = std.testing.allocator;

    var index = try Idx.init(allocator, 16, 5);
    defer index.deinit(allocator);

    var prng = std.Random.DefaultPrng.init(13);
    const rand = prng.random();
    for (0..100) |i| {
        var coords: [dim]f32 = undefined;
        for (&coords) |*c| c.* = rand.floatNorm(f32);
        try index.add(allocator, i, &coords);
    }

    var ctx = try Idx.SearchContext.init(allocator, &index, 32);
    defer ctx.deinit(allocator);

    var query: [dim]f32 = undefined;
    for (&query) |*c| c.* = rand.floatNorm(f32);
    var out: [8]SearchResult = undefined;
    const count = index.search(&ctx, &query, &out);
    try std.testing.expectEqual(@as(usize, 8), count);
    for (out[0 .. count - 1], out[1..count]) |a, b| {
        try std.testing.expect(a.score >= b.score);
    }
}
