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

/// What the stage-3 exact rerank reads.
/// - fp32: rotated originals verbatim (4 bytes/coord)
/// - sq8: per-vector max-abs scaled int8 (1 byte/coord, ~equal recall)
/// - none: stage 3 disabled, stage-2 quantized scores are final
pub const RerankStore = enum(u8) { none = 0, fp32 = 1, sq8 = 2 };

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
        /// Stage-3 rerank source; set before the first add. sq8 costs one
        /// byte per coordinate and reranks within float rounding of fp32.
        rerank_store: RerankStore = .sq8,
        /// TQ+ per-coordinate calibration: fit a shift/scale per coordinate
        /// from the first `calibration_sample` adds (which are buffered),
        /// then freeze. Standardized coordinates feed only the 3-bit
        /// quantizer (decoded through the inverse affine at scoring time);
        /// routing bits and stage 3 stay in raw rotated space. Set before
        /// the first add; call `freeze` after ingest (or it triggers
        /// automatically at the sample threshold). Measured: no recall gain
        /// over the per-vector calibration on tested datasets — off by
        /// default, see benchmarks/RESULTS.md.
        tq_plus: bool = false,
        calibration_sample: usize = 1000,
        calib_shift: []f32 = &.{},
        calib_scale: []f32 = &.{},
        calib_frozen: bool = false,
        pending_coords: std.ArrayList(f32) = .empty,
        pending_ids: std.ArrayList(u64) = .empty,
        calib_buf: []f32,
        /// Rotated FP32 originals (rerank_store == .fp32).
        originals: std.ArrayList(f32) = .empty,
        /// int8 codes + one max-abs scale per vector (rerank_store == .sq8).
        sq8_codes: std.ArrayList(i8) = .empty,
        sq8_scales: std.ArrayList(f32) = .empty,

        pub fn init(allocator: std.mem.Allocator, ef_construction: usize, seed: u64) !Self {
            var rot = try Rotation.init(allocator, seed);
            errdefer rot.deinit(allocator);
            const rotate_buf = try allocator.alloc(f32, dim);
            errdefer allocator.free(rotate_buf);
            const calib_buf = try allocator.alloc(f32, dim);
            return .{
                .rotation = rot,
                .routing = Graph.init(ef_construction),
                .rotate_buf = rotate_buf,
                .calib_buf = calib_buf,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.rotation.deinit(allocator);
            self.payloads.deinit(allocator);
            self.routing.deinit(allocator);
            self.originals.deinit(allocator);
            self.sq8_codes.deinit(allocator);
            self.sq8_scales.deinit(allocator);
            allocator.free(self.calib_shift);
            allocator.free(self.calib_scale);
            self.pending_coords.deinit(allocator);
            self.pending_ids.deinit(allocator);
            allocator.free(self.calib_buf);
            allocator.free(self.rotate_buf);
            self.* = undefined;
        }

        pub fn len(self: *const Self) usize {
            return self.payloads.items.len;
        }

        pub fn add(self: *Self, allocator: std.mem.Allocator, id: u64, coords: []const f32) !void {
            std.debug.assert(coords.len == dim);
            self.rotation.apply(coords, self.rotate_buf);

            if (self.tq_plus and !self.calib_frozen) {
                try self.pending_coords.appendSlice(allocator, self.rotate_buf);
                errdefer self.pending_coords.shrinkRetainingCapacity(self.pending_coords.items.len - dim);
                try self.pending_ids.append(allocator, id);
                if (self.pending_ids.items.len >= self.calibration_sample) {
                    try self.freeze(allocator);
                }
                return;
            }
            try self.insertRotated(allocator, id, self.rotate_buf);
        }

        /// Fits the TQ+ per-coordinate calibration from the buffered sample
        /// and indexes the buffered vectors. Must be called once after
        /// ingest when `tq_plus` is on and fewer than `calibration_sample`
        /// vectors were added; no-op otherwise.
        pub fn freeze(self: *Self, allocator: std.mem.Allocator) !void {
            if (!self.tq_plus or self.calib_frozen) return;
            const n = self.pending_ids.items.len;

            const shift = try allocator.alloc(f32, dim);
            errdefer allocator.free(shift);
            const scale = try allocator.alloc(f32, dim);
            errdefer allocator.free(scale);
            if (n == 0) {
                @memset(shift, 0);
                @memset(scale, 1);
            } else {
                const n_f: f64 = @floatFromInt(n);
                for (shift, scale, 0..) |*sh, *sc, j| {
                    var sum: f64 = 0;
                    var sum_sq: f64 = 0;
                    var i: usize = 0;
                    while (i < n) : (i += 1) {
                        const v = self.pending_coords.items[i * dim + j];
                        sum += v;
                        sum_sq += @as(f64, v) * v;
                    }
                    const mean = sum / n_f;
                    const variance = @max(0.0, sum_sq / n_f - mean * mean);
                    sh.* = @floatCast(mean);
                    sc.* = @floatCast(@max(@sqrt(variance), 1e-6));
                }
            }
            self.calib_shift = shift;
            self.calib_scale = scale;
            self.calib_frozen = true;

            for (0..n) |i| {
                try self.insertRotated(
                    allocator,
                    self.pending_ids.items[i],
                    self.pending_coords.items[i * dim ..][0..dim],
                );
            }
            self.pending_coords.clearAndFree(allocator);
            self.pending_ids.clearAndFree(allocator);
        }

        fn insertRotated(self: *Self, allocator: std.mem.Allocator, id: u64, rotated: []const f32) !void {
            // Only the 3-bit payload quantizes standardized coordinates
            // (decoded back through the inverse affine at scoring time).
            // Routing bits stay raw: centering signs strips the shared mean
            // component that raw-cosine ranking heavily weights, which
            // measurably hurts Hamming routing on embedding data.
            const internal_id: u64 = self.payloads.items.len;
            try self.payloads.append(allocator, Payload.encodeWithCalibration(
                id,
                rotated,
                self.calib_shift,
                self.calib_scale,
            ));
            errdefer _ = self.payloads.pop();

            switch (self.rerank_store) {
                .none => {},
                .fp32 => {
                    std.debug.assert(self.originals.items.len == internal_id * dim);
                    try self.originals.appendSlice(allocator, rotated);
                },
                .sq8 => {
                    std.debug.assert(self.sq8_codes.items.len == internal_id * dim);
                    var max_abs: f32 = 0;
                    for (rotated) |v| max_abs = @max(max_abs, @abs(v));
                    const scale = @max(max_abs, 1e-20) / 127.0;
                    try self.sq8_scales.append(allocator, scale);
                    try self.sq8_codes.ensureUnusedCapacity(allocator, dim);
                    for (rotated) |v| {
                        self.sq8_codes.appendAssumeCapacity(@intFromFloat(@round(v / scale)));
                    }
                },
            }
            errdefer switch (self.rerank_store) {
                .none => {},
                .fp32 => self.originals.shrinkRetainingCapacity(internal_id * dim),
                .sq8 => {
                    self.sq8_codes.shrinkRetainingCapacity(internal_id * dim);
                    self.sq8_scales.shrinkRetainingCapacity(internal_id);
                },
            };

            const bits = Graph.BitVec.fromF32(rotated);
            try self.routing.insert(allocator, internal_id, bits);
        }

        /// True when every vector has a stored rerank record, enabling the
        /// stage-3 exact rescore.
        pub fn hasRerankStore(self: *const Self) bool {
            return switch (self.rerank_store) {
                .none => false,
                .fp32 => self.originals.items.len == self.payloads.items.len * dim,
                .sq8 => self.sq8_codes.items.len == self.payloads.items.len * dim,
            };
        }

        /// Stage-3 exact score of one stored vector against a rotated query.
        pub fn rerankScore(self: *const Self, internal: usize, rotated_query: []const f32) f32 {
            return switch (self.rerank_store) {
                .none => unreachable,
                .fp32 => rotation_mod.dot(rotated_query, self.originals.items[internal * dim ..][0..dim]),
                .sq8 => dotSq8(rotated_query, self.sq8_codes.items[internal * dim ..][0..dim]) *
                    self.sq8_scales.items[internal],
            };
        }

        /// Preallocated query state. Created once (or whenever the index has
        /// grown past its capacity) and reused across searches.
        pub const SearchContext = struct {
            scratch: graph_mod.TraversalScratch = .{},
            rotated_query: []f32,
            decoded_query: []f32,
            score_lut: []f32,
            /// Stage-2 output / stage-3 input: the top rerank_factor * k
            /// candidates by quantized score (ids are internal indices).
            rerank_pool: []SearchResult,
            m: usize,
            /// Stage-3 pool size as a multiple of k. 1 disables the benefit
            /// (pool == k); 4 is a good default.
            rerank_factor: usize = 4,
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
                    .rerank_pool = try allocator.alloc(SearchResult, m),
                    .m = m,
                };
                errdefer allocator.free(ctx.rotated_query);
                errdefer allocator.free(ctx.decoded_query);
                errdefer allocator.free(ctx.score_lut);
                errdefer allocator.free(ctx.rerank_pool);
                try ctx.scratch.ensureCapacity(allocator, @max(index.len(), 1), m);
                return ctx;
            }

            pub fn deinit(self: *SearchContext, allocator: std.mem.Allocator) void {
                self.scratch.deinit(allocator);
                allocator.free(self.rotated_query);
                allocator.free(self.decoded_query);
                allocator.free(self.score_lut);
                allocator.free(self.rerank_pool);
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
        ///
        /// Stage 1 routes the 1-bit graph to m candidates; stage 2 ranks
        /// them by quantized LUT score; when FP32 originals are stored,
        /// stage 3 exactly rescores the top rerank_factor * k of those, so
        /// quantization noise can no longer reorder the final results.
        pub fn searchRotated(self: *const Self, ctx: *SearchContext, rotated: []const f32, out: []SearchResult) usize {
            std.debug.assert(rotated.len == dim);
            std.debug.assert(!self.tq_plus or self.calib_frozen); // freeze() after ingest
            if (self.len() == 0 or out.len == 0) return 0;
            std.debug.assert(ctx.scratch.visited.bit_length >= self.len());

            // Routing bits are raw-space on both sides (see insertRotated);
            // stage-2 LUT scoring and stage 3 also estimate inner products
            // in the raw rotated space.
            const query_bits = Graph.BitVec.fromF32(rotated);

            if (ctx.symmetric) {
                turboquant.quantizeQuery(rotated, ctx.decoded_query);
            } else {
                @memcpy(ctx.decoded_query, rotated);
            }
            const consts = if (self.calib_frozen)
                turboquant.buildScoreLutCalibrated(ctx.decoded_query, self.calib_shift, self.calib_scale, ctx.score_lut)
            else
                turboquant.buildScoreLut(ctx.decoded_query, ctx.score_lut);

            const candidates = self.routing.route(&query_bits, ctx.m, &ctx.scratch);

            if (!self.hasRerankStore()) {
                var topk = heap_mod.TopK.fromBuffer(out);
                for (candidates) |candidate| {
                    const payload = &self.payloads.items[@intCast(candidate.id)];
                    topk.offer(.{
                        .id = payload.id,
                        .score = payload.scoreLut(ctx.score_lut, consts.a, consts.b),
                    });
                }
                return topk.sortDescending();
            }

            // Stage 2: keep the rerank pool by quantized score, tracking
            // internal indices so stage 3 can address the stored originals.
            const pool_size = @min(@max(ctx.rerank_factor * out.len, out.len), ctx.rerank_pool.len);
            var pool = heap_mod.TopK.fromBuffer(ctx.rerank_pool[0..pool_size]);
            for (candidates) |candidate| {
                const payload = &self.payloads.items[@intCast(candidate.id)];
                pool.offer(.{
                    .id = candidate.id,
                    .score = payload.scoreLut(ctx.score_lut, consts.a, consts.b),
                });
            }
            const pooled = pool.sortDescending();

            // Stage 3: exact inner products over the pool.
            var topk = heap_mod.TopK.fromBuffer(out);
            for (ctx.rerank_pool[0..pooled]) |entry| {
                const internal: usize = @intCast(entry.id);
                topk.offer(.{
                    .id = self.payloads.items[internal].id,
                    .score = self.rerankScore(internal, rotated),
                });
            }
            return topk.sortDescending();
        }
    };
}

/// SIMD dot of an FP32 query against int8 codes (caller applies the scale).
fn dotSq8(q: []const f32, codes: []const i8) f32 {
    const width = std.simd.suggestVectorLength(f32) orelse 4;
    var acc: @Vector(width, f32) = @splat(0);
    var i: usize = 0;
    while (i + width <= q.len) : (i += width) {
        const vq: @Vector(width, f32) = q[i..][0..width].*;
        const vi: @Vector(width, i8) = codes[i..][0..width].*;
        const vc: @Vector(width, f32) = @floatFromInt(vi);
        acc = @mulAdd(@Vector(width, f32), vq, vc, acc);
    }
    var total = @reduce(.Add, acc);
    while (i < q.len) : (i += 1) {
        total = @mulAdd(f32, q[i], @floatFromInt(codes[i]), total);
    }
    return total;
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

test "exact rerank returns the stored vector itself on self-query" {
    const dim = 32;
    const Idx = Index(dim, 8);
    const allocator = std.testing.allocator;

    var index = try Idx.init(allocator, 32, 3);
    defer index.deinit(allocator);
    try std.testing.expectEqual(RerankStore.sq8, index.rerank_store);

    var prng = std.Random.DefaultPrng.init(17);
    const rand = prng.random();
    var stored: [40][dim]f32 = undefined;
    for (&stored, 0..) |*coords, i| {
        var norm_sq: f32 = 0;
        for (coords) |*c| {
            c.* = rand.floatNorm(f32);
            norm_sq += c.* * c.*;
        }
        const inv = 1.0 / @sqrt(norm_sq);
        for (coords) |*c| c.* *= inv;
        try index.add(allocator, i, coords);
    }
    try std.testing.expect(index.hasRerankStore());

    var ctx = try Idx.SearchContext.init(allocator, &index, 40);
    defer ctx.deinit(allocator);

    // With exact stage-3 scoring, querying a stored unit vector must rank
    // the vector itself first with score == cosine(v, v) == 1 (sq8 rounding
    // perturbs the dot by well under 1%).
    var out: [5]SearchResult = undefined;
    for (stored, 0..) |coords, i| {
        const count = index.search(&ctx, &coords, &out);
        try std.testing.expect(count == 5);
        try std.testing.expectEqual(@as(u64, i), out[0].id);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0].score, 5e-3);
    }
}

test "sq8 and fp32 rerank stores agree on results" {
    const dim = 48;
    const Idx = Index(dim, 8);
    const allocator = std.testing.allocator;

    var idx_fp32 = try Idx.init(allocator, 32, 9);
    defer idx_fp32.deinit(allocator);
    idx_fp32.rerank_store = .fp32;
    var idx_sq8 = try Idx.init(allocator, 32, 9);
    defer idx_sq8.deinit(allocator);
    idx_sq8.rerank_store = .sq8;

    var prng = std.Random.DefaultPrng.init(31);
    const rand = prng.random();
    var coords: [dim]f32 = undefined;
    for (0..300) |i| {
        for (&coords) |*c| c.* = rand.floatNorm(f32);
        try idx_fp32.add(allocator, i, &coords);
        try idx_sq8.add(allocator, i, &coords);
    }

    var ctx_a = try Idx.SearchContext.init(allocator, &idx_fp32, 64);
    defer ctx_a.deinit(allocator);
    var ctx_b = try Idx.SearchContext.init(allocator, &idx_sq8, 64);
    defer ctx_b.deinit(allocator);

    var agree: usize = 0;
    var out_a: [10]SearchResult = undefined;
    var out_b: [10]SearchResult = undefined;
    for (0..20) |_| {
        for (&coords) |*c| c.* = rand.floatNorm(f32);
        const n_a = idx_fp32.search(&ctx_a, &coords, &out_a);
        const n_b = idx_sq8.search(&ctx_b, &coords, &out_b);
        try std.testing.expectEqual(n_a, n_b);
        if (out_a[0].id == out_b[0].id) agree += 1;
        try std.testing.expectApproxEqRel(out_a[0].score, out_b[0].score, 0.01);
    }
    try std.testing.expect(agree >= 19);
}

test "rerank_store=none falls back to quantized scoring" {
    const dim = 32;
    const Idx = Index(dim, 4);
    const allocator = std.testing.allocator;

    var index = try Idx.init(allocator, 16, 5);
    defer index.deinit(allocator);
    index.rerank_store = .none;

    var prng = std.Random.DefaultPrng.init(13);
    const rand = prng.random();
    var coords: [dim]f32 = undefined;
    for (0..50) |i| {
        for (&coords) |*c| c.* = rand.floatNorm(f32);
        try index.add(allocator, i, &coords);
    }
    try std.testing.expect(!index.hasRerankStore());

    var ctx = try Idx.SearchContext.init(allocator, &index, 16);
    defer ctx.deinit(allocator);
    var out: [5]SearchResult = undefined;
    for (&coords) |*c| c.* = rand.floatNorm(f32);
    try std.testing.expectEqual(@as(usize, 5), index.search(&ctx, &coords, &out));
}

test "tq_plus calibration: buffered ingest, freeze, and self-recall" {
    const dim = 32;
    const Idx = Index(dim, 8);
    const allocator = std.testing.allocator;

    var index = try Idx.init(allocator, 32, 23);
    defer index.deinit(allocator);
    index.tq_plus = true;
    index.calibration_sample = 64; // exercise auto-freeze mid-ingest

    var prng = std.Random.DefaultPrng.init(41);
    const rand = prng.random();
    var stored: [100][dim]f32 = undefined;
    for (&stored, 0..) |*coords, i| {
        // Deliberately skewed coordinates: per-coordinate offsets that the
        // calibration should absorb.
        for (coords, 0..) |*c, j| c.* = rand.floatNorm(f32) + 0.5 * @as(f32, @floatFromInt(j % 4));
        try index.add(allocator, i, coords);
    }
    try index.freeze(allocator); // no-op: auto-freeze already happened
    try std.testing.expect(index.calib_frozen);
    try std.testing.expectEqual(@as(usize, 100), index.len());

    var ctx = try Idx.SearchContext.init(allocator, &index, 64);
    defer ctx.deinit(allocator);
    var out: [5]SearchResult = undefined;
    var hits: usize = 0;
    for (stored, 0..) |coords, i| {
        const count = index.search(&ctx, &coords, &out);
        try std.testing.expect(count == 5);
        if (out[0].id == i) hits += 1;
    }
    if (hits < 90) std.debug.print("tq_plus self-recall hits: {d}/100\n", .{hits});
    try std.testing.expect(hits >= 90);
}

test "tq_plus with fewer adds than the sample needs explicit freeze" {
    const dim = 32;
    const Idx = Index(dim, 4);
    const allocator = std.testing.allocator;

    var index = try Idx.init(allocator, 16, 2);
    defer index.deinit(allocator);
    index.tq_plus = true;

    var prng = std.Random.DefaultPrng.init(6);
    const rand = prng.random();
    var coords: [dim]f32 = undefined;
    for (0..20) |i| {
        for (&coords) |*c| c.* = rand.floatNorm(f32);
        try index.add(allocator, i, &coords);
    }
    try std.testing.expectEqual(@as(usize, 0), index.len()); // still buffered
    try index.freeze(allocator);
    try std.testing.expectEqual(@as(usize, 20), index.len());

    var ctx = try Idx.SearchContext.init(allocator, &index, 16);
    defer ctx.deinit(allocator);
    var out: [5]SearchResult = undefined;
    try std.testing.expectEqual(@as(usize, 5), index.search(&ctx, &coords, &out));
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
