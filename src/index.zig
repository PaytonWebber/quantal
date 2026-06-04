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

/// Recommended routing_bits for a given data dimension, derived from the
/// dimension-crossover and d=1536 sweeps (benchmarks/RESULTS.md):
///   - low dim: the d-bit code is starved, so lift it to an absolute floor
///     (the sweep showed ≈0.90 routing recall@10 needs ~512 bits by d=100,
///     ~1024 by d=200);
///   - high dim: the d-bit code already saturates the angular structure and
///     adding a projection only costs throughput, so use dim unchanged.
/// The result is always ≥ dim — below-dim codes were measured to be a net
/// loss. Used when routing_bits is left at 0 (auto) at the build/C-ABI layer.
pub fn autoRoutingBits(dim: usize) usize {
    if (dim <= 64) return 256;
    if (dim <= 128) return 512;
    if (dim <= 256) return 1024;
    return dim;
}

/// `routing_bits` is the SimHash routing-code length. The default, `dim`,
/// routes on the sign bits of the rotated vector exactly as the 1-bit-per-
/// dimension design always has (no projection, no cost). Any other value
/// adds a `routing_bits × dim` random projection and routes on
/// sign(R · rotated): larger than `dim` sharpens routing on low-dimensional
/// data (the low-dim fix), while *smaller* than `dim` is a Johnson-
/// Lindenstrauss reduction that trades a little routing recall for a shorter
/// code (cheaper Hamming + projection) on high-dimensional data. See
/// benchmarks/RESULTS.md.
///
/// Concurrency: this follows the single-writer / many-reader contract.
/// Concurrent searches are fine (searchBatch runs them in parallel), but a
/// mutation (add, addBatch, remove, freeze) must not overlap with any search
/// or other mutation.
pub fn Index(comptime dim: usize, comptime max_edges: usize, comptime routing_bits: usize) type {
    comptime std.debug.assert(routing_bits >= 1);
    return struct {
        const Self = @This();
        pub const Payload = turboquant.TurboQuantPayload(dim);
        pub const Graph = graph_mod.RoutingGraph(routing_bits, max_edges);
        pub const Rotation = rotation_mod.RandomRotation(dim);
        pub const Projection = rotation_mod.RandomProjection(routing_bits, dim);
        pub const dimension = dim;
        pub const routing_bit_count = routing_bits;
        /// When false (routing_bits == dim) the routing code is the rotated
        /// vector's signs and no projection matrix exists.
        pub const uses_projection = routing_bits != dim;

        rotation: Rotation,
        /// Routing projection (only allocated when uses_projection).
        projection: if (uses_projection) Projection else void = if (uses_projection) undefined else {},
        /// Scratch holding the projected routing vector during ingest
        /// (length routing_bits; empty when !uses_projection).
        route_buf: []f32 = &.{},
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
        /// User id -> internal index; backs `remove` and filtered search.
        id_to_internal: std.AutoHashMapUnmanaged(u64, u64) = .empty,
        /// Tombstoned internals (lazy: empty until the first remove). The
        /// graph keeps routing through deleted nodes; they are only skipped
        /// at scoring time.
        tombstones: std.DynamicBitSetUnmanaged = .{},
        live_count: usize = 0,

        pub fn init(allocator: std.mem.Allocator, ef_construction: usize, seed: u64) !Self {
            var rot = try Rotation.init(allocator, seed);
            errdefer rot.deinit(allocator);
            const rotate_buf = try allocator.alloc(f32, dim);
            errdefer allocator.free(rotate_buf);
            const calib_buf = try allocator.alloc(f32, dim);
            errdefer allocator.free(calib_buf);

            var self = Self{
                .rotation = rot,
                .routing = Graph.init(ef_construction),
                .rotate_buf = rotate_buf,
                .calib_buf = calib_buf,
            };
            if (uses_projection) {
                // Independent seed so the projection is uncorrelated with Π.
                self.projection = try Projection.init(allocator, seed ^ 0xD1CE_5EED_1234_ABCD);
                self.route_buf = try allocator.alloc(f32, routing_bits);
            }
            return self;
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.rotation.deinit(allocator);
            if (uses_projection) {
                self.projection.deinit(allocator);
                allocator.free(self.route_buf);
            }
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
            self.id_to_internal.deinit(allocator);
            self.tombstones.deinit(allocator);
            self.* = undefined;
        }

        /// Number of live (non-deleted) vectors.
        pub fn len(self: *const Self) usize {
            return self.live_count;
        }

        /// Number of internal slots, including tombstoned ones (graph nodes
        /// and payload records are never compacted in place).
        pub fn capacity(self: *const Self) usize {
            return self.payloads.items.len;
        }

        pub fn isDeleted(self: *const Self, internal: usize) bool {
            return internal < self.tombstones.bit_length and self.tombstones.isSet(internal);
        }

        /// Tombstones a vector by user id in O(1). Returns false when the
        /// id is unknown (or already removed).
        pub fn remove(self: *Self, allocator: std.mem.Allocator, id: u64) !bool {
            const entry = self.id_to_internal.fetchRemove(id) orelse return false;
            const internal: usize = @intCast(entry.value);
            if (internal >= self.tombstones.bit_length) {
                try self.tombstones.resize(allocator, self.payloads.items.len, false);
            }
            self.tombstones.set(internal);
            self.live_count -= 1;
            return true;
        }

        /// Adds one vector. Errors (leaving the index unchanged) on a
        /// wrong-length slice or an id that is already present.
        pub fn add(self: *Self, allocator: std.mem.Allocator, id: u64, coords: []const f32) !void {
            if (coords.len != dim) return error.DimensionMismatch;
            if (self.id_to_internal.contains(id)) return error.DuplicateId;
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

            try self.appendRerankRecord(allocator, rotated);
            errdefer switch (self.rerank_store) {
                .none => {},
                .fp32 => self.originals.shrinkRetainingCapacity(internal_id * dim),
                .sq8 => {
                    self.sq8_codes.shrinkRetainingCapacity(internal_id * dim);
                    self.sq8_scales.shrinkRetainingCapacity(internal_id);
                },
            };

            const bits = self.routeCode(rotated, self.route_buf);
            try self.routing.insert(allocator, internal_id, bits);
            try self.registerId(allocator, id, internal_id);
        }

        /// Routing code for a rotated vector. With routing_bits == dim this
        /// is just the rotated signs (the original 1-bit-per-dim code). With
        /// routing_bits > dim it is sign(R · rotated); since R · (Π·x) is a
        /// valid SimHash of x, scratch must be `routing_bits` long.
        fn routeCode(self: *const Self, rotated: []const f32, scratch: []f32) Graph.BitVec {
            if (uses_projection) {
                self.projection.project(rotated, scratch);
                return Graph.BitVec.fromF32(scratch);
            }
            return Graph.BitVec.fromF32(rotated);
        }

        fn registerId(self: *Self, allocator: std.mem.Allocator, user_id: u64, internal: u64) !void {
            const gop = try self.id_to_internal.getOrPut(allocator, user_id);
            if (gop.found_existing) return error.DuplicateId;
            gop.value_ptr.* = internal;
            self.live_count += 1;
        }

        /// Rejects a batch (before any mutation) if any id is already indexed
        /// or appears twice within the batch. Uses a scratch set so the check
        /// is O(n), not O(n^2).
        fn validateNewIds(self: *Self, allocator: std.mem.Allocator, ids: []const u64) !void {
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(allocator);
            try seen.ensureTotalCapacity(allocator, @intCast(ids.len));
            for (ids) |id| {
                if (self.id_to_internal.contains(id)) return error.DuplicateId;
                if (seen.fetchPutAssumeCapacity(id, {}) != null) return error.DuplicateId;
            }
        }

        fn appendRerankRecord(self: *Self, allocator: std.mem.Allocator, rotated: []const f32) !void {
            switch (self.rerank_store) {
                .none => {},
                .fp32 => try self.originals.appendSlice(allocator, rotated),
                .sq8 => {
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
        }

        /// Multi-threaded bulk ingest. Equivalent to calling `add` for each
        /// (id, vector) pair, except vectors within the same internal batch
        /// do not see each other while their graph neighbors are planned
        /// (batch size grows with the graph, so early inserts stay serial).
        /// Not allowed while a TQ+ calibration buffer is still open.
        ///
        /// Ids and dimensions are validated up front, so a wrong-length
        /// `coords` or any duplicate id (within the batch or already in the
        /// index) is rejected before any vector is inserted. Allocation
        /// failure mid-batch is not rolled back, however.
        pub fn addBatch(
            self: *Self,
            allocator: std.mem.Allocator,
            ids: []const u64,
            coords: []const f32,
            thread_count: usize,
        ) !void {
            std.debug.assert(!self.tq_plus or self.calib_frozen);
            if (coords.len != ids.len * dim) return error.DimensionMismatch;
            try self.validateNewIds(allocator, ids);
            const threads = @max(thread_count, 1);
            const max_batch: usize = 1024;

            const Staged = struct {
                rotated: []f32,
                payloads: []Payload,
                bits: []Graph.BitVec,
                levels: []usize,
                plans: []Graph.InsertPlan,
            };
            const staged = Staged{
                .rotated = try allocator.alloc(f32, max_batch * dim),
                .payloads = try allocator.alloc(Payload, max_batch),
                .bits = try allocator.alloc(Graph.BitVec, max_batch),
                .levels = try allocator.alloc(usize, max_batch),
                .plans = try allocator.alloc(Graph.InsertPlan, max_batch),
            };
            defer {
                allocator.free(staged.rotated);
                allocator.free(staged.payloads);
                allocator.free(staged.bits);
                allocator.free(staged.levels);
                allocator.free(staged.plans);
            }
            const scratches = try allocator.alloc(graph_mod.TraversalScratch, threads);
            defer {
                for (scratches) |*s| s.deinit(allocator);
                allocator.free(scratches);
            }
            @memset(scratches, .{});
            // Per-worker projection scratch (routing_bits each); unused when
            // !uses_projection.
            const route_scratch = try allocator.alloc(f32, if (uses_projection) threads * routing_bits else 0);
            defer allocator.free(route_scratch);
            const worker_threads = try allocator.alloc(std.Thread, threads);
            defer allocator.free(worker_threads);

            const Worker = struct {
                index: *const Self,
                staged: *const Staged,
                batch_coords: []const f32,
                batch_ids: []const u64,
                batch_len: usize,
                stride: usize,

                fn run(w: @This(), t: usize, scratch: *graph_mod.TraversalScratch, route: []f32) void {
                    var i = t;
                    while (i < w.batch_len) : (i += w.stride) {
                        const rotated = w.staged.rotated[i * dim ..][0..dim];
                        w.index.rotation.apply(w.batch_coords[i * dim ..][0..dim], rotated);
                        w.staged.payloads[i] = Payload.encodeWithCalibration(
                            w.batch_ids[i],
                            rotated,
                            w.index.calib_shift,
                            w.index.calib_scale,
                        );
                        w.staged.bits[i] = w.index.routeCode(rotated, route);
                        w.staged.plans[i] = w.index.routing.planInsert(&w.staged.bits[i], w.staged.levels[i], scratch);
                    }
                }

                fn routeFor(_: @This(), buf: []f32, t: usize) []f32 {
                    return if (uses_projection) buf[t * routing_bits ..][0..routing_bits] else &.{};
                }
            };

            var next: usize = 0;
            while (next < ids.len) {
                const batch = @min(@min(max_batch, @max(self.routing.node_count, 1)), ids.len - next);
                for (staged.levels[0..batch]) |*level| level.* = self.routing.drawLevel();
                for (scratches) |*s| {
                    try s.ensureCapacity(allocator, self.routing.node_count + 1, self.routing.ef_construction);
                }

                const worker = Worker{
                    .index = self,
                    .staged = &staged,
                    .batch_coords = coords[next * dim ..],
                    .batch_ids = ids[next..],
                    .batch_len = batch,
                    .stride = threads,
                };
                if (threads == 1 or batch == 1) {
                    worker.run(0, &scratches[0], worker.routeFor(route_scratch, 0));
                } else {
                    for (worker_threads[0..threads], 0..) |*thread, t| {
                        thread.* = try std.Thread.spawn(.{}, Worker.run, .{ worker, t, &scratches[t], worker.routeFor(route_scratch, t) });
                    }
                    for (worker_threads[0..threads]) |thread| thread.join();
                }

                for (0..batch) |i| {
                    const internal_id: u64 = self.payloads.items.len;
                    const rotated = staged.rotated[i * dim ..][0..dim];
                    try self.payloads.append(allocator, staged.payloads[i]);
                    try self.appendRerankRecord(allocator, rotated);
                    if (staged.plans[i].planned) {
                        try self.routing.commitPlanned(allocator, internal_id, staged.bits[i], staged.plans[i]);
                    } else {
                        try self.routing.insertWithLevel(allocator, internal_id, staged.bits[i], staged.levels[i]);
                    }
                    try self.registerId(allocator, ids[next + i], internal_id);
                }
                next += batch;
            }
        }

        /// Multi-threaded query batch: `queries` is n*dim coordinates,
        /// `out_results` n*k slots, `out_counts` n entries. Each thread owns
        /// a private SearchContext (this call allocates; per-query search
        /// remains allocation-free).
        pub fn searchBatch(
            self: *const Self,
            allocator: std.mem.Allocator,
            queries: []const f32,
            k: usize,
            m: usize,
            thread_count: usize,
            symmetric: bool,
            out_results: []SearchResult,
            out_counts: []usize,
        ) !void {
            const n = queries.len / dim;
            std.debug.assert(queries.len == n * dim);
            std.debug.assert(out_results.len >= n * k and out_counts.len >= n);
            const threads = @min(@max(thread_count, 1), @max(n, 1));

            const contexts = try allocator.alloc(SearchContext, threads);
            var ready: usize = 0;
            defer {
                for (contexts[0..ready]) |*ctx| ctx.deinit(allocator);
                allocator.free(contexts);
            }
            for (contexts) |*ctx| {
                ctx.* = try SearchContext.init(allocator, self, m);
                ctx.symmetric = symmetric;
                ready += 1;
            }

            const Worker = struct {
                fn run(index: *const Self, ctx: *SearchContext, qs: []const f32, t: usize, stride: usize, k_: usize, results: []SearchResult, counts: []usize) void {
                    var i = t;
                    const total = qs.len / dim;
                    while (i < total) : (i += stride) {
                        counts[i] = index.search(ctx, qs[i * dim ..][0..dim], results[i * k_ ..][0..k_]);
                    }
                }
            };

            if (threads == 1) {
                Worker.run(self, &contexts[0], queries, 0, 1, k, out_results, out_counts);
                return;
            }
            const worker_threads = try allocator.alloc(std.Thread, threads);
            defer allocator.free(worker_threads);
            for (worker_threads, 0..) |*thread, t| {
                thread.* = try std.Thread.spawn(.{}, Worker.run, .{ self, &contexts[t], queries, t, threads, k, out_results, out_counts });
            }
            for (worker_threads) |thread| thread.join();
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
            /// Projected routing query (length routing_bits; empty when
            /// !uses_projection).
            projected_query: []f32 = &.{},
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
                if (uses_projection) ctx.projected_query = try allocator.alloc(f32, routing_bits);
                errdefer if (uses_projection) allocator.free(ctx.projected_query);
                try ctx.scratch.ensureCapacity(allocator, @max(index.capacity(), 1), m);
                return ctx;
            }

            pub fn deinit(self: *SearchContext, allocator: std.mem.Allocator) void {
                self.scratch.deinit(allocator);
                allocator.free(self.rotated_query);
                if (uses_projection) allocator.free(self.projected_query);
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
            std.debug.assert(ctx.scratch.visited.bit_length >= self.capacity());

            // Routing bits are raw-space on both sides (see insertRotated);
            // stage-2 LUT scoring and stage 3 also estimate inner products
            // in the raw rotated space.
            const query_bits = self.routeCode(rotated, ctx.projected_query);

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
                    if (self.isDeleted(@intCast(candidate.id))) continue;
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
                if (self.isDeleted(@intCast(candidate.id))) continue;
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

        /// Restricts results to an allowlist of user ids (e.g. produced by
        /// an external SQL/BM25/ACL stage). Scores every allowed vector
        /// directly — exact for any rerank store, no routing recall loss,
        /// O(|allowlist| * dim) — which beats graph traversal for the
        /// selective filters allowlists are used for. Unknown and removed
        /// ids are skipped. Allocation-free.
        pub fn searchFiltered(
            self: *const Self,
            ctx: *SearchContext,
            query: []const f32,
            allowed_ids: []const u64,
            out: []SearchResult,
        ) usize {
            std.debug.assert(query.len == dim);
            std.debug.assert(!self.tq_plus or self.calib_frozen);
            if (out.len == 0) return 0;
            self.rotation.apply(query, ctx.rotated_query);

            // LUT only needed when there is no exact store to score with.
            var consts: turboquant.LutConstants = .{ .a = 0, .b = 0 };
            const exact = self.hasRerankStore();
            if (!exact) {
                if (ctx.symmetric) {
                    turboquant.quantizeQuery(ctx.rotated_query, ctx.decoded_query);
                } else {
                    @memcpy(ctx.decoded_query, ctx.rotated_query);
                }
                consts = if (self.calib_frozen)
                    turboquant.buildScoreLutCalibrated(ctx.decoded_query, self.calib_shift, self.calib_scale, ctx.score_lut)
                else
                    turboquant.buildScoreLut(ctx.decoded_query, ctx.score_lut);
            }

            var topk = heap_mod.TopK.fromBuffer(out);
            for (allowed_ids) |user_id| {
                const internal: usize = @intCast(self.id_to_internal.get(user_id) orelse continue);
                topk.offer(.{
                    .id = user_id,
                    .score = if (exact)
                        self.rerankScore(internal, ctx.rotated_query)
                    else
                        self.payloads.items[internal].scoreLut(ctx.score_lut, consts.a, consts.b),
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
    const Idx = Index(32, 4, 32);
    var index = try Idx.init(std.testing.allocator, 16, 1);
    defer index.deinit(std.testing.allocator);

    var ctx = try Idx.SearchContext.init(std.testing.allocator, &index, 8);
    defer ctx.deinit(std.testing.allocator);

    const query: [32]f32 = @splat(1.0);
    var out: [4]SearchResult = undefined;
    try std.testing.expectEqual(@as(usize, 0), index.search(&ctx, &query, &out));
}

test "autoRoutingBits matches the measured table and never goes below dim" {
    try std.testing.expectEqual(@as(usize, 256), autoRoutingBits(25));
    try std.testing.expectEqual(@as(usize, 256), autoRoutingBits(64));
    try std.testing.expectEqual(@as(usize, 512), autoRoutingBits(100));
    try std.testing.expectEqual(@as(usize, 512), autoRoutingBits(128));
    try std.testing.expectEqual(@as(usize, 1024), autoRoutingBits(200));
    try std.testing.expectEqual(@as(usize, 1024), autoRoutingBits(256));
    try std.testing.expectEqual(@as(usize, 768), autoRoutingBits(768));
    try std.testing.expectEqual(@as(usize, 1536), autoRoutingBits(1536));
    // Invariant: auto is never a below-dim (lossy) code.
    for ([_]usize{ 16, 50, 100, 200, 384, 1536, 3072 }) |d| {
        try std.testing.expect(autoRoutingBits(d) >= d);
    }
}

test "multi-bit routing: projection improves low-dim recall and self-query works" {
    const dim = 16;
    const allocator = std.testing.allocator;
    const Wide = Index(dim, 8, 256); // routing_bits > dim -> uses projection
    try std.testing.expect(Wide.uses_projection);

    var index = try Wide.init(allocator, 32, 4);
    defer index.deinit(allocator);

    var prng = std.Random.DefaultPrng.init(88);
    const rand = prng.random();
    var stored: [120][dim]f32 = undefined;
    for (&stored, 0..) |*coords, i| {
        for (coords) |*c| c.* = rand.floatNorm(f32);
        try index.add(allocator, i, coords);
    }

    var ctx = try Wide.SearchContext.init(allocator, &index, 64);
    defer ctx.deinit(allocator);
    var out: [5]SearchResult = undefined;
    var hits: usize = 0;
    for (stored, 0..) |coords, i| {
        const count = index.search(&ctx, &coords, &out);
        try std.testing.expect(count == 5);
        if (out[0].id == i) hits += 1;
    }
    // Exact rerank guarantees the self-vector ranks first whenever routing
    // surfaces it; a 256-bit code on 16-dim data routes almost all of them.
    if (hits < 110) std.debug.print("multi-bit self-recall hits: {d}/120\n", .{hits});
    try std.testing.expect(hits >= 110);

    // Serialize/deserialize must carry the projection matrix.
    var labels: [120][]const u8 = undefined;
    for (&labels) |*l| l.* = "v";
    const storage = @import("storage.zig");
    const bytes = try storage.serialize(Wide, allocator, &index, &labels);
    defer allocator.free(bytes);
    var loaded = try storage.deserialize(Wide, allocator, bytes);
    defer loaded.deinit(allocator);
    try std.testing.expect(Wide.uses_projection);
    var ctx2 = try Wide.SearchContext.init(allocator, &loaded.index, 64);
    defer ctx2.deinit(allocator);
    var out2: [5]SearchResult = undefined;
    const n2 = loaded.index.search(&ctx2, &stored[3], &out2);
    try std.testing.expectEqual(@as(usize, 5), n2);
    try std.testing.expectEqual(@as(u64, 3), out2[0].id);
}

test "clustered recall: queries land in their own cluster" {
    const dim = 32;
    const clusters = 10;
    const per_cluster = 50;
    const Idx = Index(dim, 8, dim);

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
    const Idx = Index(dim, 8, dim);
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
    const Idx = Index(dim, 8, dim);
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

test "edge cases: duplicate id, dimension mismatch, all-deleted, k>m" {
    const dim = 32;
    const Idx = Index(dim, 8, dim);
    const allocator = std.testing.allocator;

    var index = try Idx.init(allocator, 32, 1);
    defer index.deinit(allocator);

    var prng = std.Random.DefaultPrng.init(2);
    const rand = prng.random();
    var coords: [dim]f32 = undefined;
    for (&coords) |*c| c.* = rand.floatNorm(f32);

    // dimension mismatch and duplicate id are rejected, index unchanged.
    try std.testing.expectError(error.DimensionMismatch, index.add(allocator, 1, coords[0 .. dim - 1]));
    try index.add(allocator, 1, &coords);
    try std.testing.expectError(error.DuplicateId, index.add(allocator, 1, &coords));
    try std.testing.expectEqual(@as(usize, 1), index.len());

    // addBatch rejects a within-batch duplicate before mutating.
    var batch: [3 * dim]f32 = undefined;
    for (&batch) |*c| c.* = rand.floatNorm(f32);
    try std.testing.expectError(error.DuplicateId, index.addBatch(allocator, &.{ 2, 3, 2 }, &batch, 2));
    try std.testing.expectEqual(@as(usize, 1), index.len()); // unchanged

    // k > m: asking for more than the beam returns at most what routing found.
    var ctx = try Idx.SearchContext.init(allocator, &index, 4);
    defer ctx.deinit(allocator);
    var out: [16]SearchResult = undefined;
    const n = index.search(&ctx, &coords, &out);
    try std.testing.expect(n >= 1 and n <= 16);

    // all-deleted index returns zero results, no crash.
    try std.testing.expect(try index.remove(allocator, 1));
    try std.testing.expectEqual(@as(usize, 0), index.len());
    try std.testing.expectEqual(@as(usize, 0), index.search(&ctx, &coords, &out));
}

test "rerank_store=none falls back to quantized scoring" {
    const dim = 32;
    const Idx = Index(dim, 4, dim);
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
    const Idx = Index(dim, 8, dim);
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
    const Idx = Index(dim, 4, dim);
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

test "remove tombstones a vector and search skips it" {
    const dim = 32;
    const Idx = Index(dim, 8, dim);
    const allocator = std.testing.allocator;

    var index = try Idx.init(allocator, 32, 19);
    defer index.deinit(allocator);

    var prng = std.Random.DefaultPrng.init(3);
    const rand = prng.random();
    var stored: [50][dim]f32 = undefined;
    for (&stored, 0..) |*coords, i| {
        for (coords) |*c| c.* = rand.floatNorm(f32);
        try index.add(allocator, 100 + i, coords);
    }
    try std.testing.expectEqual(@as(usize, 50), index.len());

    var ctx = try Idx.SearchContext.init(allocator, &index, 50);
    defer ctx.deinit(allocator);
    var out: [5]SearchResult = undefined;

    // Self-query finds id 107, then removing it must hide it.
    _ = index.search(&ctx, &stored[7], &out);
    try std.testing.expectEqual(@as(u64, 107), out[0].id);

    try std.testing.expect(try index.remove(allocator, 107));
    try std.testing.expect(!try index.remove(allocator, 107)); // idempotent
    try std.testing.expect(!try index.remove(allocator, 9999)); // unknown
    try std.testing.expectEqual(@as(usize, 49), index.len());

    const count = index.search(&ctx, &stored[7], &out);
    for (out[0..count]) |result| {
        try std.testing.expect(result.id != 107);
    }
}

test "searchFiltered restricts results to the allowlist" {
    const dim = 32;
    const Idx = Index(dim, 8, dim);
    const allocator = std.testing.allocator;

    var index = try Idx.init(allocator, 32, 29);
    defer index.deinit(allocator);

    var prng = std.Random.DefaultPrng.init(8);
    const rand = prng.random();
    var coords: [dim]f32 = undefined;
    for (0..200) |i| {
        for (&coords) |*c| c.* = rand.floatNorm(f32);
        try index.add(allocator, i, &coords);
    }

    var ctx = try Idx.SearchContext.init(allocator, &index, 32);
    defer ctx.deinit(allocator);

    const allowlist = [_]u64{ 3, 17, 42, 99, 150, 7777 }; // 7777 unknown
    var query: [dim]f32 = undefined;
    for (&query) |*q| q.* = rand.floatNorm(f32);

    var out: [10]SearchResult = undefined;
    const count = index.searchFiltered(&ctx, &query, &allowlist, &out);
    try std.testing.expectEqual(@as(usize, 5), count);
    for (out[0..count]) |result| {
        try std.testing.expect(std.mem.indexOfScalar(u64, &allowlist, result.id) != null);
    }
    for (out[0 .. count - 1], out[1..count]) |a, b| {
        try std.testing.expect(a.score >= b.score);
    }

    // Removing an allowed id shrinks the result set.
    try std.testing.expect(try index.remove(allocator, 42));
    try std.testing.expectEqual(@as(usize, 4), index.searchFiltered(&ctx, &query, &allowlist, &out));
}

test "addBatch + searchBatch match serial quality" {
    const dim = 32;
    const Idx = Index(dim, 8, dim);
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(73);
    const rand = prng.random();
    const n = 400;
    const coords = try allocator.alloc(f32, n * dim);
    defer allocator.free(coords);
    for (coords) |*c| c.* = rand.floatNorm(f32);
    const ids = try allocator.alloc(u64, n);
    defer allocator.free(ids);
    for (ids, 0..) |*id, i| id.* = i;

    var serial = try Idx.init(allocator, 32, 12);
    defer serial.deinit(allocator);
    for (0..n) |i| try serial.add(allocator, i, coords[i * dim ..][0..dim]);

    var batched = try Idx.init(allocator, 32, 12);
    defer batched.deinit(allocator);
    try batched.addBatch(allocator, ids, coords, 4);
    try std.testing.expectEqual(serial.len(), batched.len());

    // Self-queries through both the serial and batch search paths: the
    // batched graph differs (within-batch blindness) but must stay near
    // serial quality.
    const nq = 100;
    const results = try allocator.alloc(SearchResult, nq * 5);
    defer allocator.free(results);
    const counts = try allocator.alloc(usize, nq);
    defer allocator.free(counts);
    try batched.searchBatch(allocator, coords[0 .. nq * dim], 5, 64, 4, true, results, counts);

    var ctx = try Idx.SearchContext.init(allocator, &serial, 64);
    defer ctx.deinit(allocator);
    var out: [5]SearchResult = undefined;
    var serial_hits: usize = 0;
    var batch_hits: usize = 0;
    for (0..nq) |i| {
        try std.testing.expectEqual(@as(usize, 5), counts[i]);
        _ = serial.search(&ctx, coords[i * dim ..][0..dim], &out);
        if (out[0].id == i) serial_hits += 1;
        if (results[i * 5].id == i) batch_hits += 1;
    }
    try std.testing.expect(batch_hits + 5 >= serial_hits);
    try std.testing.expect(batch_hits >= 90);
}

test "results are sorted by descending score" {
    const dim = 32;
    const Idx = Index(dim, 4, dim);
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
