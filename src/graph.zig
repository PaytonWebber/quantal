//! Stage 1: HNSW-style routing graph over 1-bit sign vectors.
//!
//! Distances are Hamming distances (XOR + POPCOUNT). Nodes carry fixed-size
//! neighbor arrays, so traversal never chases per-node allocations, and all
//! query-time scratch lives in a caller-owned TraversalScratch.

const std = @import("std");
const bitvec = @import("bitvec.zig");
const heap_mod = @import("heap.zig");

pub fn GraphNode(comptime bits: usize, comptime max_edges: usize) type {
    return struct {
        id: u64,
        bit_vector: bitvec.BitVector(bits),
        edge_count: u32,
        neighbors: [max_edges]u64,
    };
}

pub const Candidate = struct {
    dist: u32,
    id: u64,
};

fn closer(a: Candidate, b: Candidate) bool {
    return a.dist < b.dist;
}

fn farther(a: Candidate, b: Candidate) bool {
    return a.dist > b.dist;
}

const CandidateMinHeap = heap_mod.BinaryHeap(Candidate, closer);
const CandidateMaxHeap = heap_mod.BinaryHeap(Candidate, farther);

/// Counters describing the last routed query, for diagnostics.
pub const TraversalStats = struct {
    layers_traversed: usize = 0,
    nodes_evaluated: usize = 0,
};

/// Reusable buffers for beam search. Sized so a traversal can never overflow:
/// every node is pushed to the candidate heap at most once (the visited bit
/// is set before pushing), so node_count slots suffice.
pub const TraversalScratch = struct {
    visited: std.DynamicBitSetUnmanaged = .{},
    cand_buf: []Candidate = &.{},
    result_buf: []Candidate = &.{},
    stats: TraversalStats = .{},

    pub fn ensureCapacity(
        self: *TraversalScratch,
        allocator: std.mem.Allocator,
        node_count: usize,
        ef: usize,
    ) !void {
        if (self.visited.bit_length < node_count) {
            try self.visited.resize(allocator, node_count, false);
        }
        if (self.cand_buf.len < node_count) {
            allocator.free(self.cand_buf);
            self.cand_buf = try allocator.alloc(Candidate, node_count);
        }
        if (self.result_buf.len < ef) {
            allocator.free(self.result_buf);
            self.result_buf = try allocator.alloc(Candidate, ef);
        }
    }

    pub fn deinit(self: *TraversalScratch, allocator: std.mem.Allocator) void {
        self.visited.deinit(allocator);
        allocator.free(self.cand_buf);
        allocator.free(self.result_buf);
        self.* = .{};
    }
};

/// `bits` is the routing-code length (the SimHash/sign-bit count), which the
/// caller decouples from the data dimension; the graph only ever compares
/// bit vectors by Hamming distance and never sees the original vectors.
pub fn RoutingGraph(comptime bits: usize, comptime max_edges: usize) type {
    comptime std.debug.assert(max_edges >= 2);
    return struct {
        const Self = @This();
        pub const Node = GraphNode(bits, max_edges);
        pub const BitVec = bitvec.BitVector(bits);

        const Layer = struct {
            nodes: std.ArrayList(Node) = .empty,
            /// Contiguous mirror of the node bit vectors. Distance checks
            /// read 16-64 bytes per neighbor; serving them from this compact
            /// slab keeps the working set L2-resident instead of dragging a
            /// full node struct through the cache per evaluation.
            bit_vectors: std.ArrayList(BitVec) = .empty,
            slot_by_id: std.AutoHashMapUnmanaged(u64, u32) = .empty,
            /// The base layer receives every node in id order, so slot == id
            /// and the hash map can be skipped on the hottest lookup path.
            dense: bool = false,

            fn deinit(layer: *Layer, allocator: std.mem.Allocator) void {
                layer.nodes.deinit(allocator);
                layer.bit_vectors.deinit(allocator);
                layer.slot_by_id.deinit(allocator);
            }

            fn slotOf(layer: *const Layer, id: u64) u32 {
                if (layer.dense) return @intCast(id);
                return layer.slot_by_id.get(id).?;
            }

            fn nodePtr(layer: *const Layer, id: u64) *const Node {
                return &layer.nodes.items[layer.slotOf(id)];
            }

            fn nodePtrMut(layer: *Layer, id: u64) *Node {
                return &layer.nodes.items[layer.slotOf(id)];
            }

            fn bitsPtr(layer: *const Layer, id: u64) *const BitVec {
                return &layer.bit_vectors.items[layer.slotOf(id)];
            }

            fn addNode(layer: *Layer, allocator: std.mem.Allocator, node: Node) !void {
                const slot: u32 = @intCast(layer.nodes.items.len);
                try layer.nodes.append(allocator, node);
                try layer.bit_vectors.append(allocator, node.bit_vector);
                if (layer.dense) {
                    std.debug.assert(node.id == slot);
                } else {
                    try layer.slot_by_id.put(allocator, node.id, slot);
                }
            }
        };

        layers: std.ArrayList(Layer) = .empty,
        entry_id: u64 = 0,
        node_count: usize = 0,
        ef_construction: usize,
        level_rng: std.Random.DefaultPrng,
        build_scratch: TraversalScratch = .{},

        pub fn init(ef_construction: usize) Self {
            return .{
                .ef_construction = @max(ef_construction, max_edges),
                .level_rng = std.Random.DefaultPrng.init(0x9E3779B97F4A7C15),
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            for (self.layers.items) |*layer| layer.deinit(allocator);
            self.layers.deinit(allocator);
            self.build_scratch.deinit(allocator);
            self.* = undefined;
        }

        /// Heap bytes owned by the graph, counting allocated capacity.
        /// Hash-map overhead is approximated from entry size.
        pub fn memoryBytes(self: *const Self) usize {
            var total: usize = self.layers.capacity * @sizeOf(Layer);
            for (self.layers.items) |*layer| {
                total += layer.nodes.capacity * @sizeOf(Node);
                total += layer.bit_vectors.capacity * @sizeOf(BitVec);
                total += layer.slot_by_id.capacity() * (@sizeOf(u64) + @sizeOf(u32) + 1);
            }
            total += (self.build_scratch.visited.bit_length + 63) / 64 * 8;
            total += self.build_scratch.cand_buf.len * @sizeOf(Candidate);
            total += self.build_scratch.result_buf.len * @sizeOf(Candidate);
            return total;
        }

        /// Inserts a node under a dense id (ids must arrive as 0, 1, 2, ...;
        /// they double as indices into the visited bitset and payload array).
        pub fn insert(self: *Self, allocator: std.mem.Allocator, id: u64, code: BitVec) !void {
            try self.insertWithLevel(allocator, id, code, self.randomLevel());
        }

        /// Serial insert at a pre-drawn level (used by the batched build
        /// when a plan could not be applied).
        pub fn insertWithLevel(self: *Self, allocator: std.mem.Allocator, id: u64, code: BitVec, level: usize) !void {
            std.debug.assert(id == self.node_count);
            try self.build_scratch.ensureCapacity(allocator, self.node_count + 1, self.ef_construction);

            const fresh = Node{
                .id = id,
                .bit_vector = code,
                .edge_count = 0,
                .neighbors = @splat(0),
            };

            if (self.node_count == 0) {
                try self.ensureLayerCount(allocator, level + 1);
                for (self.layers.items) |*layer| try layer.addNode(allocator, fresh);
                self.entry_id = id;
                self.node_count = 1;
                return;
            }

            const old_top = self.layers.items.len - 1;
            var cur = self.entry_id;
            var l = old_top;
            while (l > level) : (l -= 1) {
                cur = greedyDescend(&self.layers.items[l], &code, cur, &self.build_scratch.stats);
            }

            var connect = @min(level, old_top) + 1;
            while (connect > 0) {
                connect -= 1;
                const layer = &self.layers.items[connect];
                const found = searchLayer(layer, &code, cur, self.ef_construction, &self.build_scratch);

                try layer.addNode(allocator, fresh);
                const new_node = layer.nodePtrMut(id);
                new_node.edge_count = @intCast(selectNeighbors(layer, found, &new_node.neighbors));
                for (new_node.neighbors[0..new_node.edge_count]) |neighbor_id| {
                    linkBack(layer, neighbor_id, id);
                }
                if (found.len > 0) cur = found[0].id;
            }

            if (level > old_top) {
                try self.ensureLayerCount(allocator, level + 1);
                for (self.layers.items[old_top + 1 ..]) |*layer| {
                    try layer.addNode(allocator, fresh);
                }
                self.entry_id = id;
            }
            self.node_count += 1;
        }

        /// Stage-1 routing: greedy descent through the upper layers, then a
        /// beam search of width m at the base layer. Returns candidates
        /// sorted by ascending Hamming distance, in scratch.result_buf.
        /// Performs no allocation; scratch must be sized for node_count and m.
        pub fn route(self: *const Self, query: *const BitVec, m: usize, scratch: *TraversalScratch) []Candidate {
            if (self.node_count == 0 or m == 0) return &.{};
            std.debug.assert(scratch.visited.bit_length >= self.node_count);
            scratch.stats = .{ .layers_traversed = self.layers.items.len };

            var cur = self.entry_id;
            var l = self.layers.items.len - 1;
            while (l > 0) : (l -= 1) {
                cur = greedyDescend(&self.layers.items[l], query, cur, &scratch.stats);
            }
            const ef = @min(m, self.node_count);
            return searchLayer(&self.layers.items[0], query, cur, ef, scratch);
        }

        /// Maximum node level the batched (plan/commit) insert path handles;
        /// deeper draws (P ~ max_edges^-4) take the serial path instead.
        pub const plan_max_layers = 4;

        /// A precomputed insertion: the selected neighbors per layer.
        /// Produced read-only by `planInsert` (safe to run concurrently),
        /// applied by `commitPlanned` (single writer).
        pub const InsertPlan = struct {
            planned: bool = false,
            level: usize = 0,
            counts: [plan_max_layers]u32 = @splat(0),
            neighbors: [plan_max_layers][max_edges]u64 = undefined,
        };

        /// Read-only insertion planning against the current graph. Returns
        /// an unplanned result (caller must use the serial `insert`) when
        /// the graph is empty or the drawn level needs layer promotion.
        pub fn planInsert(self: *const Self, code: *const BitVec, level: usize, scratch: *TraversalScratch) InsertPlan {
            if (self.node_count == 0) return .{};
            const old_top = self.layers.items.len - 1;
            if (level > old_top or level >= plan_max_layers) return .{};

            var cur = self.entry_id;
            var l = old_top;
            while (l > level) : (l -= 1) {
                cur = greedyDescend(&self.layers.items[l], code, cur, &scratch.stats);
            }

            var plan = InsertPlan{ .planned = true, .level = level };
            var connect = level + 1;
            while (connect > 0) {
                connect -= 1;
                const layer = &self.layers.items[connect];
                const found = searchLayer(layer, code, cur, self.ef_construction, scratch);
                plan.counts[connect] = @intCast(selectNeighbors(layer, found, &plan.neighbors[connect]));
                if (found.len > 0) cur = found[0].id;
            }
            return plan;
        }

        /// Applies a plan produced by `planInsert`. The graph may have grown
        /// since planning (neighbor ids stay valid; within-batch nodes are
        /// simply invisible to each other's plans).
        pub fn commitPlanned(self: *Self, allocator: std.mem.Allocator, id: u64, code: BitVec, plan: InsertPlan) !void {
            std.debug.assert(plan.planned and id == self.node_count);
            const fresh = Node{
                .id = id,
                .bit_vector = code,
                .edge_count = 0,
                .neighbors = @splat(0),
            };

            var l: usize = 0;
            while (l <= plan.level) : (l += 1) {
                const layer = &self.layers.items[l];
                try layer.addNode(allocator, fresh);
                const node = layer.nodePtrMut(id);
                node.edge_count = plan.counts[l];
                @memcpy(node.neighbors[0..plan.counts[l]], plan.neighbors[l][0..plan.counts[l]]);
                for (node.neighbors[0..node.edge_count]) |neighbor_id| {
                    linkBack(layer, neighbor_id, id);
                }
            }
            self.node_count += 1;
        }

        /// Draws the level for the next insert (the rng is not thread-safe;
        /// call serially before parallel planning).
        pub fn drawLevel(self: *Self) usize {
            return self.randomLevel();
        }

        /// Deserialization support: appends an empty layer (the first one is
        /// dense, matching ensureLayerCount).
        pub fn appendLayerForLoad(self: *Self, allocator: std.mem.Allocator) !void {
            try self.ensureLayerCount(allocator, self.layers.items.len + 1);
        }

        /// Deserialization support: appends a fully-formed node to a layer.
        /// Nodes must arrive in their original per-layer order.
        pub fn appendNodeForLoad(self: *Self, allocator: std.mem.Allocator, layer_idx: usize, node: Node) !void {
            try self.layers.items[layer_idx].addNode(allocator, node);
        }

        fn randomLevel(self: *Self) usize {
            const inv_log_degree = 1.0 / @log(@as(f64, @floatFromInt(max_edges)));
            const u = 1.0 - self.level_rng.random().float(f64);
            const level: usize = @intFromFloat(-@log(u) * inv_log_degree);
            return @min(level, 31);
        }

        fn ensureLayerCount(self: *Self, allocator: std.mem.Allocator, count: usize) !void {
            while (self.layers.items.len < count) {
                try self.layers.append(allocator, .{ .dense = self.layers.items.len == 0 });
            }
        }

        /// Spec traversal rule: hop to the closest neighbor until no neighbor
        /// improves on the current node.
        fn greedyDescend(layer: *const Layer, query: *const BitVec, start_id: u64, stats: *TraversalStats) u64 {
            var cur_id = start_id;
            var cur_dist = query.distance(layer.bitsPtr(cur_id));
            stats.nodes_evaluated += 1;
            while (true) {
                var improved = false;
                const node = layer.nodePtr(cur_id);
                stats.nodes_evaluated += node.edge_count;
                for (node.neighbors[0..node.edge_count]) |neighbor_id| {
                    const d = query.distance(layer.bitsPtr(neighbor_id));
                    if (d < cur_dist) {
                        cur_dist = d;
                        cur_id = neighbor_id;
                        improved = true;
                    }
                }
                if (!improved) return cur_id;
            }
        }

        fn searchLayer(
            layer: *const Layer,
            query: *const BitVec,
            entry_id: u64,
            ef: usize,
            scratch: *TraversalScratch,
        ) []Candidate {
            scratch.visited.unsetAll();
            var candidates = CandidateMinHeap.fromBuffer(scratch.cand_buf);
            var results = CandidateMaxHeap.fromBuffer(scratch.result_buf[0..ef]);

            const entry_dist = query.distance(layer.bitsPtr(entry_id));
            scratch.stats.nodes_evaluated += 1;
            scratch.visited.set(@intCast(entry_id));
            candidates.push(.{ .dist = entry_dist, .id = entry_id });
            results.push(.{ .dist = entry_dist, .id = entry_id });

            while (candidates.pop()) |current| {
                if (results.len == ef and current.dist > results.peek().?.dist) break;
                const node = layer.nodePtr(current.id);
                // The beam jumps between random nodes; request every
                // neighbor's bit vector before touching the first.
                for (node.neighbors[0..node.edge_count]) |neighbor_id| {
                    @prefetch(layer.bitsPtr(neighbor_id), .{ .rw = .read, .locality = 2 });
                }
                for (node.neighbors[0..node.edge_count]) |neighbor_id| {
                    if (scratch.visited.isSet(@intCast(neighbor_id))) continue;
                    scratch.visited.set(@intCast(neighbor_id));
                    scratch.stats.nodes_evaluated += 1;
                    const d = query.distance(layer.bitsPtr(neighbor_id));
                    if (results.len < ef) {
                        results.push(.{ .dist = d, .id = neighbor_id });
                        candidates.push(.{ .dist = d, .id = neighbor_id });
                    } else if (d < results.peek().?.dist) {
                        _ = results.pop();
                        results.push(.{ .dist = d, .id = neighbor_id });
                        candidates.push(.{ .dist = d, .id = neighbor_id });
                    }
                }
            }

            const found = results.items[0..results.len];
            std.mem.sort(Candidate, found, {}, struct {
                fn lessThan(_: void, a: Candidate, b: Candidate) bool {
                    return a.dist < b.dist;
                }
            }.lessThan);
            return found;
        }

        /// HNSW neighbor-selection heuristic with backfill. A candidate is
        /// rejected when an already-selected neighbor is strictly closer to
        /// it than the base node is, or when it exactly duplicates a selected
        /// neighbor's bit pattern (a zero-information edge). Remaining slots
        /// are backfilled with the closest skipped candidates, so diverse
        /// long-range links are claimed before near-duplicates flood the
        /// list. `candidates` must be sorted by ascending distance to base.
        fn selectNeighbors(layer: *const Layer, candidates: []const Candidate, out: *[max_edges]u64) usize {
            var count: usize = 0;
            for (candidates) |c| {
                if (count == max_edges) return count;
                const c_bits = layer.bitsPtr(c.id);
                var diverse = true;
                for (out[0..count]) |selected_id| {
                    const d = c_bits.distance(layer.bitsPtr(selected_id));
                    if (d < c.dist or d == 0) {
                        diverse = false;
                        break;
                    }
                }
                if (diverse) {
                    out[count] = c.id;
                    count += 1;
                }
            }
            for (candidates) |c| {
                if (count == max_edges) break;
                if (std.mem.indexOfScalar(u64, out[0..count], c.id) == null) {
                    out[count] = c.id;
                    count += 1;
                }
            }
            return count;
        }

        /// Adds a reverse edge; when the neighbor list is full, it is rebuilt
        /// from the existing edges plus the new one via the same diversity
        /// heuristic, so long-range links survive duplicate floods.
        fn linkBack(layer: *Layer, from_id: u64, to_id: u64) void {
            const from = layer.nodePtrMut(from_id);
            if (from.edge_count < max_edges) {
                from.neighbors[from.edge_count] = to_id;
                from.edge_count += 1;
                return;
            }

            var pool: [max_edges + 1]Candidate = undefined;
            for (from.neighbors, 0..) |neighbor_id, i| {
                pool[i] = .{
                    .dist = from.bit_vector.distance(layer.bitsPtr(neighbor_id)),
                    .id = neighbor_id,
                };
            }
            pool[max_edges] = .{
                .dist = from.bit_vector.distance(layer.bitsPtr(to_id)),
                .id = to_id,
            };
            std.mem.sort(Candidate, &pool, {}, struct {
                fn lessThan(_: void, a: Candidate, b: Candidate) bool {
                    return a.dist < b.dist;
                }
            }.lessThan);

            var rebuilt: [max_edges]u64 = undefined;
            const count = selectNeighbors(layer, &pool, &rebuilt);
            from.neighbors = rebuilt;
            from.edge_count = @intCast(count);
        }
    };
}

test "single node routes to itself" {
    const Graph = RoutingGraph(64, 4);
    var graph = Graph.init(16);
    defer graph.deinit(std.testing.allocator);

    var coords: [64]f32 = @splat(1.0);
    const bits = Graph.BitVec.fromF32(&coords);
    try graph.insert(std.testing.allocator, 0, bits);

    var scratch = TraversalScratch{};
    defer scratch.deinit(std.testing.allocator);
    try scratch.ensureCapacity(std.testing.allocator, graph.node_count, 4);

    const found = graph.route(&bits, 4, &scratch);
    try std.testing.expectEqual(@as(usize, 1), found.len);
    try std.testing.expectEqual(@as(u64, 0), found[0].id);
    try std.testing.expectEqual(@as(u32, 0), found[0].dist);
}

test "routing finds exact bit-pattern matches" {
    const dim = 64;
    const Graph = RoutingGraph(dim, 8);
    var graph = Graph.init(32);
    defer graph.deinit(std.testing.allocator);

    var prng = std.Random.DefaultPrng.init(3);
    const rand = prng.random();

    var patterns: [200]Graph.BitVec = undefined;
    for (&patterns, 0..) |*bits, i| {
        var coords: [dim]f32 = undefined;
        for (&coords) |*c| c.* = rand.floatNorm(f32);
        bits.* = Graph.BitVec.fromF32(&coords);
        try graph.insert(std.testing.allocator, i, bits.*);
    }

    var scratch = TraversalScratch{};
    defer scratch.deinit(std.testing.allocator);
    try scratch.ensureCapacity(std.testing.allocator, graph.node_count, 32);

    // Querying with stored patterns must surface the node itself.
    var hits: usize = 0;
    for (patterns, 0..) |bits, i| {
        const found = graph.route(&bits, 32, &scratch);
        for (found) |c| {
            if (c.id == i) {
                try std.testing.expectEqual(@as(u32, 0), c.dist);
                hits += 1;
                break;
            }
        }
    }
    try std.testing.expect(hits >= 190);
}
