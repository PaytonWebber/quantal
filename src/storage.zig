//! .tq index file format: a serialized Index plus one text label per vector.
//!
//! Little-endian throughout. Layout:
//!   magic "TQX4"
//!   u32 dim, u32 max_edges, u64 vector_count
//!   u64 ef_construction, u64 entry_id, u32 layer_count, u8 rerank_store
//!   u8 calib_frozen; when set: dim f32 shifts, dim f32 scales (TQ+)
//!   rotation matrix: dim*dim f32
//!   payloads, each: u64 id, chunks_count * 3 bytes (true 3-byte TQ3 packing),
//!                   f32 bias_scale, f32 bias_shift, f32 renorm_scalar
//!   rerank store (rotated space):
//!     fp32: vector_count * dim f32
//!     sq8:  vector_count * dim i8, then vector_count f32 scales
//!   layers, each: u64 node_count, then nodes:
//!                   u64 id, words_count * u64 bit vector,
//!                   u32 edge_count, edge_count * u64 neighbors
//!   tombstones: u64 count, then count u64 internal indices
//!   labels, each: u16 length + bytes

const std = @import("std");
const index_type = @import("index.zig");

const magic = "TQX4";

pub const Header = struct {
    dim: u32,
    max_edges: u32,
    vector_count: u64,
};

pub fn LoadedIndex(comptime Idx: type) type {
    return struct {
        index: Idx,
        /// Slices into label_blob, one per vector.
        labels: [][]const u8,
        label_blob: []u8,

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            self.index.deinit(allocator);
            allocator.free(self.labels);
            allocator.free(self.label_blob);
            self.* = undefined;
        }
    };
}

/// Reads just the header so the caller can dispatch to the right comptime
/// instantiation before parsing the rest.
pub fn readHeader(io: std.Io, path: []const u8) !Header {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var buf: [20]u8 = undefined;
    const got = try file.readPositionalAll(io, &buf, 0);
    if (got != buf.len or !std.mem.eql(u8, buf[0..4], magic)) return error.NotATqIndex;
    return .{
        .dim = std.mem.readInt(u32, buf[4..8], .little),
        .max_edges = std.mem.readInt(u32, buf[8..12], .little),
        .vector_count = std.mem.readInt(u64, buf[12..20], .little),
    };
}

pub fn save(
    comptime Idx: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    index: *const Idx,
    labels: []const []const u8,
) !void {
    const bytes = try serialize(Idx, allocator, index, labels);
    defer allocator.free(bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

pub fn load(
    comptime Idx: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !LoadedIndex(Idx) {
    const file_bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(file_bytes);
    return deserialize(Idx, allocator, file_bytes);
}

pub fn serialize(
    comptime Idx: type,
    allocator: std.mem.Allocator,
    index: *const Idx,
    labels: []const []const u8,
) ![]u8 {
    const dim = Idx.dimension;
    std.debug.assert(labels.len == index.capacity());

    var bytes: std.ArrayList(u8) = .empty;
    errdefer bytes.deinit(allocator);
    const w = Writer{ .bytes = &bytes, .allocator = allocator };

    try w.raw(magic);
    try w.int(u32, dim);
    try w.int(u32, maxEdgesOf(Idx));
    try w.int(u64, index.capacity());
    try w.int(u64, index.routing.ef_construction);
    try w.int(u64, index.routing.entry_id);
    try w.int(u32, @intCast(index.routing.layers.items.len));
    try w.int(u8, if (index.hasRerankStore()) @intFromEnum(index.rerank_store) else 0);

    std.debug.assert(index.pending_ids.items.len == 0); // freeze() before save
    try w.int(u8, @intFromBool(index.calib_frozen));
    if (index.calib_frozen) {
        try w.raw(std.mem.sliceAsBytes(index.calib_shift));
        try w.raw(std.mem.sliceAsBytes(index.calib_scale));
    }

    try w.raw(std.mem.sliceAsBytes(index.rotation.rows));

    for (index.payloads.items) |payload| {
        try w.int(u64, payload.id);
        for (payload.chunks) |chunk| {
            var packed_bytes: [3]u8 = undefined;
            std.mem.writeInt(u24, &packed_bytes, chunk.values, .little);
            try w.raw(&packed_bytes);
        }
        try w.float(payload.bias_scale);
        try w.float(payload.bias_shift);
        try w.float(payload.renorm_scalar);
    }

    if (index.hasRerankStore()) {
        switch (index.rerank_store) {
            .none => {},
            .fp32 => try w.raw(std.mem.sliceAsBytes(index.originals.items)),
            .sq8 => {
                try w.raw(std.mem.sliceAsBytes(index.sq8_codes.items));
                try w.raw(std.mem.sliceAsBytes(index.sq8_scales.items));
            },
        }
    }

    for (index.routing.layers.items) |layer| {
        try w.int(u64, layer.nodes.items.len);
        for (layer.nodes.items) |node| {
            try w.int(u64, node.id);
            try w.raw(std.mem.sliceAsBytes(&node.bit_vector.words));
            try w.int(u32, node.edge_count);
            for (node.neighbors[0..node.edge_count]) |neighbor_id| {
                try w.int(u64, neighbor_id);
            }
        }
    }

    var deleted_count: u64 = 0;
    if (index.tombstones.bit_length > 0) deleted_count = index.tombstones.count();
    try w.int(u64, deleted_count);
    if (deleted_count > 0) {
        var it = index.tombstones.iterator(.{});
        while (it.next()) |internal| try w.int(u64, internal);
    }

    for (labels) |label| {
        try w.int(u16, @intCast(label.len));
        try w.raw(label);
    }

    return bytes.toOwnedSlice(allocator);
}

pub fn deserialize(
    comptime Idx: type,
    allocator: std.mem.Allocator,
    file_bytes: []const u8,
) !LoadedIndex(Idx) {
    const dim = Idx.dimension;
    var r = Reader{ .bytes = file_bytes };

    if (!std.mem.eql(u8, try r.raw(4), magic)) return error.NotATqIndex;
    if (try r.int(u32) != dim) return error.DimensionMismatch;
    if (try r.int(u32) != maxEdgesOf(Idx)) return error.MaxEdgesMismatch;
    const vector_count_u64 = try r.int(u64);
    const ef_construction = try r.int(u64);
    const entry_id = try r.int(u64);
    const layer_count = try r.int(u32);
    const rerank_store = std.enums.fromInt(index_type.RerankStore, try r.int(u8)) orelse {
        return error.CorruptIndex;
    };

    // Each vector contributes at least a u64 id + its chunks to the payload
    // section, so it can never exceed file size / 8. This caps every
    // subsequent vector_count-sized allocation at ~the file size.
    const vector_count: usize = try r.boundedCount(vector_count_u64, 8);
    if (vector_count > 0 and entry_id >= vector_count_u64) return error.CorruptIndex;

    const calib_frozen = (try r.int(u8)) != 0;
    var calib_shift: []f32 = &.{};
    var calib_scale: []f32 = &.{};
    errdefer allocator.free(calib_shift);
    errdefer allocator.free(calib_scale);
    if (calib_frozen) {
        calib_shift = try allocator.alloc(f32, dim);
        @memcpy(std.mem.sliceAsBytes(calib_shift), try r.raw(dim * 4));
        calib_scale = try allocator.alloc(f32, dim);
        @memcpy(std.mem.sliceAsBytes(calib_scale), try r.raw(dim * 4));
    }

    const rotation_rows = try allocator.alloc(f32, dim * dim);
    errdefer allocator.free(rotation_rows);
    @memcpy(std.mem.sliceAsBytes(rotation_rows), try r.raw(dim * dim * 4));

    var index = Idx{
        .rotation = .{ .rows = rotation_rows },
        .routing = Idx.Graph.init(ef_construction),
        .rotate_buf = try allocator.alloc(f32, dim),
        .calib_buf = try allocator.alloc(f32, dim),
        .tq_plus = calib_frozen,
        .calib_frozen = calib_frozen,
        .calib_shift = calib_shift,
        .calib_scale = calib_scale,
    };
    errdefer index.payloads.deinit(allocator);
    errdefer index.routing.deinit(allocator);
    errdefer index.originals.deinit(allocator);
    errdefer index.sq8_codes.deinit(allocator);
    errdefer index.sq8_scales.deinit(allocator);
    errdefer index.id_to_internal.deinit(allocator);
    errdefer index.tombstones.deinit(allocator);
    errdefer allocator.free(index.rotate_buf);
    errdefer allocator.free(index.calib_buf);

    try index.payloads.ensureTotalCapacityPrecise(allocator, vector_count);
    for (0..vector_count) |_| {
        var payload: Idx.Payload = undefined;
        payload.id = try r.int(u64);
        for (&payload.chunks) |*chunk| {
            chunk.values = std.mem.readInt(u24, (try r.raw(3))[0..3], .little);
        }
        payload.bias_scale = try r.float();
        payload.bias_shift = try r.float();
        payload.renorm_scalar = try r.float();
        index.payloads.appendAssumeCapacity(payload);
    }

    index.rerank_store = rerank_store;
    // vector_count is already bounded by file size, so this cannot overflow.
    const coord_count: usize = std.math.mul(usize, vector_count, dim) catch return error.CorruptIndex;
    switch (rerank_store) {
        .none => {},
        .fp32 => {
            try index.originals.ensureTotalCapacityPrecise(allocator, coord_count);
            index.originals.items.len = coord_count;
            @memcpy(std.mem.sliceAsBytes(index.originals.items), try r.raw(coord_count * 4));
        },
        .sq8 => {
            try index.sq8_codes.ensureTotalCapacityPrecise(allocator, coord_count);
            index.sq8_codes.items.len = coord_count;
            @memcpy(std.mem.sliceAsBytes(index.sq8_codes.items), try r.raw(coord_count));
            try index.sq8_scales.ensureTotalCapacityPrecise(allocator, vector_count);
            index.sq8_scales.items.len = vector_count;
            @memcpy(std.mem.sliceAsBytes(index.sq8_scales.items), try r.raw(vector_count * 4));
        },
    }

    for (0..layer_count) |layer_idx| {
        try index.routing.appendLayerForLoad(allocator);
        // A node consumes at least id(8) + bit vector + edge_count(4) bytes.
        const min_node_bytes = 8 + @sizeOf(@TypeOf(@as(Idx.Graph.Node, undefined).bit_vector.words)) + 4;
        const node_count = try r.boundedCount(try r.int(u64), min_node_bytes);
        for (0..node_count) |_| {
            var node: Idx.Graph.Node = undefined;
            node.id = try r.int(u64);
            // Ids index the dense base layer and the payload array directly;
            // an out-of-range id would become an OOB access at query time.
            if (node.id >= vector_count) return error.CorruptIndex;
            @memcpy(std.mem.sliceAsBytes(&node.bit_vector.words), try r.raw(@sizeOf(@TypeOf(node.bit_vector.words))));
            node.edge_count = try r.int(u32);
            if (node.edge_count > node.neighbors.len) return error.CorruptIndex;
            node.neighbors = @splat(0);
            for (node.neighbors[0..node.edge_count]) |*neighbor_id| {
                neighbor_id.* = try r.int(u64);
                if (neighbor_id.* >= vector_count) return error.CorruptIndex;
            }
            try index.routing.appendNodeForLoad(allocator, layer_idx, node);
        }
    }
    index.routing.entry_id = entry_id;
    index.routing.node_count = vector_count;

    const deleted_count = try r.int(u64);
    if (deleted_count > 0) {
        if (deleted_count > vector_count_u64) return error.CorruptIndex;
        try index.tombstones.resize(allocator, vector_count, false);
        for (0..@intCast(deleted_count)) |_| {
            const internal = try r.int(u64);
            if (internal >= vector_count_u64) return error.CorruptIndex;
            index.tombstones.set(@intCast(internal));
        }
    }
    try index.id_to_internal.ensureTotalCapacity(allocator, @intCast(vector_count));
    for (index.payloads.items, 0..) |payload, internal| {
        if (index.isDeleted(internal)) continue;
        index.id_to_internal.putAssumeCapacity(payload.id, @intCast(internal));
        index.live_count += 1;
    }

    const labels = try allocator.alloc([]const u8, @intCast(vector_count));
    errdefer allocator.free(labels);
    var blob: std.ArrayList(u8) = .empty;
    errdefer blob.deinit(allocator);
    var offsets = try allocator.alloc(usize, labels.len + 1);
    defer allocator.free(offsets);
    for (0..labels.len) |i| {
        offsets[i] = blob.items.len;
        const len = try r.int(u16);
        try blob.appendSlice(allocator, try r.raw(len));
    }
    offsets[labels.len] = blob.items.len;
    const label_blob = try blob.toOwnedSlice(allocator);
    for (labels, 0..) |*label, i| {
        label.* = label_blob[offsets[i]..offsets[i + 1]];
    }

    return .{ .index = index, .labels = labels, .label_blob = label_blob };
}

fn maxEdgesOf(comptime Idx: type) u32 {
    return @typeInfo(@FieldType(Idx.Graph.Node, "neighbors")).array.len;
}

const Writer = struct {
    bytes: *std.ArrayList(u8),
    allocator: std.mem.Allocator,

    fn raw(w: Writer, data: []const u8) !void {
        try w.bytes.appendSlice(w.allocator, data);
    }

    fn int(w: Writer, comptime T: type, value: T) !void {
        var buf: [@divExact(@typeInfo(T).int.bits, 8)]u8 = undefined;
        std.mem.writeInt(T, &buf, value, .little);
        try w.raw(&buf);
    }

    fn float(w: Writer, value: f32) !void {
        try w.int(u32, @bitCast(value));
    }
};

const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn remaining(r: *const Reader) usize {
        return r.bytes.len - r.pos;
    }

    fn raw(r: *Reader, len: usize) ![]const u8 {
        // Subtraction form avoids the r.pos + len overflow a malformed
        // length could trigger.
        if (len > r.remaining()) return error.CorruptIndex;
        defer r.pos += len;
        return r.bytes[r.pos..][0..len];
    }

    /// Rejects a count that cannot possibly be backed by the remaining
    /// bytes (each item consumes at least `min_item_bytes`), so the file
    /// can never drive a speculative allocation larger than itself.
    fn boundedCount(r: *const Reader, count: u64, min_item_bytes: usize) !usize {
        if (count > r.remaining() / @max(min_item_bytes, 1)) return error.CorruptIndex;
        return @intCast(count);
    }

    fn int(r: *Reader, comptime T: type) !T {
        const size = @divExact(@typeInfo(T).int.bits, 8);
        const slice = try r.raw(size);
        return std.mem.readInt(T, slice[0..size], .little);
    }

    fn float(r: *Reader) !f32 {
        return @bitCast(try r.int(u32));
    }
};

test "serialize/deserialize roundtrip preserves search results" {
    const index_mod = @import("index.zig");
    const heap_mod = @import("heap.zig");
    const dim = 32;
    const Idx = index_mod.Index(dim, 4);
    const allocator = std.testing.allocator;

    var original = try Idx.init(allocator, 16, 7);
    defer original.deinit(allocator);
    original.tq_plus = true; // also exercises calibration serialization

    var prng = std.Random.DefaultPrng.init(99);
    const rand = prng.random();
    var label_buf: [50][8]u8 = undefined;
    var labels: [50][]const u8 = undefined;
    for (0..50) |i| {
        var coords: [dim]f32 = undefined;
        for (&coords) |*c| c.* = rand.floatNorm(f32);
        try original.add(allocator, i, &coords);
        labels[i] = std.fmt.bufPrint(&label_buf[i], "word{d}", .{i}) catch unreachable;
    }
    try original.freeze(allocator);
    try std.testing.expect(try original.remove(allocator, 13));

    const bytes = try serialize(Idx, allocator, &original, &labels);
    defer allocator.free(bytes);
    var loaded = try deserialize(Idx, allocator, bytes);
    defer loaded.deinit(allocator);

    try std.testing.expectEqual(original.len(), loaded.index.len());
    try std.testing.expect(loaded.index.isDeleted(13));
    try std.testing.expectEqualStrings("word7", loaded.labels[7]);

    // Identical query against both indexes must return identical results.
    var ctx_a = try Idx.SearchContext.init(allocator, &original, 16);
    defer ctx_a.deinit(allocator);
    var ctx_b = try Idx.SearchContext.init(allocator, &loaded.index, 16);
    defer ctx_b.deinit(allocator);

    var query: [dim]f32 = undefined;
    for (&query) |*q| q.* = rand.floatNorm(f32);
    var out_a: [5]heap_mod.SearchResult = undefined;
    var out_b: [5]heap_mod.SearchResult = undefined;
    const n_a = original.search(&ctx_a, &query, &out_a);
    const n_b = loaded.index.search(&ctx_b, &query, &out_b);

    try std.testing.expectEqual(n_a, n_b);
    for (out_a[0..n_a], out_b[0..n_b]) |a, b| {
        try std.testing.expectEqual(a.id, b.id);
        try std.testing.expectEqual(a.score, b.score);
    }
}

test "deserialize rejects malformed input without crashing" {
    const index_mod = @import("index.zig");
    const dim = 16;
    const Idx = index_mod.Index(dim, 4);
    const allocator = std.testing.allocator;

    // A small valid index to corrupt.
    var original = try Idx.init(allocator, 8, 1);
    defer original.deinit(allocator);
    var labels: [12][]const u8 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    const rand = prng.random();
    for (0..12) |i| {
        var coords: [dim]f32 = undefined;
        for (&coords) |*c| c.* = rand.floatNorm(f32);
        try original.add(allocator, i, &coords);
        labels[i] = "x";
    }
    const valid = try serialize(Idx, allocator, &original, &labels);
    defer allocator.free(valid);

    // Sanity: the untouched bytes load.
    {
        var ok = try deserialize(Idx, allocator, valid);
        ok.deinit(allocator);
    }

    // Helper: a corruption must surface as an error, never UB. With the
    // testing allocator any leaked allocation on the error path also fails.
    const expectRejected = struct {
        fn run(a: std.mem.Allocator, bytes: []const u8) !void {
            if (deserialize(Idx, a, bytes)) |*loaded| {
                @constCast(loaded).deinit(a);
                return error.ShouldHaveRejected;
            } else |_| {}
        }
    }.run;

    // Bad magic.
    {
        const b = try allocator.dupe(u8, valid);
        defer allocator.free(b);
        b[0] = 'Z';
        try std.testing.expectError(error.NotATqIndex, deserialize(Idx, allocator, b));
    }
    // Truncation at every prefix length must be rejected (never read OOB).
    {
        var cut: usize = 0;
        while (cut < valid.len) : (cut += 1) try expectRejected(allocator, valid[0..cut]);
    }
    // Absurd vector_count (offset 12, u64) -> bounded-count rejection, no
    // multi-GB allocation attempt.
    {
        const b = try allocator.dupe(u8, valid);
        defer allocator.free(b);
        std.mem.writeInt(u64, b[12..20], std.math.maxInt(u64), .little);
        try expectRejected(allocator, b);
    }
    // vector_count * dim overflow.
    {
        const b = try allocator.dupe(u8, valid);
        defer allocator.free(b);
        std.mem.writeInt(u64, b[12..20], std.math.maxInt(u64) / dim + 1, .little);
        try expectRejected(allocator, b);
    }
    // entry_id out of range (offset 28, u64).
    {
        const b = try allocator.dupe(u8, valid);
        defer allocator.free(b);
        std.mem.writeInt(u64, b[28..36], 99999, .little);
        try expectRejected(allocator, b);
    }
    // Every single-byte flip must load-or-reject, never crash.
    {
        var i: usize = 0;
        while (i < valid.len) : (i += 1) {
            const b = try allocator.dupe(u8, valid);
            defer allocator.free(b);
            b[i] ^= 0xFF;
            if (deserialize(Idx, allocator, b)) |*loaded| {
                @constCast(loaded).deinit(allocator);
            } else |_| {}
        }
    }
}
