//! C-compatible API over a concrete Index instantiation.
//!
//! The dimension and graph degree are fixed at compile time via the
//! `-Dc-dim` / `-Dc-max-edges` build options (defaults: 1536 / 32).
//! See include/quantajump.h for the matching prototypes.

const std = @import("std");
const build_options = @import("build_options");
const index_mod = @import("index.zig");
const storage = @import("storage.zig");

pub const c_dim = build_options.c_dim;
pub const c_max_edges = build_options.c_max_edges;

const CIndex = index_mod.Index(c_dim, c_max_edges);
const allocator = std.heap.smp_allocator;

const max_k = 256;

pub const Handle = struct {
    index: CIndex,
};

pub const Context = CIndex.SearchContext;

export fn qj_dim() usize {
    return c_dim;
}

export fn qj_index_create(ef_construction: usize, seed: u64) ?*Handle {
    const handle = allocator.create(Handle) catch return null;
    handle.index = CIndex.init(allocator, ef_construction, seed) catch {
        allocator.destroy(handle);
        return null;
    };
    return handle;
}

export fn qj_index_destroy(handle: ?*Handle) void {
    const h = handle orelse return;
    h.index.deinit(allocator);
    allocator.destroy(h);
}

export fn qj_index_add(handle: *Handle, id: u64, coords: [*]const f32) i32 {
    handle.index.add(allocator, id, coords[0..c_dim]) catch return -1;
    return 0;
}

export fn qj_index_len(handle: *const Handle) usize {
    return handle.index.len();
}

/// Multi-threaded bulk ingest of n vectors (row-major coords, n*dim floats).
export fn qj_index_add_batch(handle: *Handle, ids: [*]const u64, coords: [*]const f32, n: usize, threads: usize) i32 {
    handle.index.addBatch(allocator, ids[0..n], coords[0 .. n * c_dim], threads) catch return -1;
    return 0;
}

/// Tombstones a vector by id. Returns 0 on success, -1 when unknown.
export fn qj_index_remove(handle: *Handle, id: u64) i32 {
    const removed = handle.index.remove(allocator, id) catch return -1;
    return if (removed) 0 else -1;
}

export fn qj_index_save(handle: *const Handle, path: [*:0]const u8) i32 {
    const empty_labels = allocator.alloc([]const u8, handle.index.capacity()) catch return -1;
    defer allocator.free(empty_labels);
    @memset(empty_labels, "");
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    storage.save(CIndex, allocator, threaded.io(), std.mem.span(path), &handle.index, empty_labels) catch return -1;
    return 0;
}

export fn qj_index_load(path: [*:0]const u8) ?*Handle {
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    var loaded = storage.load(CIndex, allocator, threaded.io(), std.mem.span(path)) catch return null;
    allocator.free(loaded.labels);
    allocator.free(loaded.label_blob);
    const handle = allocator.create(Handle) catch {
        loaded.index.deinit(allocator);
        return null;
    };
    handle.* = .{ .index = loaded.index };
    return handle;
}

/// Creates a search context sized for the index's current contents.
/// Recreate it after further inserts. `m` is the stage-1 candidate count.
export fn qj_context_create(handle: *const Handle, m: usize) ?*Context {
    if (m == 0) return null;
    const ctx = allocator.create(Context) catch return null;
    ctx.* = Context.init(allocator, &handle.index, m) catch {
        allocator.destroy(ctx);
        return null;
    };
    return ctx;
}

export fn qj_context_destroy(ctx: ?*Context) void {
    const c = ctx orelse return;
    c.deinit(allocator);
    allocator.destroy(c);
}

/// Writes up to k (capped at 256) results into out_ids/out_scores, sorted by
/// descending score. Returns the result count. Allocation-free.
export fn qj_search(
    handle: *const Handle,
    ctx: *Context,
    query: [*]const f32,
    k: usize,
    out_ids: [*]u64,
    out_scores: [*]f32,
) usize {
    var results: [max_k]index_mod.SearchResult = undefined;
    const capped = @min(k, max_k);
    const count = handle.index.search(ctx, query[0..c_dim], results[0..capped]);
    for (results[0..count], 0..) |result, i| {
        out_ids[i] = result.id;
        out_scores[i] = result.score;
    }
    return count;
}

/// Restricts results to `allowed` (user ids); see Index.searchFiltered.
export fn qj_search_filtered(
    handle: *const Handle,
    ctx: *Context,
    query: [*]const f32,
    allowed: [*]const u64,
    allowed_len: usize,
    k: usize,
    out_ids: [*]u64,
    out_scores: [*]f32,
) usize {
    var results: [max_k]index_mod.SearchResult = undefined;
    const capped = @min(k, max_k);
    const count = handle.index.searchFiltered(ctx, query[0..c_dim], allowed[0..allowed_len], results[0..capped]);
    for (results[0..count], 0..) |result, i| {
        out_ids[i] = result.id;
        out_scores[i] = result.score;
    }
    return count;
}

/// Multi-threaded batch search over n queries (row-major, n*dim floats).
/// out_ids/out_scores hold n*k slots; out_counts n entries.
export fn qj_search_batch(
    handle: *const Handle,
    queries: [*]const f32,
    n: usize,
    k: usize,
    m: usize,
    threads: usize,
    out_ids: [*]u64,
    out_scores: [*]f32,
    out_counts: [*]usize,
) i32 {
    const results = allocator.alloc(index_mod.SearchResult, n * k) catch return -1;
    defer allocator.free(results);
    handle.index.searchBatch(
        allocator,
        queries[0 .. n * c_dim],
        k,
        m,
        threads,
        true,
        results,
        out_counts[0..n],
    ) catch return -1;
    for (0..n) |q| {
        for (results[q * k ..][0..out_counts[q]], 0..) |result, i| {
            out_ids[q * k + i] = result.id;
            out_scores[q * k + i] = result.score;
        }
    }
    return 0;
}

test "C API roundtrip" {
    const handle = qj_index_create(2 * c_max_edges, 31).?;
    defer qj_index_destroy(handle);

    var prng = std.Random.DefaultPrng.init(8);
    const rand = prng.random();
    var coords: [c_dim]f32 = undefined;

    for (0..50) |i| {
        for (&coords) |*c| c.* = rand.floatNorm(f32);
        try std.testing.expectEqual(@as(i32, 0), qj_index_add(handle, 100 + i, &coords));
    }
    try std.testing.expectEqual(@as(usize, 50), qj_index_len(handle));
    try std.testing.expectEqual(c_dim, qj_dim());

    const ctx = qj_context_create(handle, 32).?;
    defer qj_context_destroy(ctx);

    var ids: [10]u64 = undefined;
    var scores: [10]f32 = undefined;
    for (&coords) |*c| c.* = rand.floatNorm(f32);
    const count = qj_search(handle, ctx, &coords, 10, &ids, &scores);
    try std.testing.expectEqual(@as(usize, 10), count);
    for (ids) |id| {
        try std.testing.expect(id >= 100 and id < 150);
    }
    for (scores[0..9], scores[1..10]) |a, b| {
        try std.testing.expect(a >= b);
    }
}
