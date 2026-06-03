//! Milestone 4 verification suite.

const std = @import("std");
const index_mod = @import("index.zig");

test "10k vectors: queries allocate exactly zero bytes" {
    const dim = 64;
    const Idx = index_mod.Index(dim, 8);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var index = try Idx.init(allocator, 32, 2026);
    var prng = std.Random.DefaultPrng.init(606);
    const rand = prng.random();

    var coords: [dim]f32 = undefined;
    for (0..10_000) |i| {
        for (&coords) |*c| c.* = rand.floatNorm(f32);
        try index.add(allocator, i, &coords);
    }

    var ctx = try Idx.SearchContext.init(allocator, &index, 64);

    // Everything the queries touch is preallocated above; if any search
    // path allocated, the arena would have to grow.
    const capacity_before = arena.queryCapacity();

    var out: [10]index_mod.SearchResult = undefined;
    for (0..100) |_| {
        for (&coords) |*c| c.* = rand.floatNorm(f32);
        const count = index.search(&ctx, &coords, &out);
        try std.testing.expectEqual(@as(usize, 10), count);
    }

    try std.testing.expectEqual(capacity_before, arena.queryCapacity());
}
