//! 1-bit sign quantization: each coordinate collapses to its sign bit, packed
//! into u64 words so distance reduces to XOR + native POPCOUNT.

const std = @import("std");

/// Hamming distance between two bit-packed vectors of `words` u64 words,
/// evaluated as one vector XOR + per-lane POPCOUNT + horizontal add.
pub inline fn hammingDistance(comptime words: usize, a: []const u64, b: []const u64) u32 {
    std.debug.assert(a.len >= words and b.len >= words);
    const va: @Vector(words, u64) = a[0..words].*;
    const vb: @Vector(words, u64) = b[0..words].*;
    const counts: @Vector(words, u32) = @intCast(@popCount(va ^ vb));
    return @reduce(.Add, counts);
}

/// A bit-packed 1-bit vector used for fast graph routing.
pub fn BitVector(comptime dim: usize) type {
    comptime std.debug.assert(dim > 0);
    const word_count = (dim + 63) / 64;
    return struct {
        const Self = @This();
        pub const words_count = word_count;

        words: [word_count]u64,

        pub const zero = Self{ .words = @splat(0) };

        /// Maps an FP32 coordinate array to its sign profile (> 0.0 -> 1).
        pub fn fromF32(coords: []const f32) Self {
            std.debug.assert(coords.len == dim);
            var self = zero;
            for (coords, 0..) |c, i| {
                if (c > 0.0) {
                    self.words[i / 64] |= @as(u64, 1) << @intCast(i % 64);
                }
            }
            return self;
        }

        pub fn distance(self: *const Self, other: *const Self) u32 {
            return hammingDistance(word_count, &self.words, &other.words);
        }
    };
}

test "sign packing maps positives to set bits" {
    const Bits = BitVector(70);
    var coords: [70]f32 = @splat(-1.0);
    coords[0] = 0.5;
    coords[63] = 2.0;
    coords[64] = 0.0; // <= 0.0 stays unset
    coords[69] = 1e-9;

    const bits = Bits.fromF32(&coords);
    try std.testing.expectEqual(@as(u64, 1) | (@as(u64, 1) << 63), bits.words[0]);
    try std.testing.expectEqual(@as(u64, 1) << 5, bits.words[1]);
}

test "hamming distance counts differing signs" {
    const Bits = BitVector(128);
    var a_coords: [128]f32 = @splat(1.0);
    var b_coords: [128]f32 = @splat(1.0);
    for (0..17) |i| b_coords[i * 7] = -1.0;

    const a = Bits.fromF32(&a_coords);
    const b = Bits.fromF32(&b_coords);
    try std.testing.expectEqual(@as(u32, 17), a.distance(&b));
    try std.testing.expectEqual(@as(u32, 0), a.distance(&a));
}
