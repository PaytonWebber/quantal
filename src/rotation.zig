//! Random rotation preconditioning (TurboQuant Algorithm 1, line 2).
//!
//! A fixed orthonormal matrix, generated once from a seed by orthonormalizing
//! Gaussian rows with modified Gram-Schmidt, is applied to every vector at
//! ingest and query time. Rotation concentrates coordinates toward N(0, 1/d)
//! so per-coordinate scalar quantization is near-optimal, and it preserves
//! inner products, so all scoring stays in rotated space.

const std = @import("std");

pub fn RandomRotation(comptime dim: usize) type {
    comptime std.debug.assert(dim > 0);
    return struct {
        const Self = @This();

        /// Row-major dim x dim orthonormal matrix.
        rows: []f32,

        pub fn init(allocator: std.mem.Allocator, seed: u64) !Self {
            const rows = try allocator.alloc(f32, dim * dim);
            errdefer allocator.free(rows);

            var prng = std.Random.DefaultPrng.init(seed);
            const rand = prng.random();

            for (0..dim) |i| {
                const row = rows[i * dim ..][0..dim];
                // Re-draw until the row keeps a usable component orthogonal
                // to the rows above it (degenerate draws are vanishingly
                // rare, but f32 cancellation makes the guard cheap insurance).
                while (true) {
                    for (row) |*v| v.* = rand.floatNorm(f32);
                    for (0..i) |k| {
                        const prev = rows[k * dim ..][0..dim];
                        const proj = dot(row, prev);
                        for (row, prev) |*v, p| v.* -= proj * p;
                    }
                    const norm = @sqrt(dot(row, row));
                    if (norm > 1e-6) {
                        for (row) |*v| v.* /= norm;
                        break;
                    }
                }
            }
            return .{ .rows = rows };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.rows);
            self.* = undefined;
        }

        /// out = PI * x
        pub fn apply(self: *const Self, x: []const f32, out: []f32) void {
            std.debug.assert(x.len == dim and out.len == dim);
            for (out, 0..) |*o, i| {
                o.* = dot(self.rows[i * dim ..][0..dim], x);
            }
        }

    };
}

/// A fixed `out_bits` x `in_dim` matrix of i.i.d. N(0,1) entries — a random
/// linear projection. Used for SimHash routing codes: sign(R·x) over many
/// rows estimates the angular distance between inputs, with the estimate
/// sharpening as `out_bits` grows (variance ~1/out_bits). Unlike the
/// orthonormal rotation, the rows are independent, not orthogonalized.
pub fn RandomProjection(comptime out_bits: usize, comptime in_dim: usize) type {
    comptime std.debug.assert(out_bits > 0 and in_dim > 0);
    return struct {
        const Self = @This();
        pub const rows_count = out_bits;
        pub const cols = in_dim;

        /// Row-major out_bits x in_dim.
        rows: []f32,

        pub fn init(allocator: std.mem.Allocator, seed: u64) !Self {
            const rows = try allocator.alloc(f32, out_bits * in_dim);
            var prng = std.Random.DefaultPrng.init(seed);
            const rand = prng.random();
            for (rows) |*v| v.* = rand.floatNorm(f32);
            return .{ .rows = rows };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.rows);
            self.* = undefined;
        }

        /// out = R · x
        pub fn project(self: *const Self, x: []const f32, out: []f32) void {
            std.debug.assert(x.len == in_dim and out.len == out_bits);
            for (out, 0..) |*o, i| {
                o.* = dot(self.rows[i * in_dim ..][0..in_dim], x);
            }
        }
    };
}

/// FMA-vectorized dot product; FP reassociation is explicit here because
/// Zig (correctly) won't reorder a scalar accumulation chain.
pub fn dot(a: []const f32, b: []const f32) f32 {
    const width = std.simd.suggestVectorLength(f32) orelse 4;
    var acc: @Vector(width, f32) = @splat(0);
    var i: usize = 0;
    while (i + width <= a.len) : (i += width) {
        const va: @Vector(width, f32) = a[i..][0..width].*;
        const vb: @Vector(width, f32) = b[i..][0..width].*;
        acc = @mulAdd(@Vector(width, f32), va, vb, acc);
    }
    var total = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) total = @mulAdd(f32, a[i], b[i], total);
    return total;
}

test "rotation rows are orthonormal" {
    const dim = 48;
    var rotation = try RandomRotation(dim).init(std.testing.allocator, 1234);
    defer rotation.deinit(std.testing.allocator);

    for (0..dim) |i| {
        const a = rotation.rows[i * dim ..][0..dim];
        for (i..dim) |j| {
            const b = rotation.rows[j * dim ..][0..dim];
            var acc: f64 = 0;
            for (a, b) |x, y| acc += @as(f64, x) * y;
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, acc, 1e-4);
        }
    }
}

test "rotation preserves inner products" {
    const dim = 48;
    var rotation = try RandomRotation(dim).init(std.testing.allocator, 99);
    defer rotation.deinit(std.testing.allocator);

    var prng = std.Random.DefaultPrng.init(5);
    const rand = prng.random();
    var a: [dim]f32 = undefined;
    var b: [dim]f32 = undefined;
    for (&a) |*v| v.* = rand.floatNorm(f32);
    for (&b) |*v| v.* = rand.floatNorm(f32);

    var ra: [dim]f32 = undefined;
    var rb: [dim]f32 = undefined;
    rotation.apply(&a, &ra);
    rotation.apply(&b, &rb);

    var ip: f64 = 0;
    var rip: f64 = 0;
    for (a, b) |x, y| ip += @as(f64, x) * y;
    for (ra, rb) |x, y| rip += @as(f64, x) * y;
    try std.testing.expectApproxEqAbs(ip, rip, 1e-3);
}

test "random projection sign codes estimate angular distance" {
    const in_dim = 50;
    const bits = 2048;
    var proj = try RandomProjection(bits, in_dim).init(std.testing.allocator, 7);
    defer proj.deinit(std.testing.allocator);

    var prng = std.Random.DefaultPrng.init(3);
    const rand = prng.random();
    var a: [in_dim]f32 = undefined;
    var b: [in_dim]f32 = undefined;
    for (&a) |*v| v.* = rand.floatNorm(f32);
    for (&b) |*v| v.* = rand.floatNorm(f32);

    const pa = try std.testing.allocator.alloc(f32, bits);
    defer std.testing.allocator.free(pa);
    const pb = try std.testing.allocator.alloc(f32, bits);
    defer std.testing.allocator.free(pb);
    proj.project(&a, pa);
    proj.project(&b, pb);

    // Fraction of differing signs estimates angle/pi (SimHash). Compare to
    // the true angle between a and b.
    var diff: usize = 0;
    for (pa, pb) |x, y| {
        if ((x > 0) != (y > 0)) diff += 1;
    }
    const est_angle = @as(f64, @floatFromInt(diff)) / bits * std.math.pi;

    var dotab: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dotab += @as(f64, x) * y;
        na += @as(f64, x) * x;
        nb += @as(f64, y) * y;
    }
    const true_angle = std.math.acos(dotab / (@sqrt(na) * @sqrt(nb)));
    try std.testing.expectApproxEqAbs(true_angle, est_angle, 0.1);
}
