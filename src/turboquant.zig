//! 3-bit TurboQuant payloads (arXiv:2504.19874).
//!
//! After random rotation the coordinates of a vector are approximately
//! Gaussian, so each one is quantized independently with the optimal 8-level
//! Lloyd-Max scalar quantizer for N(0,1), after a per-vector affine
//! calibration (bias_scale/bias_shift) that standardizes the coordinates.
//! A stored renormalization scalar makes the inner-product estimator
//! unbiased, mirroring the paper's Q_prod correction (Theorem 2).

const std = @import("std");

/// Canonical 8-level Lloyd-Max centroids for the standard normal
/// distribution, ascending. The paper's b=2 values (±0.453, ±1.51) come from
/// the same table.
pub const lloyd_max_centroids = [8]f32{
    -2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520,
};

/// Voronoi boundaries: midpoints between consecutive centroids.
pub const decision_thresholds = [7]f32{
    -1.7480, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7480,
};

/// Maps a standardized coordinate to the index of its nearest centroid.
pub fn quantizeCoord(value: f32) u3 {
    var code: u3 = 0;
    for (decision_thresholds) |threshold| {
        if (value < threshold) break;
        code += 1;
    }
    return code;
}

/// A compressed chunk of 8 3-bit coordinates (24 bits).
/// Note: Zig pads u24 to 4 bytes in memory; the 3-byte density is realized
/// when chunks are serialized, which is out of scope here.
pub const TQ3Chunk = packed struct {
    values: u24,
};

pub fn packChunk(codes: [8]u3) TQ3Chunk {
    var bits: u24 = 0;
    inline for (codes, 0..) |code, i| {
        bits |= @as(u24, code) << (3 * i);
    }
    return .{ .values = bits };
}

pub fn unpackChunk(chunk: TQ3Chunk) [8]u3 {
    var codes: [8]u3 = undefined;
    var bits = chunk.values;
    inline for (&codes) |*code| {
        code.* = @truncate(bits);
        bits >>= 3;
    }
    return codes;
}

/// The high-precision reranking payload, stored contiguously per vector.
pub fn TurboQuantPayload(comptime dim: usize) type {
    comptime std.debug.assert(dim > 0);
    const chunk_count = (dim + 7) / 8;
    return struct {
        const Self = @This();
        pub const chunks_count = chunk_count;

        id: u64,
        chunks: [chunk_count]TQ3Chunk,
        bias_scale: f32,
        bias_shift: f32,
        renorm_scalar: f32,

        /// Quantizes a (rotated) coordinate array. The affine calibration is
        /// derived from the vector's own mean and standard deviation, and the
        /// renormalization scalar is fit so that the reconstruction scores
        /// the original vector at exactly its squared norm:
        ///   renorm = ||v||^2 / <v, v_hat>
        pub fn encode(id: u64, coords: []const f32) Self {
            return encodeWithCalibration(id, coords, &.{}, &.{});
        }

        /// TQ+ variant: coordinates are standardized per coordinate before
        /// the per-vector calibration and quantization, but the stored
        /// renormalization keeps the estimator unbiased in the ORIGINAL
        /// rotated space — reconstruction decodes through the inverse
        /// affine: y_hat_j = (c[code]*s_v + m_v)*calib_scale_j + calib_shift_j.
        /// Empty calibration slices mean identity.
        pub fn encodeWithCalibration(
            id: u64,
            coords: []const f32,
            calib_shift: []const f32,
            calib_scale: []const f32,
        ) Self {
            std.debug.assert(coords.len == dim);
            const calibrated = calib_shift.len == dim;
            std.debug.assert(calibrated or calib_shift.len == 0);
            const dim_f: f64 = @floatFromInt(dim);

            var sum: f64 = 0;
            var sum_sq: f64 = 0;
            for (coords, 0..) |x, j| {
                const y: f64 = if (calibrated) (x - calib_shift[j]) / calib_scale[j] else x;
                sum += y;
                sum_sq += y * y;
            }
            const mean = sum / dim_f;
            const variance = @max(0.0, sum_sq / dim_f - mean * mean);
            const scale: f32 = @floatCast(@max(@sqrt(variance), 1e-12));
            const shift: f32 = @floatCast(mean);

            var self = Self{
                .id = id,
                .chunks = undefined,
                .bias_scale = scale,
                .bias_shift = shift,
                .renorm_scalar = 1.0,
            };

            var inner: f64 = 0; // <v, v_hat>, both in original rotated space
            var norm_sq: f64 = 0; // ||v||^2
            var j: usize = 0;
            for (&self.chunks) |*chunk| {
                var codes: [8]u3 = @splat(0);
                for (&codes) |*code| {
                    if (j >= dim) break;
                    const x = coords[j];
                    const y: f32 = if (calibrated) (x - calib_shift[j]) / calib_scale[j] else x;
                    code.* = quantizeCoord((y - shift) / scale);
                    var reconstructed = lloyd_max_centroids[code.*] * scale + shift;
                    if (calibrated) reconstructed = reconstructed * calib_scale[j] + calib_shift[j];
                    inner += @as(f64, x) * reconstructed;
                    norm_sq += @as(f64, x) * x;
                    j += 1;
                }
                chunk.* = packChunk(codes);
            }

            if (@abs(inner) > 1e-12) {
                self.renorm_scalar = @floatCast(norm_sq / inner);
            }
            return self;
        }

        /// Reconstructs the calibrated coordinates (without renormalization).
        pub fn decode(self: *const Self, out: []f32) void {
            std.debug.assert(out.len == dim);
            var j: usize = 0;
            for (self.chunks) |chunk| {
                const codes = unpackChunk(chunk);
                for (codes) |code| {
                    if (j >= dim) return;
                    out[j] = lloyd_max_centroids[code] * self.bias_scale + self.bias_shift;
                    j += 1;
                }
            }
        }

        /// Unbiased inner-product estimate against a precomputed lookup
        /// table (see `buildScoreLut`). Algebraically identical to `score`:
        ///   sum q_j * (c[code_j]*scale + shift)
        ///     = scale * sum q_j*c[code_j] + shift * sum q_j
        /// but needs only one table-add per coordinate. `a`/`b` are the
        /// query constants returned by the LUT builder; with per-coordinate
        /// (TQ+) calibration they fold the inverse affine into the same
        /// kernel, so the estimate stays in the original rotated space.
        pub fn scoreLut(self: *const Self, lut: []const f32, a: f32, b: f32) f32 {
            std.debug.assert(lut.len == dim * 8);
            var acc: f32 = 0;
            var j: usize = 0;
            outer: for (self.chunks) |chunk| {
                var bits = chunk.values;
                for (0..8) |_| {
                    if (j >= dim) break :outer;
                    const code: u3 = @truncate(bits);
                    bits >>= 3;
                    acc += lut[j * 8 + code];
                    j += 1;
                }
            }
            return (self.bias_scale * acc + self.bias_shift * a + b) * self.renorm_scalar;
        }

        /// Unbiased inner-product estimate against an already-decoded query.
        pub fn score(self: *const Self, decoded_query: []const f32) f32 {
            std.debug.assert(decoded_query.len == dim);
            var acc: f32 = 0;
            var j: usize = 0;
            for (self.chunks) |chunk| {
                const codes = unpackChunk(chunk);
                for (codes) |code| {
                    if (j >= dim) break;
                    const reconstructed = lloyd_max_centroids[code] * self.bias_scale + self.bias_shift;
                    acc += decoded_query[j] * reconstructed;
                    j += 1;
                }
            }
            return acc * self.renorm_scalar;
        }
    };
}

/// Standardizes, quantizes, and reconstructs a query in one pass, so stage-2
/// scoring compares 3-bit representations on both sides.
pub fn quantizeQuery(query: []const f32, out: []f32) void {
    std.debug.assert(query.len == out.len and query.len > 0);
    const dim_f: f64 = @floatFromInt(query.len);

    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    for (query) |x| {
        sum += x;
        sum_sq += @as(f64, x) * x;
    }
    const mean = sum / dim_f;
    const variance = @max(0.0, sum_sq / dim_f - mean * mean);
    const scale: f32 = @floatCast(@max(@sqrt(variance), 1e-12));
    const shift: f32 = @floatCast(mean);

    for (query, out) |x, *o| {
        const code = quantizeCoord((x - shift) / scale);
        o.* = lloyd_max_centroids[code] * scale + shift;
    }
}

pub const LutConstants = struct { a: f32, b: f32 };

/// Fills `lut[j*8 + k] = query[j] * centroid[k]` and returns the query
/// constants `scoreLut` needs (a = sum(q), b = 0).
pub fn buildScoreLut(query: []const f32, lut: []f32) LutConstants {
    std.debug.assert(lut.len == query.len * 8);
    const centroids: @Vector(8, f32) = lloyd_max_centroids;
    var sum: f32 = 0;
    for (query, 0..) |q, j| {
        lut[j * 8 ..][0..8].* = centroids * @as(@Vector(8, f32), @splat(q));
        sum += q;
    }
    return .{ .a = sum, .b = 0 };
}

/// TQ+ variant: folds the per-coordinate inverse affine into the table so
/// `scoreLut` estimates the inner product in the original rotated space:
///   lut[j][k] = q_j * calib_scale_j * c[k]
///   a = sum(q_j * calib_scale_j), b = sum(q_j * calib_shift_j)
pub fn buildScoreLutCalibrated(
    query: []const f32,
    calib_shift: []const f32,
    calib_scale: []const f32,
    lut: []f32,
) LutConstants {
    std.debug.assert(lut.len == query.len * 8);
    std.debug.assert(calib_shift.len == query.len and calib_scale.len == query.len);
    const centroids: @Vector(8, f32) = lloyd_max_centroids;
    var a: f32 = 0;
    var b: f32 = 0;
    for (query, calib_shift, calib_scale, 0..) |q, sh, sc, j| {
        lut[j * 8 ..][0..8].* = centroids * @as(@Vector(8, f32), @splat(q * sc));
        a += q * sc;
        b += q * sh;
    }
    return .{ .a = a, .b = b };
}

test "quantizeCoord picks nearest centroid at boundaries" {
    try std.testing.expectEqual(@as(u3, 0), quantizeCoord(-3.0));
    try std.testing.expectEqual(@as(u3, 1), quantizeCoord(-1.2));
    try std.testing.expectEqual(@as(u3, 3), quantizeCoord(-0.1));
    try std.testing.expectEqual(@as(u3, 4), quantizeCoord(0.0));
    try std.testing.expectEqual(@as(u3, 4), quantizeCoord(0.3));
    try std.testing.expectEqual(@as(u3, 7), quantizeCoord(5.0));
}

test "chunk pack/unpack roundtrip" {
    const patterns = [_][8]u3{
        .{ 0, 1, 2, 3, 4, 5, 6, 7 },
        .{ 7, 7, 7, 7, 7, 7, 7, 7 },
        .{ 0, 0, 0, 0, 0, 0, 0, 0 },
        .{ 5, 0, 7, 2, 1, 6, 3, 4 },
    };
    for (patterns) |codes| {
        try std.testing.expectEqual(codes, unpackChunk(packChunk(codes)));
    }
}

test "payload reconstruction tracks gaussian data" {
    const dim = 100;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    var coords: [dim]f32 = undefined;
    for (&coords) |*c| c.* = 3.0 * rand.floatNorm(f32) + 0.5;

    const payload = TurboQuantPayload(dim).encode(1, &coords);
    var decoded: [dim]f32 = undefined;
    payload.decode(&decoded);

    var err_sq: f64 = 0;
    var norm_sq: f64 = 0;
    for (coords, decoded) |x, x_hat| {
        err_sq += (x - x_hat) * (x - x_hat);
        norm_sq += @as(f64, x) * x;
    }
    // 3-bit Lloyd-Max MSE for a Gaussian is ~0.0345 * variance.
    try std.testing.expect(err_sq / norm_sq < 0.08);
}

test "renorm scalar makes self-score exactly the squared norm" {
    const dim = 64;
    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();

    var coords: [dim]f32 = undefined;
    var norm_sq: f64 = 0;
    for (&coords) |*c| {
        c.* = rand.floatNorm(f32);
        norm_sq += @as(f64, c.*) * c.*;
    }

    const payload = TurboQuantPayload(dim).encode(1, &coords);
    var acc: f64 = 0;
    var decoded: [dim]f32 = undefined;
    payload.decode(&decoded);
    for (coords, decoded) |x, x_hat| acc += @as(f64, x) * x_hat;
    acc *= payload.renorm_scalar;

    try std.testing.expectApproxEqRel(norm_sq, acc, 1e-5);
}

test "scoreLut matches the direct scoring kernel" {
    const dim = 45; // exercises a partial tail chunk
    var prng = std.Random.DefaultPrng.init(21);
    const rand = prng.random();

    var coords: [dim]f32 = undefined;
    var query: [dim]f32 = undefined;
    for (&coords) |*c| c.* = 2.0 * rand.floatNorm(f32) - 0.3;
    for (&query) |*q| q.* = rand.floatNorm(f32);

    const payload = TurboQuantPayload(dim).encode(3, &coords);
    var lut: [dim * 8]f32 = undefined;
    const consts = buildScoreLut(&query, &lut);

    try std.testing.expectApproxEqRel(
        payload.score(&query),
        payload.scoreLut(&lut, consts.a, consts.b),
        1e-4,
    );
}

test "calibrated encode + LUT estimates the raw-space inner product" {
    const dim = 64;
    var prng = std.Random.DefaultPrng.init(57);
    const rand = prng.random();

    // Skewed per-coordinate distribution the calibration should absorb.
    var calib_shift: [dim]f32 = undefined;
    var calib_scale: [dim]f32 = undefined;
    for (&calib_shift, &calib_scale, 0..) |*sh, *sc, j| {
        sh.* = 0.3 * @as(f32, @floatFromInt(j % 5));
        sc.* = 0.5 + 0.1 * @as(f32, @floatFromInt(j % 7));
    }

    var coords: [dim]f32 = undefined;
    var query: [dim]f32 = undefined;
    for (&coords, calib_shift, calib_scale) |*c, sh, sc| {
        c.* = rand.floatNorm(f32) * sc + sh;
    }
    // Correlate the query with the stored vector so the exact inner product
    // is large relative to the quantizer noise (a near-orthogonal pair would
    // make any relative tolerance meaningless).
    for (&query, coords) |*q, c| q.* = c + 0.3 * rand.floatNorm(f32);

    const payload = TurboQuantPayload(dim).encodeWithCalibration(1, &coords, &calib_shift, &calib_scale);
    var lut: [dim * 8]f32 = undefined;
    const consts = buildScoreLutCalibrated(&query, &calib_shift, &calib_scale, &lut);
    const estimate = payload.scoreLut(&lut, consts.a, consts.b);

    var exact: f64 = 0;
    for (coords, query) |x, q| exact += @as(f64, x) * q;

    // 3-bit estimate of the raw-space inner product: loose tolerance, but it
    // must be in the right space and ballpark (the broken all-calibrated
    // variant was off by the affine transform entirely).
    try std.testing.expectApproxEqRel(exact, @as(f64, estimate), 0.05);

    // And the self-estimate must renormalize to the exact squared norm.
    var lut_self: [dim * 8]f32 = undefined;
    const self_consts = buildScoreLutCalibrated(&coords, &calib_shift, &calib_scale, &lut_self);
    var norm_sq: f64 = 0;
    for (coords) |x| norm_sq += @as(f64, x) * x;
    try std.testing.expectApproxEqRel(
        norm_sq,
        @as(f64, payload.scoreLut(&lut_self, self_consts.a, self_consts.b)),
        1e-4,
    );
}

test "tail coordinates beyond dim never contribute" {
    const dim = 5; // one chunk with 3 dead slots
    const coords = [dim]f32{ 1.0, -2.0, 0.5, 3.0, -1.5 };
    const payload = TurboQuantPayload(dim).encode(9, &coords);

    var query: [dim]f32 = @splat(1.0);
    var decoded_query: [dim]f32 = undefined;
    quantizeQuery(&query, &decoded_query);

    var expected: f32 = 0;
    var decoded: [dim]f32 = undefined;
    payload.decode(&decoded);
    for (decoded, decoded_query) |x_hat, q| expected += x_hat * q;
    expected *= payload.renorm_scalar;

    try std.testing.expectApproxEqAbs(expected, payload.score(&decoded_query), 1e-6);
}
