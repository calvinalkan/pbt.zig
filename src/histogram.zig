//! `histogram` counts how often chosen values come up among the draws of a
//! property test. `Histogram` counts them and checks the counts: the share of
//! each value, that every chosen value came up, and that nothing else did.
//! `withBothSigns` builds lists of float values to watch with both signs.
const std = @import("std");
const assert = std.debug.assert;
const Exhaustive = @import("exhaustive.zig");

/// `Histogram` returns a type that counts how often each of `watched_values`
/// comes up among the values added to it:
///
/// ```zig
/// const watched_values = [_]u32{ 0, 9 };
/// var digits: histogram.Histogram(u32, &watched_values) = .{};
///
/// for (0..10_000) |_| {
///     digits.add(prng.intBetween(u32, 0, 9));
/// }
///
/// // 0 and 9 each come up in about 1 draw of 10. Over 10,000 draws, that
/// // share has a standard error of 0.003, so 0.02 is about seven standard
/// // errors.
/// try digits.expectShare(0, 0.1, 0.02);
/// try digits.expectShare(9, 0.1, 0.02);
/// ```
///
/// To check how often a condition holds, add booleans and watch `true`.
///
/// `Histogram` compares floats by their bits, so a NaN matches a NaN with the
/// same bits, and 0 and -0 count as different values. `count`, `share`, and
/// `expectShare` require a `value` from `watched_values`, and panic for any
/// other.
pub fn Histogram(comptime T: type, comptime watched_values: []const T) type {
    return struct {
        const Self = @This();

        counts: [watched_values.len]usize = @splat(0),
        total_count: usize = 0,
        first_unwatched_value: ?T = null,

        /// `add` counts `value` toward the total, and toward its own count if
        /// it is one of `watched_values`.
        pub fn add(counter: *Self, value: T) void {
            counter.total_count += 1;

            if (watchedIndex(value)) |watched_index| {
                counter.counts[watched_index] += 1;
            } else if (counter.first_unwatched_value == null) {
                counter.first_unwatched_value = value;
            }
        }

        /// `count` returns how often `value` came up.
        pub fn count(counter: *const Self, value: T) usize {
            return counter.counts[watchedIndexOrPanic(value)];
        }

        /// `share` returns the share of all values added that were `value`,
        /// from 0 to 1. At least one value must have been added.
        pub fn share(counter: *const Self, value: T) f64 {
            assert(counter.total_count > 0);

            const value_count: f64 = @floatFromInt(counter.count(value));
            const total_count: f64 = @floatFromInt(counter.total_count);

            return value_count / total_count;
        }

        /// `expectShare` returns `error.TestUnexpectedResult` unless `value`
        /// made up `expected_share` of all values added, within `tolerance`,
        /// and it prints both shares when it fails. A `tolerance` of 0
        /// requires the exact share, so `expectShare(value, 0, 0)` checks that
        /// `value` never came up.
        pub fn expectShare(
            counter: *const Self,
            value: T,
            expected_share: f64,
            tolerance: f64,
        ) !void {
            if (!counter.shareIsWithin(value, expected_share, tolerance)) {
                std.debug.print(
                    "share of {any} is {d}, not within {d} of {d}\n",
                    .{ value, counter.share(value), tolerance, expected_share },
                );

                return error.TestUnexpectedResult;
            }
        }

        /// `expectAllSeen` returns `error.TestUnexpectedResult` unless every
        /// watched value came up at least once, and it prints the first one
        /// that didn't.
        pub fn expectAllSeen(counter: *const Self) !void {
            if (counter.firstUnseenValue()) |unseen_value| {
                std.debug.print("{any} never came up\n", .{unseen_value});

                return error.TestUnexpectedResult;
            }
        }

        /// `expectNothingElse` returns `error.TestUnexpectedResult` if a value
        /// outside `watched_values` was added, and it prints the first such
        /// value.
        pub fn expectNothingElse(counter: *const Self) !void {
            if (counter.first_unwatched_value) |unwatched_value| {
                std.debug.print("unexpected value {any}\n", .{unwatched_value});

                return error.TestUnexpectedResult;
            }
        }

        fn shareIsWithin(counter: *const Self, value: T, expected_share: f64, tolerance: f64) bool {
            return @abs(counter.share(value) - expected_share) <= tolerance;
        }

        fn firstUnseenValue(counter: *const Self) ?T {
            for (counter.counts, watched_values) |value_count, watched_value| {
                if (value_count == 0) {
                    return watched_value;
                }
            }

            return null;
        }

        fn watchedIndexOrPanic(value: T) usize {
            return watchedIndex(value) orelse {
                std.debug.panic("{any} is not a watched value", .{value});
            };
        }

        fn watchedIndex(value: T) ?usize {
            for (watched_values, 0..) |watched_value, watched_index| {
                if (isSameValue(watched_value, value)) {
                    return watched_index;
                }
            }

            return null;
        }

        fn isSameValue(a: T, b: T) bool {
            if (@typeInfo(T) != .float) {
                return a == b;
            }

            const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));

            return @as(Bits, @bitCast(a)) == @as(Bits, @bitCast(b));
        }
    };
}

/// `withBothSigns` returns each of `magnitudes` followed by its negation, for
/// watching a float that may come up with either sign. It runs only at
/// compile time, so bind its result with `comptime`:
///
/// ```zig
/// // 0, -0, NaN, and the NaN with the sign bit set.
/// const edges = comptime histogram.withBothSigns(f32, &.{ 0, std.math.nan(f32) });
/// var edge_counter: histogram.Histogram(f32, edges) = .{};
/// ```
pub fn withBothSigns(comptime F: type, comptime magnitudes: []const F) []const F {
    comptime {
        var values: [2 * magnitudes.len]F = undefined;
        for (magnitudes, 0..) |magnitude, magnitude_index| {
            values[2 * magnitude_index] = magnitude;
            values[2 * magnitude_index + 1] = -magnitude;
        }

        const result = values;

        return &result;
    }
}

// A failing check prints why it failed, and `zig build test` treats any
// output as a failure, so the tests call the `expect` methods only where they
// pass. The failures are checked through what those methods read:
// `shareIsWithin`, `firstUnseenValue`, and `first_unwatched_value`.

test "histogram.Histogram counts every sequence of added values" {
    // Every sequence of up to 4 values from 0 to 3, with 0 and 1 watched:
    // each count, the total, the first unseen value, and the first unwatched
    // value match a count of the sequence itself. Two unwatched values, 2 and
    // 3, tell the first unwatched value from the last.
    const watched_values = [_]u8{ 0, 1 };

    var exhaustive: Exhaustive = .{};

    while (!exhaustive.done()) {
        const length = exhaustive.intBetween(usize, 0, 4);

        var added_values: [4]u8 = undefined;
        for (added_values[0..length]) |*added_value| {
            added_value.* = exhaustive.intBetween(u8, 0, 3);
        }

        var counter: Histogram(u8, &watched_values) = .{};

        for (added_values[0..length]) |added_value| {
            counter.add(added_value);
        }

        const sequence = added_values[0..length];
        const zero_count = std.mem.count(u8, sequence, &.{0});
        const one_count = std.mem.count(u8, sequence, &.{1});

        const expected_first_unseen_value: ?u8 = if (zero_count == 0)
            0
        else if (one_count == 0)
            1
        else
            null;

        const expected_first_unwatched_value: ?u8 = if (std.mem.indexOfAny(
            u8,
            sequence,
            &.{ 2, 3 },
        )) |unwatched_index|
            sequence[unwatched_index]
        else
            null;

        try std.testing.expectEqual(zero_count, counter.count(0));
        try std.testing.expectEqual(one_count, counter.count(1));
        try std.testing.expectEqual(length, counter.total_count);
        try std.testing.expectEqual(expected_first_unseen_value, counter.firstUnseenValue());
        try std.testing.expectEqual(expected_first_unwatched_value, counter.first_unwatched_value);

        if (expected_first_unseen_value == null) {
            try counter.expectAllSeen();
        }

        if (expected_first_unwatched_value == null) {
            try counter.expectNothingElse();
        }
    }
}

test "histogram.Histogram.share divides a value's count by the total" {
    // Each case adds `added_values` and checks the share of `true`, which
    // `expectShare` then accepts with a tolerance of 0.
    const Case = struct { added_values: []const bool, share: f64 };

    const cases = [_]Case{
        .{ .added_values = &.{true}, .share = 1 },
        .{ .added_values = &.{false}, .share = 0 },
        .{ .added_values = &.{ true, false }, .share = 0.5 },
        .{ .added_values = &.{ true, false, false, false }, .share = 0.25 },
        .{ .added_values = &.{ false, true, true }, .share = 2.0 / 3.0 },
    };

    const watched_values = [_]bool{true};

    for (cases) |case| {
        var counter: Histogram(bool, &watched_values) = .{};

        for (case.added_values) |added_value| {
            counter.add(added_value);
        }

        try std.testing.expectEqual(case.share, counter.share(true));
        try counter.expectShare(true, case.share, 0);
    }
}

test "histogram.Histogram.expectShare accepts a share within the tolerance only" {
    // `true` makes up a quarter of the values added. Each case asks for a
    // share and a tolerance, and says whether 0.25 lies within them.
    const Case = struct { expected_share: f64, tolerance: f64, is_within: bool };

    const cases = [_]Case{
        .{ .expected_share = 0.25, .tolerance = 0, .is_within = true },
        .{ .expected_share = 0.3, .tolerance = 0.06, .is_within = true },
        .{ .expected_share = 0.2, .tolerance = 0.06, .is_within = true },
        .{ .expected_share = 0.3, .tolerance = 0.04, .is_within = false },
        .{ .expected_share = 0.2, .tolerance = 0.04, .is_within = false },
        .{ .expected_share = 0, .tolerance = 0, .is_within = false },
    };

    const watched_values = [_]bool{true};
    var counter: Histogram(bool, &watched_values) = .{};

    for ([_]bool{ true, false, false, false }) |added_value| {
        counter.add(added_value);
    }

    for (cases) |case| {
        try std.testing.expectEqual(
            case.is_within,
            counter.shareIsWithin(true, case.expected_share, case.tolerance),
        );

        if (case.is_within) {
            try counter.expectShare(true, case.expected_share, case.tolerance);
        }
    }
}

test "histogram.Histogram matches enum tags, and floats by their bits" {
    // Tags compare as tags. A NaN matches a NaN with the same bits, although
    // `==` never matches NaN, and -0 does not match 0, although `==` says it
    // does.
    const Color = enum { red, green, blue };

    const watched_colors = [_]Color{ .red, .blue };
    var colors: Histogram(Color, &watched_colors) = .{};

    for ([_]Color{ .red, .green, .red }) |color| {
        colors.add(color);
    }

    try std.testing.expectEqual(2, colors.count(.red));
    try std.testing.expectEqual(0, colors.count(.blue));
    try std.testing.expectEqual(Color.green, colors.first_unwatched_value);

    const nan = std.math.nan(f32);
    const watched_floats = [_]f32{ nan, 0 };
    var floats: Histogram(f32, &watched_floats) = .{};

    floats.add(nan);
    floats.add(-0.0);

    try std.testing.expectEqual(1, floats.count(nan));
    try std.testing.expectEqual(0, floats.count(0));
    try std.testing.expect(floats.first_unwatched_value != null);
}

test "histogram.withBothSigns follows each magnitude with its negation" {
    const nan = std.math.nan(f32);
    const values = comptime withBothSigns(f32, &.{ 1, 0, nan });

    // Bits tell 0 from -0 and the two NaNs apart, which `==` does not.
    const expected_bits = [_]u32{
        @bitCast(@as(f32, 1)), @bitCast(@as(f32, -1)),
        @bitCast(@as(f32, 0)), @bitCast(@as(f32, -0.0)),
        @bitCast(nan),         @bitCast(-nan),
    };

    var actual_bits: [values.len]u32 = undefined;
    for (values, &actual_bits) |value, *value_bits| {
        value_bits.* = @bitCast(value);
    }

    try std.testing.expectEqualSlices(u32, &expected_bits, &actual_bits);
}
