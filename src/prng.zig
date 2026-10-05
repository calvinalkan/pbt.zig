//! `Prng` is the seeded random source for property tests.
//!
//! A property test creates one `Prng` from `std.testing.random_seed` and draws
//! every random input from it, for one case after another:
//!
//! ```zig
//! test "sort" {
//!     const ValueKind = enum { literal, any };
//!
//!     var prng = Prng.init(std.testing.random_seed);
//!     var storage: [4096]u32 = undefined;
//!
//!     for (0..1024) |_| {
//!         const values_count = @min(prng.intExponential(usize, 256), storage.len);
//!         const values = storage[0..values_count];
//!
//!         // Each case draws its own mix: some cases draw only literals,
//!         // which repeat and so give sort many ties, some only arbitrary
//!         // u32s, and some both.
//!         const value_kind_weights = prng.enumWeights(ValueKind, .full);
//!
//!         for (values) |*value| {
//!             value.* = switch (prng.enumWeighted(ValueKind, value_kind_weights)) {
//!                 .literal => prng.intLiteral(u32),
//!                 .any => prng.intAny(u32),
//!             };
//!         }
//!
//!         std.mem.sort(u32, values, {}, std.sort.asc(u32));
//!
//!         const is_sorted = std.sort.isSorted(u32, values, {}, std.sort.asc(u32));
//!         try std.testing.expect(is_sorted);
//!     }
//! }
//! ```
//!
//! The build runner picks a new seed for every `zig build` and passes it to
//! the test binaries, which store it in `std.testing.random_seed`, so repeated
//! runs keep exploring new inputs. A failing `zig build test` prints the seed
//! in its failed command as `--seed=0x…`. The same seed produces the same
//! draws, so `zig build test --seed 0x… -- "sort"` replays the failure
//! exactly: the cases before the failing one pass again, and the failing case
//! fails again.
//!
//! A replay reaches the failing case only if the test runs the same cases in
//! the same order. The number of cases must therefore be fixed, like the 1024
//! cases above, and must never depend on elapsed time; otherwise, a replay on
//! a slower machine could stop before it reaches the failing case.
//!
//! A seed replays the same draws only with the same Zig version. `Prng` uses
//! std's Xoshiro256 generator and std's code for turning random bits into
//! numbers, and a Zig upgrade may change either. A seed is therefore no
//! substitute for a regression test.
//!
//! A seed replays the same draws on 32-bit and 64-bit targets, even for
//! `prng.intBetween(usize, 0, 9)`. The exceptions are draws whose possible
//! values depend on the size of `usize`, such as `prng.intAny(usize)` and
//! `prng.intEdge(usize)`.
//!
//! ## Concepts
//!
//! Will Wilson's talk "Will Wilson on Swarm Testing"
//! (https://www.youtube.com/watch?v=wzfC7Q-xNik) makes the case this file is
//! built on. Uniform random inputs look the same everywhere: a random string
//! of operations almost never contains a long run of inserts followed by a
//! long run of removes, and a random number is almost never 255 or 0. Real
//! users produce exactly such structured inputs, and bugs hide in them. Swarm
//! testing fixes part of this by turning features on and off per case, so that
//! some cases never delete and others do almost nothing else.
//!
//! Wilson offers a conjecture for why that works: the inputs most likely to
//! find bugs in code written by people are the ones with a short description.
//! Kolmogorov complexity makes "short description" precise. The Kolmogorov
//! complexity K(x) of a value x is the length of the shortest program that
//! prints x and stops:
//!
//! ```text
//! K(x) = min { length(p) : program p prints x }
//! ```
//!
//! The programs are written in one fixed language; choosing another language
//! changes K by at most a constant. Some values and their shortest programs:
//!
//! ```text
//! value                                  program that prints it     K
//! 1                                      print("1")                 low
//! 1_000                                  print("1" + "0" * 3)       low
//! the largest u64, 64 ones in binary     print("1" * 64)            low
//! a slice of 4_096 zeros                 print([0] * 4096)          low
//! 7_381_026_594_117, drawn at random     print("7381026594117")     as long as the value
//! a slice of 4_096 random bytes          print(<all 4_096 bytes>)   as long as the slice
//! ```
//!
//! A random value has no program much shorter than spelling the value out, so
//! its K is as high as it gets for its length. Wilson conjectures that the best
//! inputs are drawn with a weight that falls as K grows: the higher an input's
//! complexity, the less often it should come up. The classic form of such a
//! weight is 2^-K(x), which halves with every bit that the shortest program
//! needs. In this view, swarm testing works because it pushes inputs toward
//! low K.
//!
//! K can't be computed: no program finds the shortest program for every value.
//! Use it as a way of thinking instead, by asking how short a program could
//! print a given input. `Prng` uses it as a rule of thumb, not a measurement:
//! it favors families of values that take little to write down, such as the
//! edges of types and slices, numbers made of a few digits, the bounds of a
//! range, and simple arrangements of a slice. The simplest values come up
//! most, and each extra digit makes a value rarer, though more slowly than
//! 2^-K would, so that complex values still come up. The exact shares are
//! judgment calls, not tuned against real bugs.
//!
//! Every such model misses something, so tests still draw fully random values
//! from the noise functions. Make noise one more kind in the case's weights,
//! next to edges and literals, as the mixing example below shows, rather than
//! a fixed share of every draw. Then some cases draw no noise at all and keep
//! their structure from start to end, while others draw mostly noise. A fixed
//! share of 1 draw in 20 would put noise into practically every case of 1_000
//! draws: all 1_000 draws skip it with probability about 5e-23. A property
//! that must hold across a whole range, such as the accuracy of
//! `sigmoidGated` for every gate in [-80, 80], also gets its own loop of plain
//! `floatBetween` draws.
//!
//! ## Functions
//!
//! - Noise: `intAny` and `intBetween` give every integer the same weight, and
//!   `floatAny` and `floatFinite` every bit pattern, so every exponent is
//!   equally likely. `floatBetween` gives equal widths of its range the same
//!   weight, so magnitudes far below the range's width are rare.
//! - Edges: `intEdge`, `floatEdge`, `floatEdgeFinite`, and `indexEdge` return
//!   values where a type or a slice ends, such as 255, NaN, or the last index.
//! - Literals: `intLiteral`, `floatLiteral`, `intLiteralBetween`, and
//!   `floatLiteralBetween` return numbers people commonly type in source code,
//!   such as 4_096 for a buffer, 8 or 16 for a vector length, 1_000 for a
//!   limit, 255 for a mask, or 0.5 for a scale, and the bounds of a range.
//!   Such numbers are made of a few digits, so they take little to write down.
//! - Sizes: `intLogUniform`, `floatLogUniform`, and `intExponential` return
//!   sizes and counts across every scale.
//! - Structure: `fillPattern` arranges a few values across a slice, such as
//!   all equal or one odd value, and `enumSet` and `enumWeights` give each case
//!   its own lopsided mix of operations.
//! - Boundaries: `nudge` moves a value to its neighbors, to probe both sides of
//!   a boundary.
//! - Plain draws: `boolean`, `chance`, `enumTag`, `enumWeighted`, `index`,
//!   `pick`, `shuffle`, and `floatGaussian`.
//!
//! `Prng` draws only from what the caller passes, such as a type, a range, or
//! a list. Everything that needs to know what the values mean belongs in the
//! test: domain values such as timestamps or Unicode code points, reusing keys
//! drawn earlier, and the choice of which functions to draw from. A test mixes
//! the functions with `enumWeights`, the same way it mixes operations:
//!
//! ```zig
//! const IntKind = enum { edge, literal, any };
//!
//! // Once per case, so some cases draw only edges, only literals, or only
//! // noise, and others a mix.
//! const int_kind_weights = prng.enumWeights(IntKind, .full);
//!
//! // Per draw.
//! const key = switch (prng.enumWeighted(IntKind, int_kind_weights)) {
//!     .edge => prng.intEdge(u32),
//!     .literal => prng.intLiteral(u32),
//!     .any => prng.intAny(u32),
//! };
//! ```
//!
//! A kind that a case leaves out should not come back through another kind.
//! `floatAny` and `floatEdge` also return NaN and the infinities, so a test
//! that gives non-finite values a kind of their own draws its other kinds
//! from `floatFinite`, `floatLiteral`, and `floatEdgeFinite`.
//!
//! The design is inspired by TigerBeetle's `src/stdx/prng.zig` and
//! `src/testing/fuzz.zig`
//! (https://github.com/tigerbeetle/tigerbeetle/tree/47aeb2212a255273dda508288412e537d11e4b7c/src),
//! which are licensed under the Apache License, Version 2.0.
const Prng = @This();
const std = @import("std");
const assert = std.debug.assert;

// `std.Random.DefaultPrng` is the same generator today, but std may point it
// at a different one, which would change the draws of every seed.
generator: std.Random.Xoshiro256,

/// `init` returns a `Prng` whose draws are determined by `seed`. Property
/// tests normally pass `std.testing.random_seed`.
pub fn init(seed: u64) Prng {
    return .{ .generator = .init(seed) };
}

// ─── Integers ───────────────────────────────────────────────────────────────

/// `intAny` returns a random value of the integer type `T`, with every value of
/// `T` equally likely.
pub fn intAny(prng: *Prng, comptime T: type) T {
    return prng.generator.random().int(T);
}

/// `intBetween` returns a random integer in [`min`, `max`], with every value
/// equally likely. The result does not depend on `T`, so
/// `prng.intBetween(u8, 0, 7)` and `prng.intBetween(usize, 0, 7)` return the
/// same values from the same seed, on any target. `T` must have at most 64
/// bits, and `min` must not exceed `max`.
pub fn intBetween(prng: *Prng, comptime T: type, min: T, max: T) T {
    comptime assert(@bitSizeOf(T) <= 64);
    assert(min <= max);

    // std draws an integer as wide as the type it is given, so drawing in
    // `T` would make `usize` draws differ between 32-bit and 64-bit targets.
    const Wide = if (@typeInfo(T).int.signedness == .signed) i64 else u64;

    return @intCast(prng.generator.random().intRangeAtMost(Wide, min, max));
}

/// `intExponential` returns a random integer from an exponential distribution
/// with mean `mean`, rounded down. Rounding down makes the mean of many
/// draws about `mean - 0.5`. Small results are the most common: about 63% of
/// the draws fall below `mean`, 86% fall below twice `mean`, and 5% land at
/// or above three times `mean`. Use it for the sizes of test inputs, which
/// are then mostly small but sometimes large:
///
/// ```zig
/// // Mostly sizes below 1_000, some between 1_000 and 3_000, and now and then
/// // one above 3_000.
/// const items_count = prng.intExponential(usize, 1_000);
/// ```
///
/// The result has no upper bound other than `std.math.maxInt(T)`, where
/// larger values stop, so clamp it to a buffer's length with `@min`. `T` must
/// be unsigned.
///
/// With a fixed `mean`, results at or above 10 times `mean` almost never come
/// up: about 1 draw in 22_000. To get both tiny and huge inputs, draw a new
/// `mean` for each case:
///
/// ```zig
/// // Most cases stay small, and some reach tens of thousands of elements.
/// const mean = prng.intLogUniform(usize, 4, 16_384);
/// const items_count = prng.intExponential(usize, mean);
/// ```
pub fn intExponential(prng: *Prng, comptime T: type, mean: T) T {
    comptime assert(@typeInfo(T).int.signedness == .unsigned);

    // Rounding an exponentially distributed float down gives geometrically
    // distributed integers. TigerBeetle draws them through a float because it
    // found no quick integer-only way to draw them, and this function does the
    // same. `lossyCast` truncates the scaled float toward zero and returns
    // `maxInt(T)` for any value above `maxInt(T)`.
    const value_float = prng.generator.random().floatExp(f64) * @as(f64, @floatFromInt(mean));

    return std.math.lossyCast(T, value_float);
}

/// `intLogUniform` returns a random integer in [`min`, `max`] with every order
/// of magnitude equally likely. For [1, 999], results from 1 to 9, from 10 to
/// 99, and from 100 to 999 each come up in 1 draw of 3:
///
/// ```zig
/// // Lengths from 0 to 4_096: 1 to 15 come up as often as 256 to 4_095, and
/// // about 1 length in 13 is 0.
/// const items_count = prng.intLogUniform(usize, 0, 4_096);
/// ```
///
/// `intBetween` over the same range would almost never return a small length:
/// fewer than 1 draw in 250 would be below 16. Within one order of magnitude,
/// smaller values are more likely: for [1, 3], 1 comes up in half the draws, 2
/// in 29%, and 3 in 21%.
///
/// 0 has no place on a log scale, so `intLogUniform` gives it the same share
/// as 1: 1 draw in 3 for [0, 3], and 1 in 13 for [0, 4_096]. To draw 0 more
/// often, make it one more kind in the case's weights, as the file header
/// shows. Pass a `min` of 1 where 0 is not valid, such as for a weight or a
/// divisor.
///
/// `T` must be unsigned with at most 64 bits, and `min` must not exceed `max`.
/// Results far above `min` skip some integers, because the draw goes through
/// f64: with a `min` of 0 or 1, results above about 2^47 can't all come up.
pub fn intLogUniform(prng: *Prng, comptime T: type, min: T, max: T) T {
    comptime assert(@typeInfo(T).int.signedness == .unsigned);
    comptime assert(@bitSizeOf(T) <= 64);
    assert(min <= max);

    // Rounding e^u down, with u drawn evenly from ln(min) to ln(max + 1),
    // gives each integer k the share between ln(k) and ln(k + 1), so every
    // tenfold range gets the same share.

    // 0 has no logarithm, so for a `min` of 0, u starts at ln(1/2). Rounding
    // down turns everything from 1/2 to 1 into 0, a stretch as long on the log
    // scale as 1's stretch from 1 to 2, so 0 comes up as often as 1.
    // `log_ratio` is u - ln(1/2), so it starts at 0.
    if (min == 0) {
        const max_float: f64 = @floatFromInt(max);
        const log_ratio = prng.floatBetween(f64, 0, @log(2 * (max_float + 1)));
        const value = std.math.lossyCast(T, @floor(0.5 * @exp(log_ratio)));

        // Rounding in `@exp` can put the result just past `max`.
        return @min(value, max);
    }

    // For a `min` of 1 or more, the code draws `log_ratio` = u - ln(min) and
    // returns min + floor(min * (e^log_ratio - 1)), which is the same value.
    // Drawing u directly would lose narrow ranges of large numbers: in f64,
    // ln(2^60) and ln(2^60 + 100) round to the same number, so every draw
    // would return `min`. `log_ratio` starts at 0, where f64 has precision to
    // spare. A range from 0 is never a narrow range of large numbers, which is
    // why the branch above can skip this. `min` is at least 1 here, so
    // `max - min + 1` fits in `T`.
    const min_float: f64 = @floatFromInt(min);
    const width: f64 = @floatFromInt(max - min + 1);
    const log_ratio = prng.floatBetween(f64, 0, std.math.log1p(width / min_float));
    const excess = std.math.lossyCast(T, @floor(min_float * std.math.expm1(log_ratio)));

    // Rounding in `expm1` can put the excess just past `max - min`.
    return min + @min(excess, max - min);
}

/// `intEdge` returns an integer at an edge of `T` or of a narrower integer
/// type, each equally likely. The edges are:
///
/// - 0, 1, and -1
/// - the minimum and maximum of `T`
/// - the minimums and maximums of i8, u8, i16, u16, i32, u32, i64, and u64
///
/// Values that don't fit in `T` are left out:
///
/// - `u8`: 0, 1, 127, 255
/// - `i8`: 0, 1, -1, -128, 127
/// - `u12`: 0, 1, 127, 255, 4_095
/// - `i32`: 0, 1, -1, -128, 127, 255, -32_768, 32_767, 65_535,
///   -2_147_483_648, 2_147_483_647
///
/// Code tends to break at these values when it squeezes a number into a
/// narrower type: 255 is the largest `u32` that `@truncate` to `u8` leaves
/// unchanged, and -1 becomes 255 or 65_535 when read as unsigned.
///
/// `intAny` almost never returns these values, so draw them as one kind
/// among `intLiteral` and `intAny` in the case's weights, as the file header
/// shows.
///
/// For the values next to an edge, such as 256 after 255, pass the result to
/// `nudge`: `prng.nudge(u32, prng.intEdge(u32), 1)`. Values that programs
/// merely tend to use, such as buffer sizes of 100 or 4_096, are not edges of
/// a type; `intLiteral` returns those. `T` must have at most 64 bits.
pub fn intEdge(prng: *Prng, comptime T: type) T {
    comptime assert(@bitSizeOf(T) <= 64);

    // `i128` holds every candidate, including the largest `u64`, and any value
    // of `T`, as in `nudge`.
    const edges = comptime selectDistinctExact(T, &[_]i128{
        0,                    1,                    -1,
        std.math.minInt(i8),  std.math.maxInt(i8),  std.math.maxInt(u8),
        std.math.minInt(i16), std.math.maxInt(i16), std.math.maxInt(u16),
        std.math.minInt(i32), std.math.maxInt(i32), std.math.maxInt(u32),
        std.math.minInt(i64), std.math.maxInt(i64), std.math.maxInt(u64),
        std.math.minInt(T),   std.math.maxInt(T),
    });

    return prng.pick(T, edges);
}

/// `intLiteral` returns a number that people often write as a literal in
/// code, drawn from all of `T` the way `intLiteralBetween` draws from a range:
/// the minimum or maximum of `T`, 0, or a number made of a few digits, such as
/// 1, 3, 16, 64, 1_000, 31 (11111 in binary), or 999. For `u16`, 0 and 65_535
/// come up about 12% of the time each, 1 about 5%, and numbers such as 7, 8,
/// 15, 16, 31, 32, 255, and 256 about 0.4% to 1% each. For a signed `T`, about
/// half the nonzero results are negative.
///
/// Code uses these numbers far more often than random ones: lengths of two or
/// three, capacities of 64, limits of 1_000, masks of 255, and shifts by 32,
/// where `x << 32` on a `u32` is a bug. `intAny` almost never returns them, so
/// draw them as one kind among `intEdge` and `intAny` in the case's weights,
/// as the file header shows. Pass the result to `nudge` to also get 65 after
/// 64 or 1_001 after 1_000.
///
/// Domain values such as timestamps or Unicode code points belong in the
/// test. `T` must have at most 64 bits.
pub fn intLiteral(prng: *Prng, comptime T: type) T {
    return prng.intLiteralBetween(T, std.math.minInt(T), std.math.maxInt(T));
}

/// `intLiteralBetween` returns an integer in [`min`, `max`] that takes little
/// to write down: `min`, `max`, or 0, or a number made of a few digits, such
/// as 1, 64, 1_000, 31 (11111 in binary), or 999. Values that take less to
/// write come up more often. For [-80, 80], the most common results are:
///
/// ```text
/// -80, 0, 80    about 11% each
/// -1, 1         about 5% each
/// -2, 2         about 3% each
/// ±3, ±4        about 2% each
/// ±5, ±7, ±8    about 1.2% each
/// ```
///
/// Each draw picks binary or decimal with equal odds, then how many leading
/// digits to keep: from 0 up to all the digits of the range's largest
/// magnitude, with 0 digits as likely as 1 and each further digit rarer. With
/// 0 digits, the result is `min`, `max`, or 0, whichever lie in the range, with
/// equal odds. Otherwise, it draws a number with every order of magnitude
/// equally likely, from 1 or the range's inner bound up, keeps that many of
/// its leading digits, and turns the rest into zeros or repeats of the kept
/// digits, with equal odds:
///
/// ```text
/// 37 = 100101 in binary, 1 digit kept    100000 = 32    111111 = 63
///                        2 digits kept   100000 = 32    101010 = 42
/// 912 in decimal, 1 digit kept           900            999
/// 1_234 in decimal, 2 digits kept        1_200          1_212
/// ```
///
/// With zeros, the result rounds to the nearest such number. If a result
/// leaves the range, one more digit is kept. Arbitrary values such as 937
/// need all their digits, so they come up least often. For [37, 1_234], the
/// bounds come up most often, at about 13% each, then numbers such as 64,
/// 111, 127, 128, 255, 256, 511, 512, 1_023, and 1_024, at about 1.5% each.
///
/// As an index, it reaches the ends of a slice and the round positions where
/// SIMD chunks start, without knowing the vector width:
///
/// ```zig
/// // 0, the last index, 8, 16, 32, 12, 24, 100, and so on.
/// const value_index = prng.intLiteralBetween(usize, 0, values.len - 1);
/// ```
///
/// For the values in between, add `intBetween` as another kind in the case's
/// weights, like `intAny` in the file header. `T` must have at most 64 bits,
/// and `min` must not exceed `max`.
pub fn intLiteralBetween(prng: *Prng, comptime T: type, min: T, max: T) T {
    comptime assert(@bitSizeOf(T) <= 64);
    assert(min <= max);

    return literalBetween(prng, T, min, max);
}

/// `literalBetween` draws the literals of `intLiteralBetween` and
/// `floatLiteralBetween`, whose documentation describes the draw. `T` is an
/// integer or float type that those functions accept.
fn literalBetween(prng: *Prng, comptime T: type, min: T, max: T) T {
    // A range of one value leaves nothing to draw, and every other range has
    // magnitudes to spread.
    if (min == max) {
        return min;
    }

    const is_float = @typeInfo(T) == .float;

    // The math runs in a type that holds every value of `T` and every shaped
    // value, and each result is cast back to `T`.
    const Wide = if (is_float) f64 else i128;

    const min_wide: Wide = min;
    const max_wide: Wide = max;
    const magnitude_max = @max(@abs(min_wide), @abs(max_wide));

    const base: u8 = if (prng.boolean()) 2 else 10;

    // An integer has at most as many digits as the range's largest magnitude.
    // A float keeps at most as many digits as any `T` round-trips through:
    // its precision in binary, and 5, 9, or 17 decimal digits for f16, f32,
    // or f64.
    const decimal_digits_count_max: u8 = switch (T) {
        f16 => 5,
        f32 => 9,
        else => 17,
    };

    const digits_count_max: u8 = if (!is_float)
        countDigits(magnitude_max, base)
    else if (base == 2)
        std.math.floatFractionalBits(T) + 1
    else
        decimal_digits_count_max;

    // The count of kept digits sets how many digits the result takes to
    // write. 0 digits means `min`, `max`, or 0, and keeping every digit of a
    // number leaves the number itself, the noisy end. The zero rule of
    // `intLogUniform` makes 0 digits as common as 1, and each further digit
    // rarer.
    var kept_digits_count = prng.intLogUniform(u8, 0, digits_count_max);
    if (kept_digits_count == 0) {
        // 0 digits name a value instead of writing one: `min`, `max`, or 0 if
        // it lies strictly between them, with equal odds. 1 and -1 are left
        // out: they are the smallest round numbers, which come up often
        // anyway, and naming them here too would give them twice the share of
        // `min`, `max`, and 0.
        const named_values = [_]T{ min, max, 0 };
        const named_values_count: usize = if (min < 0 and 0 < max) 3 else 2;

        return prng.pick(T, named_values[0..named_values_count]);
    }

    // Integers start at 1. Floats start at 2^-p, where p is the precision of
    // `T` in bits and 1 + 2^-p rounds back to 1, and a range within (-1, 1)
    // scales that floor down with its largest magnitude.
    //
    // Two limits keep the float floor usable in f64. It never drops below the
    // smallest f64, 4.9e-324, because in a range such as [0, 1e-310] it would
    // otherwise round to 0. It also stays within 2^1000 of the largest
    // magnitude, because `floatLogUniform` needs their ratio to be finite,
    // which a floor of 2^-53 breaks for [-1e300, 1e300].
    const magnitude_floor: Wide = if (is_float)
        @max(
            std.math.ldexp(@min(magnitude_max, 1), -(std.math.floatFractionalBits(T) + 1)),
            std.math.ldexp(magnitude_max, -1000),
            std.math.floatTrueMin(f64),
        )
    else
        1;

    const drawn = signedLogUniform(prng, Wide, min_wide, max_wide, magnitude_floor);

    // Zeros after the kept digits give numbers such as 32 or 1_200, and
    // repeats give numbers such as 63 (111111 in binary) or 999. Both take
    // about as little to write, so they come up with equal odds.
    const fill = prng.enumTag(Fill);

    // Keeping few digits can leave the range, as 37 → 32 does for
    // [37, 1_234], and each further digit brings the result closer to
    // `drawn`. The range check runs before the cast to `T`: a float in
    // [`min`, `max`] can't round past bounds that are `T` values themselves.
    while (kept_digits_count <= digits_count_max) : (kept_digits_count += 1) {
        const shaped = if (is_float)
            shapeFloatLeadingDigits(drawn, base, digits_count_max, kept_digits_count, fill)
        else
            shapeLeadingDigits(drawn, base, kept_digits_count, fill);

        if (min_wide <= shaped and shaped <= max_wide) {
            return std.math.lossyCast(T, shaped);
        }
    }

    // An integer with all its digits kept is `drawn` itself, so the loop
    // always returns for integers. A float's decimal digits are cut off after
    // the last kept one, which can still land just past a bound, so `drawn`
    // itself ends the ladder.
    return std.math.lossyCast(T, drawn);
}

/// `signedLogUniform` returns a nonzero `T`, an i128 or an f64, in [`min`,
/// `max`], whose magnitude has every order of magnitude from
/// `magnitude_floor` up equally likely. It never returns 0, because the named
/// values of `literalBetween` already give 0 its share. For a range that
/// crosses 0, it picks the negative or the positive side in proportion to how
/// much of the log scale each side covers, so the side with only -1 in
/// [-1, 1_000] doesn't get half the draws.
///
/// `min` must be below `max`, and `magnitude_floor` must be positive and at
/// most the range's largest magnitude.
fn signedLogUniform(prng: *Prng, comptime T: type, min: T, max: T, magnitude_floor: T) T {
    assert(min < max);

    // A side of 0 gets draws only if it reaches the floor. The side with the
    // largest magnitude always does.
    const has_positive_side = max >= magnitude_floor;
    const has_negative_side = -min >= magnitude_floor;

    var is_negative = !has_positive_side;

    if (has_positive_side and has_negative_side) {
        // The range holds 0, so both sides start at the floor.
        const positive_log_width = logWidth(T, magnitude_floor, max);
        const negative_log_width = logWidth(T, magnitude_floor, -min);

        const total_log_width = positive_log_width + negative_log_width;

        is_negative = prng.floatBetween(f64, 0, total_log_width) < negative_log_width;
    }

    // The magnitudes run from the floor, or from the range's inner bound if
    // that lies beyond it, up to the range's outer bound.
    const inner_bound = if (is_negative)
        -max
    else
        min;

    const outer_bound = if (is_negative)
        -min
    else
        max;

    const magnitude_min = @max(magnitude_floor, inner_bound);

    const magnitude: T = if (@typeInfo(T) == .int)
        prng.intLogUniform(u64, @intCast(magnitude_min), @intCast(outer_bound))
    else
        prng.floatLogUniform(T, magnitude_min, outer_bound);

    return if (is_negative)
        -magnitude
    else
        magnitude;
}

/// `logWidth` returns the width of the log scale that `intLogUniform` or
/// `floatLogUniform` spreads [`min`, `max`] over: ln((max + 1) / min) for an
/// i128, and ln(max / min) for an f64.
fn logWidth(comptime T: type, min: T, max: T) f64 {
    return if (@typeInfo(T) == .int)
        @log(@as(f64, @floatFromInt(max + 1)) / @as(f64, @floatFromInt(min)))
    else
        @log(max / min);
}

/// `Fill` names what replaces the digits after the kept ones in
/// `shapeLeadingDigits`.
const Fill = enum { zeros, repeats };

/// `shapeLeadingDigits` keeps the first `kept_digits_count` digits of `value`
/// in `base` and replaces the digits after them as `fill` says. With 2
/// decimal digits, 1_234 becomes 1_200 with zeros and 1_212 with repeats; with
/// 1 binary digit, 37 (100101) becomes 32 (100000) and 63 (111111). Zeros
/// round to the nearest such number, which can add a digit: 97 with 1 decimal
/// digit becomes 100. `value` itself comes back when `kept_digits_count`
/// covers all its digits.
fn shapeLeadingDigits(value: i128, base: u8, kept_digits_count: u8, fill: Fill) i128 {
    const magnitude = @abs(value);
    const magnitude_digits_count = countDigits(magnitude, base);
    if (kept_digits_count >= magnitude_digits_count) {
        return value;
    }

    // `kept_place_value` is the place value of the lowest kept digit: 100 for
    // 1_234 with 2 decimal digits.
    const kept_place_value = std.math.pow(u128, base, magnitude_digits_count - kept_digits_count);

    const shaped_magnitude = switch (fill) {
        .zeros => (magnitude + kept_place_value / 2) / kept_place_value * kept_place_value,

        .repeats => repeats: {
            // The loop appends copies of the kept digits until they cover
            // every digit of `value`, and the division cuts off the digits
            // past the last one: for 12_345 and 3 digits, 123 becomes 123_123
            // and then 12_312.
            const block = magnitude / kept_place_value;
            const block_place_value = std.math.pow(u128, base, kept_digits_count);

            var repeated = block;
            var repeated_digits_count = kept_digits_count;

            while (repeated_digits_count < magnitude_digits_count) {
                repeated = repeated * block_place_value + block;
                repeated_digits_count += kept_digits_count;
            }

            const extra_digits_count = repeated_digits_count - magnitude_digits_count;

            break :repeats repeated / std.math.pow(u128, base, extra_digits_count);
        },
    };

    const shaped: i128 = @intCast(shaped_magnitude);

    return if (value < 0)
        -shaped
    else
        shaped;
}

/// `countDigits` returns how many digits `magnitude` has in `base`, counting 0
/// as one digit.
fn countDigits(magnitude: u128, base: u8) u8 {
    var remaining = magnitude;
    var digits_count: u8 = 1;

    while (remaining >= base) {
        remaining /= base;
        digits_count += 1;
    }

    return digits_count;
}

// ─── Probabilities ──────────────────────────────────────────────────────────

/// `boolean` returns true or false, each half the time.
pub fn boolean(prng: *Prng) bool {
    return prng.generator.random().boolean();
}

/// `chance` returns true with probability `numerator` / `denominator`, so
/// `prng.chance(1, 16)` returns true in about 1 draw of 16. `denominator` must
/// be positive, and `numerator` must not exceed it.
pub fn chance(prng: *Prng, numerator: u64, denominator: u64) bool {
    assert(denominator > 0);
    assert(numerator <= denominator);

    return prng.intBetween(u64, 0, denominator - 1) < numerator;
}

// ─── Enums ──────────────────────────────────────────────────────────────────

/// `enumTag` returns a random tag of `E`, with every tag equally likely.
pub fn enumTag(prng: *Prng, comptime E: type) E {
    // A `u64` position keeps the draw the same on every target, as in `index`.
    return prng.generator.random().enumValueWithIndex(E, u64);
}

/// `EnumWeightsType` returns a struct type with one `u64` weight field per tag
/// of `E`, named after the tag. The fields have no defaults, so a weights
/// literal must give the weight of every tag.
pub fn EnumWeightsType(comptime E: type) type {
    return std.enums.EnumFieldStruct(E, u64, null);
}

/// `enumWeighted` returns a random tag of `E`, with probability proportional
/// to its weight. At least one weight must be positive, and the sum of the
/// weights must fit in a `u64`.
///
/// ```zig
/// const Operation = enum { insert, remove, lookup };
///
/// // `operation` is `.insert` with probability 4/6, and `.remove` or
/// // `.lookup` with probability 1/6 each.
/// const operation = prng.enumWeighted(Operation, .{ .insert = 4, .remove = 1, .lookup = 1 });
/// ```
pub fn enumWeighted(prng: *Prng, comptime E: type, weights: EnumWeightsType(E)) E {
    const tags = comptime std.enums.values(E);

    // The loop stores each tag's weight at the tag's position in `tags`, so
    // the position that `weightedIndex` returns in `tag_weights` also selects
    // the tag in `tags`.
    var tag_weights: [tags.len]u64 = undefined;
    inline for (&tag_weights, tags) |*tag_weight, tag| {
        tag_weight.* = @field(weights, @tagName(tag));
    }

    const tag_index = prng.generator.random().weightedIndex(u64, &tag_weights);

    return tags[tag_index];
}

/// `enumSet` returns a random subset of `allowed` with at least one tag in it.
/// Every size is equally likely, and no tag is favored: with all 5 tags in
/// `allowed`, 1 in 5 draws returns a single tag, and 1 in 5 returns all five.
///
/// ```zig
/// const Operation = enum { insert, remove, lookup, flush, restart };
///
/// // For example {flush}, {insert, remove, lookup}, or all five.
/// const operations = prng.enumSet(Operation, .full);
///
/// // {insert}, {remove}, or {insert, remove}.
/// const writes = prng.enumSet(Operation, .initMany(&.{ .insert, .remove }));
/// ```
///
/// A test can call `enumSet` once per case to decide which tags that case may
/// use, and then pass the set to `enumWeights`. `allowed` must not be empty.
pub fn enumSet(prng: *Prng, comptime E: type, allowed: std.EnumSet(E)) std.EnumSet(E) {
    assert(allowed.count() > 0);

    // The allowed tags in a random order, cut after a random count, give
    // every count the same odds, and every group of that many tags the same
    // odds too.
    var tags_count: usize = 0;
    var iterator = allowed.iterator();

    var tags_storage: [std.EnumSet(E).len]E = undefined;
    while (iterator.next()) |tag| : (tags_count += 1) {
        tags_storage[tags_count] = tag;
    }

    const tags = tags_storage[0..tags_count];

    prng.shuffle(E, tags);

    var subset: std.EnumSet(E) = .empty;

    for (tags[0..prng.intBetween(usize, 1, tags.len)]) |tag| {
        subset.insert(tag);
    }

    return subset;
}

/// `enumWeights` returns random weights for `enumWeighted`. It picks some of
/// the tags in `allowed`, the same way `enumSet` does, and gives each of them a
/// weight from 1 to 1000. Every other tag gets 0, so `enumWeighted` never
/// returns it:
///
/// ```zig
/// const Operation = enum { insert, remove, lookup, flush, restart };
///
/// // For example `.{ .insert = 900, .remove = 10, .lookup = 0, .flush = 0,
/// // .restart = 90 }`. `enumWeighted` then returns `insert` 90% of the time,
/// // `restart` 9%, `remove` 1%, and never `lookup` or `flush`.
/// const weights = prng.enumWeights(Operation, .full);
/// ```
///
/// The weights come from `intLiteralBetween(u64, 1, 1000)`: 1 and 1000 come up
/// most, at about 21% and 13%, then weights that take few digits, such as 2,
/// 3, 8, 10, 64, 100, or 999. For two tags, one outweighs the other by ten
/// times or more about 55% of the time, against 9% if the weights came evenly
/// from 1 to 100, as in TigerBeetle, so a rare tag often fires once among many
/// common ones. The two weights are also exactly equal about 7% of the time,
/// so the two tags come up equally often, against 1% for weights drawn evenly
/// from 1 to 100.
///
/// Draw new weights for each case, as swarm testing does. Some cases then use
/// almost only one tag, and others never use some tags at all, while fixed
/// weights would give every case the same mix. Three ways to draw them:
///
/// One mix per case: draw the weights once, and use them for every step.
///
/// ```zig
/// const weights = prng.enumWeights(Operation, .full);
///
/// for (0..4096) |_| {
///     const operation = prng.enumWeighted(Operation, weights);
///     // ...
/// }
/// ```
///
/// A fixed limit: leave tags out of `allowed`. Here, no case ever restarts.
///
/// ```zig
/// const weights = prng.enumWeights(Operation, .initMany(&.{ .insert, .remove, .lookup, .flush }));
/// ```
///
/// Phases: pick the allowed tags once per case with `enumSet`, then draw new
/// weights from them at the start of each phase. A case can then go from
/// mostly inserts to mostly removes to a mix of both. If `allowed` leaves out
/// `restart`, that case never restarts, in any phase. Drawing the mean
/// phase length once per case gives some cases many short phases and others
/// a few long ones.
///
/// ```zig
/// const allowed = prng.enumSet(Operation, .full);
/// const phase_mean = prng.intLogUniform(usize, 16, 4_096);
///
/// var weights = prng.enumWeights(Operation, allowed);
/// var phase_end_index = 1 + prng.intExponential(usize, phase_mean);
///
/// for (0..4096) |step_index| {
///     if (step_index == phase_end_index) {
///         weights = prng.enumWeights(Operation, allowed);
///         phase_end_index += 1 + prng.intExponential(usize, phase_mean);
///     }
///
///     const operation = prng.enumWeighted(Operation, weights);
///     // ...
/// }
/// ```
///
/// `allowed` must not be empty.
pub fn enumWeights(prng: *Prng, comptime E: type, allowed: std.EnumSet(E)) EnumWeightsType(E) {
    const enabled_tags = prng.enumSet(E, allowed);

    var weights: EnumWeightsType(E) = undefined;
    inline for (comptime std.enums.values(E)) |tag| {
        if (enabled_tags.contains(tag)) {
            @field(weights, @tagName(tag)) = prng.intLiteralBetween(u64, 1, 1000);
        } else {
            @field(weights, @tagName(tag)) = 0;
        }
    }

    return weights;
}

// ─── Slices ─────────────────────────────────────────────────────────────────

/// `index` returns a random index in [0, `items.len` - 1], with every index
/// equally likely. `items` is a slice or array and must not be empty. Use
/// `index` when the position matters rather than the element:
///
/// ```zig
/// // Put a zero at a random position of `values`, first and last included.
/// const value_index = prng.index(values);
/// values[value_index] = 0;
/// ```
pub fn index(prng: *Prng, items: anytype) usize {
    // A `u64` draw is the same on 32-bit and 64-bit targets, as the file
    // header promises; a `usize` draw is not.
    return @intCast(prng.generator.random().uintLessThan(u64, items.len));
}

/// `indexEdge` returns an index at an end of `items`: the first, the
/// second, the second-to-last, or the last, each equally likely:
///
/// ```text
/// index:    0    1    2    3    4    5    6    7    8    9
///         ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
/// values: │ a  │ b  │ c  │ d  │ e  │ f  │ g  │ h  │ i  │ j  │
///         └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
///           ^    ^                                  ^    ^
///         first second                     second-to-last last
///           ¼    ¼                                  ¼    ¼
/// ```
///
/// Loops that start one late or stop one early miss exactly these positions,
/// and `index` rarely returns them in a long slice. Make `indexEdge` and
/// `index` two kinds in the case's weights, as the file header shows:
///
/// ```zig
/// const IndexKind = enum { edge, any };
///
/// // Once per case.
/// const index_kind_weights = prng.enumWeights(IndexKind, .full);
///
/// // Per draw.
/// const value_index = switch (prng.enumWeighted(IndexKind, index_kind_weights)) {
///     .edge => prng.indexEdge(values),
///     .any => prng.index(values),
/// };
///
/// values[value_index] = std.math.nan(f32);
/// ```
///
/// In slices shorter than 4 items the ends overlap: for 3 items, index 1 comes
/// up half the time. `items` must not be empty.
pub fn indexEdge(prng: *Prng, items: anytype) usize {
    assert(items.len > 0);

    // `edge_distance` counts from the chosen end: 0 is the first or last
    // index, and 1 the second or second-to-last. A one-item slice has only
    // edge distance 0.
    const edge_distance = prng.intBetween(usize, 0, @min(1, items.len - 1));

    return if (prng.boolean())
        edge_distance
    else
        items.len - 1 - edge_distance;
}

/// `pick` returns a random element of `items`, which must not be empty. Every
/// position in `items` is equally likely:
///
/// ```zig
/// // `buffer_size` is 512, 4_096, or 65_536, each in about a third of the
/// // draws.
/// const buffer_size = prng.pick(usize, &.{ 512, 4_096, 65_536 });
/// ```
pub fn pick(prng: *Prng, comptime T: type, items: []const T) T {
    return items[prng.index(items)];
}

/// `shuffle` puts `items` in a random order, with every order equally likely:
///
/// ```zig
/// // Put zeros at three distinct random positions of the eight-element array
/// // `values`.
/// var value_indexes = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7 };
///
/// prng.shuffle(usize, &value_indexes);
/// for (value_indexes[0..3]) |value_index| {
///     values[value_index] = 0;
/// }
/// ```
pub fn shuffle(prng: *Prng, comptime T: type, items: []T) void {
    // `u64` positions keep the draws the same on every target, as in `index`.
    prng.generator.random().shuffleWithIndex(T, items, u64);
}

/// `Pattern` names the arrangements that `fillPattern` can give a slice.
/// Each one arranges values from the test's pool; none draws values of its
/// own.
pub const Pattern = enum {
    /// Every item gets the same pool value.
    constant,

    /// Every item gets one pool value, the fill value, except for the
    /// outliers: 1 item in slices of up to 7 items, and 1 up to a quarter of
    /// the items in longer ones, with a single outlier the most common case.
    /// Each outlier takes its value from another pool position than the fill
    /// value. Positions come from `intLiteralBetween` over the slice's
    /// indexes, so the first and last item and round positions such as 8, 16,
    /// and 32 get outliers most often.
    mostly_constant,

    /// The items cycle through the first p pool values, for a p from 2 to the
    /// pool size, starting at a random point in the cycle.
    periodic,

    /// The items form runs of one pool value each, and each run takes its
    /// value from another pool position than the run before it. Run lengths
    /// come from `intLiteralBetween`, so single items, runs over the rest of
    /// the slice, and round lengths such as 8 or 16 come up most. The first
    /// run ends before the last item, so a slice of 2 items or more has at
    /// least 2 runs.
    runs,

    /// Every item gets a random pool value, so a slice has few distinct values
    /// and ties everywhere.
    scattered,
};

/// `fillPattern` fills `items` with values from `pool`, arranged in
/// `pattern`. The test draws `pool` the way it draws its other values;
/// `fillPattern` only decides which item gets which pool value. With the pool
/// { 0, 7, -∞, 1.5 } and 16 items:
///
/// ```text
/// constant         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
/// mostly_constant  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  7
/// periodic         0  7 -∞  0  7 -∞  0  7 -∞  0  7 -∞  0  7 -∞  0
/// runs             0  0  0  0  0 -∞ -∞ -∞ -∞ -∞ -∞ -∞ -∞ -∞  7  7
/// scattered       -∞  7  0 1.5  0 -∞  7  7 1.5  0 -∞ 1.5  7  0 -∞ 1.5
/// ```
///
/// Independent draws almost never give such slices: every item equal, a
/// single odd item in the tail, or the same maximum in two places. Code that
/// works through a slice in chunks, such as SIMD kernels, breaks on exactly
/// these: two lanes that tie, or an odd item after the last full chunk.
///
/// A test typically patterns some slices and fills others independently.
/// When it fills many slices per case, such as the rows of a matrix, it draws
/// weights for both choices, independent or patterned and which pattern, once
/// per case, as the file header explains. Some cases then consist only of
/// independent rows, some only of patterned rows, and others of a mix. Here,
/// each pool holds 1 to 8 values, drawn the same way as the independent
/// values:
///
/// ```zig
/// const Arrangement = enum { independent, patterned };
///
/// // Once per case.
/// const arrangement_weights = prng.enumWeights(Arrangement, .full);
/// const pattern_weights = prng.enumWeights(Prng.Pattern, .full);
///
/// for (rows) |row| {
///     switch (prng.enumWeighted(Arrangement, arrangement_weights)) {
///         .independent => {
///             for (row) |*value| value.* = drawValue(&prng);
///         },
///
///         .patterned => {
///             var pool_storage: [8]f32 = undefined;
///             const pool = pool_storage[0..prng.intLogUniform(usize, 1, pool_storage.len)];
///
///             for (pool) |*value| value.* = drawValue(&prng);
///
///             prng.fillPattern(f32, row, prng.enumWeighted(Prng.Pattern, pattern_weights), pool);
///         },
///     }
/// }
/// ```
///
/// Keep the independent slices: the patterns are a guess at what matters, and
/// independent values cover what the guess misses. The pool decides what the
/// values look like. A pool of one value gives a constant slice in every
/// pattern, and a pool drawn around one value with `nudge` gives near-ties
/// everywhere.
///
/// With one slice per case, draw the choices with `boolean` and `enumTag`
/// instead: weights drawn for a single pick leave every choice equally likely
/// on average, so they add nothing. To leave out patterns that mean nothing
/// for a kernel, pass `enumWeights` a smaller set, such as
/// `.initMany(&.{ .constant, .mostly_constant, .runs })`, or with one slice
/// per case, draw from such a set with
/// `prng.pick(Prng.Pattern, &.{ .constant, .mostly_constant, .runs })`.
///
/// `pool` must not be empty and must not overlap `items`. With an empty
/// `items`, `fillPattern` does nothing.
pub fn fillPattern(
    prng: *Prng,
    comptime T: type,
    items: []T,
    pattern: Pattern,
    pool: []const T,
) void {
    assert(pool.len > 0);

    // An empty slice has no positions to arrange, and `mostly_constant` needs at
    // least one item to pick a position from.
    if (items.len == 0) {
        return;
    }

    switch (pattern) {
        .constant => {
            // A random pool value rather than always `pool[0]`, so that a
            // hand-picked pool such as { 0, NaN } also gives slices of only
            // NaN.
            @memset(items, prng.pick(T, pool));
        },

        .mostly_constant => {
            // The fill value goes everywhere first; the outliers then replace
            // a few items.
            const fill_pool_index = prng.index(pool);

            @memset(items, pool[fill_pool_index]);

            // The log-uniform count makes a single outlier the most common
            // case, while 2, 3, or 10 outliers still come up. Capping the count
            // at a quarter of the items keeps the fill value the clear
            // majority in slices of 8 items or more.
            const outlier_count = prng.intLogUniform(usize, 1, @max(1, items.len / 4));

            for (0..outlier_count) |_| {
                // Positions that take little to write come up most: the first
                // and last item, which loops that start one late or stop one
                // early miss, and round positions such as 8, 16, 31, or 32,
                // where SIMD chunks start and end.
                const item_index = prng.intLiteralBetween(usize, 0, items.len - 1);

                // An outlier takes its value from another pool position than
                // the fill value, so with a pool of distinct values, a slice
                // of 2 items or more always has an odd item. Two outliers can
                // land on the same item, which leaves fewer odd items than
                // `outlier_count`; drawing positions again would add a loop
                // for little gain.
                items[item_index] = pool[prng.indexOtherThan(pool, fill_pool_index)];
            }
        },

        .periodic => {
            // The period is at least 2, so neighboring items differ, except
            // with a pool of one value, where the cycle is that value alone.
            const period = prng.intBetween(usize, @min(2, pool.len), pool.len);

            // A random start in the cycle keeps it from always lining up with
            // the first item, and so with the first lane of every vector.
            const phase = prng.intBetween(usize, 0, period - 1);

            for (items, 0..) |*item, item_index| {
                item.* = pool[(item_index + phase) % period];
            }
        },

        .runs => {
            // Run lengths that take little to write come up most: a single
            // item, the rest of the slice, and round lengths such as 8, 16, or
            // 63, so runs often end at chunk boundaries. The first run ends
            // before the last item, and each run takes its value from another
            // pool position than the run before it, so with a pool of
            // distinct values, a slice of 2 items or more is never constant.
            var run_start_index: usize = 0;
            var pool_index = prng.index(pool);

            while (run_start_index < items.len) {
                const run_item_count_max = if (run_start_index == 0 and items.len > 1)
                    items.len - 1
                else
                    items.len - run_start_index;

                const run_item_count = prng.intLiteralBetween(usize, 1, run_item_count_max);

                @memset(items[run_start_index..][0..run_item_count], pool[pool_index]);

                run_start_index += run_item_count;
                pool_index = prng.indexOtherThan(pool, pool_index);
            }
        },

        .scattered => {
            for (items) |*item| {
                item.* = prng.pick(T, pool);
            }
        },
    }
}

/// `indexOtherThan` returns a random index of `items` other than
/// `excluded_index`, with every other index equally likely, or
/// `excluded_index` itself if `items` holds one item.
fn indexOtherThan(prng: *Prng, items: anytype, excluded_index: usize) usize {
    if (items.len == 1) {
        return excluded_index;
    }

    // Counting 1 to `items.len - 1` places on from `excluded_index`, and
    // wrapping around at the end, reaches every other index once.
    return (excluded_index + 1 + prng.intBetween(usize, 0, items.len - 2)) % items.len;
}

// ─── Floats ─────────────────────────────────────────────────────────────────

/// `floatAny` returns an `F` whose bits are all random. Every exponent is
/// equally likely, so tiny and huge magnitudes come up as often as magnitudes
/// near 1. For f32, NaNs come up in about 1 draw of 256, and so do subnormals.
/// Each infinity and each zero is a single bit pattern, so they practically
/// never come up. For functions that must accept every `F`, make `floatAny`
/// one kind in the case's weights, as the file header shows, next to
/// `floatEdge` and `floatLiteral`, which reach the values that random bits
/// miss.
///
/// `F` must not be f80. f80 stores the leading bit of the significand
/// explicitly, so about half of all random bit patterns are invalid numbers,
/// called unnormals, that arithmetic turns into NaN.
pub fn floatAny(prng: *Prng, comptime F: type) F {
    comptime assert(@typeInfo(F).float.bits != 80);

    return @bitCast(prng.intAny(std.meta.Int(.unsigned, @bitSizeOf(F))));
}

/// `floatFinite` returns a random finite `F`. It draws like `floatAny` but
/// never returns NaN or infinity, and every finite exponent remains equally
/// likely. As for `floatAny`, `F` must not be f80.
pub fn floatFinite(prng: *Prng, comptime F: type) F {
    // Only an all-ones exponent encodes infinity or NaN, so the loop draws
    // again whenever `floatAny` returns one, which happens in about 1 draw of
    // 256 for f32. Replacing such a draw with one fixed finite value instead
    // would make that value far more common than any other.
    while (true) {
        const value = prng.floatAny(F);
        if (std.math.isFinite(value)) {
            return value;
        }
    }
}

/// `floatBetween` returns a random `F` in [`min`, `max`]. Intervals of equal
/// width within that range are equally likely to contain the result, so
/// results much smaller in magnitude than `max - min` are rare: for [-3, 5],
/// results in [-0.001, 0.001] come up in about 1 draw of 4_000. Before
/// rounding, every result lies below `max`, so `max` itself comes up only when
/// a result rounds up to it. `F` must be f16, f32, or f64, `min` must not
/// exceed `max`, and `max - min` must be finite. For f16, the difference is
/// computed in f32, so any two finite f16 values work.
pub fn floatBetween(prng: *Prng, comptime F: type, min: F, max: F) F {
    comptime assert(F == f16 or F == f32 or F == f64);

    // std can't draw an f16 fraction, so this draws in f32 and rounds to f16.
    // Rounding can't carry the result past `min` or `max`, because both are
    // f16 values themselves. The f32 call checks the bounds. Checking them
    // here in f16 would reject ranges such as [-40_000, 40_000], whose width
    // overflows f16.
    if (F == f16) {
        return @floatCast(prng.floatBetween(f32, min, max));
    }

    assert(min <= max);
    assert(std.math.isFinite(max - min));

    // `fraction` lies in [0, 1), and `lerp` computes
    // `min + (max - min) * fraction` with a single rounding, so the result
    // cannot pass `max`. For f32, rounding `max - min` up multiplies it by at
    // most 1 + 2^-24, `fraction` is at most 1 - 2^-24, and the product of the
    // two factors is below 1, so the exact value stays below `max`; f64 has
    // the same argument with 2^-53. The final rounding can reach `max` but
    // cannot pass it.
    //
    // `@min` repeats the bound so that the result stays in [`min`, `max`]
    // even if std changes how `float` or `lerp` computes its result.
    const fraction = prng.generator.random().float(F);
    const value = std.math.lerp(min, max, fraction);

    return @min(value, max);
}

/// `floatLogUniform` returns a random `F` in [`min`, `max`] with every order
/// of magnitude equally likely. For [0.001, 1000], results from 0.001 to 0.01,
/// from 0.01 to 0.1, and so on up to 100 to 1000 each come up in 1 draw of 6:
///
/// ```zig
/// const bound = prng.floatLogUniform(f32, 0.001, 1000);
/// ```
///
/// `floatBetween` over the same range would return a value below 1 in only
/// about 1 draw of 1_000.
///
/// `F` must be f16, f32, or f64, `min` must be above 0, and `min` must not
/// exceed `max`. `max / min` must be finite, which matters only for f64, such
/// as for [1e-320, 1e300]. For negative values, negate the result.
///
/// Unlike `intLogUniform`, `floatLogUniform` never returns 0. Between 0 and
/// any positive float lie endless orders of magnitude, such as 0.1, 0.01, and
/// 1e-10, so floats have no natural share for 0. To include 0, make it one more
/// kind in the case's weights, as the file header shows.
pub fn floatLogUniform(prng: *Prng, comptime F: type, min: F, max: F) F {
    comptime assert(F == f16 or F == f32 or F == f64);
    assert(min > 0);
    assert(min <= max);

    // The math runs in f64 for every `F`, which also covers f16, and the
    // result is rounded to `F` at the end.
    const min_wide: f64 = min;
    const max_wide: f64 = max;
    const width_ratio = (max_wide - min_wide) / min_wide;
    assert(std.math.isFinite(width_ratio));

    // e^u, with u drawn evenly from ln(min) to ln(max), gives every tenfold
    // range the same share. As in `intLogUniform`, the code draws
    // `log_ratio` = u - ln(min) and returns min + min * (e^log_ratio - 1),
    // which is the same value. That keeps the spread of a narrow range of
    // large numbers, whose logarithms can round to the same f64.
    const log_ratio = prng.floatBetween(f64, 0, std.math.log1p(width_ratio));
    const value: F = @floatCast(min_wide + min_wide * std.math.expm1(log_ratio));

    // Rounding can put the result just outside [`min`, `max`].
    return std.math.clamp(value, min, max);
}

/// `floatGaussian` returns a random `F` from the Gaussian, or normal,
/// distribution with mean `mean` and standard deviation `standard_deviation`,
/// so about two thirds of the draws lie within `standard_deviation` of
/// `mean`. `F` must be f16, f32, or f64.
pub fn floatGaussian(prng: *Prng, comptime F: type, mean: F, standard_deviation: F) F {
    comptime assert(F == f16 or F == f32 or F == f64);

    // std can't draw an f16 normal value, so this draws in f32 and rounds to
    // f16.
    if (F == f16) {
        return @floatCast(prng.floatGaussian(f32, mean, standard_deviation));
    }

    return mean + standard_deviation * prng.generator.random().floatNorm(F);
}

/// `floatEdge` returns a float at an edge, with a random sign. It picks one
/// of the groups of edges below with equal odds, then an edge from that group
/// with equal odds:
///
/// - Edges of `F`: 0, the smallest subnormal, the smallest normal, 1, the
///   largest finite value, infinity, and NaN.
/// - Edges of each narrower float type among f16, f32, and f64: its smallest
///   subnormal, smallest normal, and largest finite value, which survive a
///   cast to that type, and the two values where the cast starts to fail: the
///   largest that becomes 0 and the smallest that becomes infinity. f16 has
///   no narrower type, so it has no such group.
/// - Edges of integers: 2^7, 2^8, 2^15, 2^16, 2^31, 2^32, 2^63, and 2^64,
///   where `@intFromFloat` into an 8-, 16-, 32-, or 64-bit integer stops
///   fitting, and 2^11, 2^24, 2^53, and `F`'s own limit, above which f16,
///   f32, f64, or `F` can't hold every integer, so that `x + 1 == x`.
///
/// Values that `F` can't hold exactly are left out. For f32, the groups are:
///
/// - f32: 0, 1.4e-45, 1.2e-38, 1, 3.4e38, infinity, NaN
/// - f16: 6.0e-8, 6.1e-5, and 65_504, which a cast to f16 keeps, and 3.0e-8
///   and 65_520, which it turns into 0 and infinity
/// - integers: 2^7, 2^8, 2^11, 2^15, 2^16, 2^24, 2^31, 2^32, 2^53, 2^63, 2^64
///
/// For f16, whose largest value is 65_504, the integer group is only 2^7,
/// 2^8, 2^11, and 2^15.
///
/// The sign matters: -2^31 is the smallest `i32`, but 2^31 is one past the
/// largest, and 0 also comes up as -0. To probe both sides of an edge, pass
/// the result to `nudge`. `prng.nudge(f32, 65_520, 1)` returns 65_519.996,
/// which a cast to f16 rounds to 65_504, or 65_520 or 65_520.004, which the
/// cast turns into infinity.
///
/// The other float functions almost never return these values, apart from 0,
/// 1, and the powers of two that `floatLiteral` returns, so make `floatEdge`
/// one kind among them in the case's weights, as the file header shows. To
/// leave out infinity and NaN, use `floatEdgeFinite`.
pub fn floatEdge(prng: *Prng, comptime F: type) F {
    const own_edges = [_]F{
        0,
        std.math.floatTrueMin(F),
        std.math.floatMin(F),
        1,
        std.math.floatMax(F),
        std.math.inf(F),
        std.math.nan(F),
    };

    return prng.floatEdgeFrom(F, &own_edges);
}

/// `floatEdgeFinite` returns a finite float at an edge, with a random sign.
/// It draws like `floatEdge`, from the same groups with equal odds, but
/// leaves infinity and NaN out of the edges of `F`: 0, the smallest
/// subnormal, the smallest normal, 1, and the largest finite value remain.
/// The edges of narrower types and of integers are finite already.
///
/// Use it where infinity and NaN are a kind of their own, so that a case that
/// leaves that kind out gets none of them from its edges either:
///
/// ```zig
/// const InputKind = enum { edge, literal, non_finite };
///
/// // Once per case.
/// const input_kind_weights = prng.enumWeights(InputKind, .full);
///
/// // Per draw.
/// const input = switch (prng.enumWeighted(InputKind, input_kind_weights)) {
///     .edge => prng.floatEdgeFinite(f32),
///     .literal => prng.floatLiteral(f32),
///     .non_finite => prng.pick(f32, &.{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }),
/// };
/// ```
///
/// Drawing `floatEdge` again for each infinity or NaN would shift the odds:
/// for f32, the group of its own edges would drop from a third of the draws
/// to about 26%.
pub fn floatEdgeFinite(prng: *Prng, comptime F: type) F {
    const own_edges = [_]F{
        0,
        std.math.floatTrueMin(F),
        std.math.floatMin(F),
        1,
        std.math.floatMax(F),
    };

    return prng.floatEdgeFrom(F, &own_edges);
}

/// `floatEdgeFrom` picks one of the groups of edges with equal odds, then an
/// edge from that group with equal odds, and gives it a random sign. The
/// groups are `own_edges`, the edges of the narrower float types, and the
/// edges of integers, as the `floatEdge` doc describes.
fn floatEdgeFrom(prng: *Prng, comptime F: type, comptime own_edges: []const F) F {
    // `F` holds every edge of a narrower type exactly, including the two
    // midpoints below, because it has more bits of both significand and
    // exponent.
    const narrower_edges = comptime narrower: {
        var edges_count = 0;

        var edges: [15]F = undefined;
        for ([_]type{ f16, f32, f64 }) |narrower_type| {
            if (@bitSizeOf(narrower_type) < @bitSizeOf(F)) {
                const true_min: F = std.math.floatTrueMin(narrower_type);
                const max: F = std.math.floatMax(narrower_type);
                const max_exponent = std.math.floatExponentMax(narrower_type);
                const max_next_power_of_two: F = 1 << (max_exponent + 1);

                // A cast rounds a value exactly halfway between two floats to
                // the one whose significand is even. Half the smallest
                // subnormal lies halfway between 0, which is even, and that
                // subnormal, which is odd, so it becomes 0. The midpoint above
                // the largest finite value, which is odd, rounds up to the
                // next power of two, which is too large and becomes infinity.
                edges[edges_count..][0..5].* = .{
                    true_min / 2,
                    true_min,
                    std.math.floatMin(narrower_type),
                    max,
                    (max + max_next_power_of_two) / 2,
                };
                edges_count += 5;
            }
        }

        const result = edges[0..edges_count].*;

        break :narrower &result;
    };

    const precision = comptime std.math.floatFractionalBits(F) + 1;

    const integer_edges = comptime selectDistinctExact(F, &[_]f128{
        0x1p7,  0x1p8,  0x1p15, 0x1p16,
        0x1p31, 0x1p32, 0x1p63, 0x1p64,
        0x1p11, 0x1p24, 0x1p53, 1 << precision,
    });

    const groups: []const []const F = comptime if (narrower_edges.len == 0)
        &.{ own_edges, integer_edges }
    else
        &.{ own_edges, narrower_edges, integer_edges };

    const group = groups[prng.index(groups)];
    const magnitude = prng.pick(F, group);

    return if (prng.boolean())
        -magnitude
    else
        magnitude;
}

/// `selectDistinctExact` returns the candidates, `i128` or `f128` values, that
/// `T` holds exactly, each once and in their original order. It runs only at
/// compile time.
fn selectDistinctExact(comptime T: type, comptime candidates: anytype) []const T {
    comptime {
        var values_count: usize = 0;

        var values: [candidates.len]T = undefined;
        for (candidates) |candidate| {
            // A candidate that `T` can't hold comes out of the cast changed:
            // an integer saturates at the end of `T`, and a float rounds or
            // turns into infinity or 0.
            const value = std.math.lossyCast(T, candidate);
            const value_is_taken = std.mem.indexOfScalar(T, values[0..values_count], value) != null;

            if (value != candidate or value_is_taken) {
                continue;
            }

            values[values_count] = value;
            values_count += 1;
        }

        const result = values[0..values_count].*;

        return &result;
    }
}

/// `floatLiteral` returns a float that people often write as a literal in
/// code. It draws from [-2^p, 2^p] the way `floatLiteralBetween` draws from a
/// range, where p is the precision of `F` in bits, 24 for f32. Results are
/// 2^p, -2^p, 0, or a number made of a few digits, such as 1, 0.5, 2.5, 3,
/// 0.1, 1_000, or 0.333333333. For f32, 2^24, -2^24, and 0 come up about 7% of
/// the time each. The other results have every order of magnitude from 2^-24
/// to 2^24 about equally likely, so 1, 2, and 0.5 come up about 0.1% of the
/// time each, and 3, 0.1, and 1_000 about 0.05%.
///
/// Past 2^p, `F` no longer holds every integer, so that x + 1 == x, and
/// below 2^-p, 1 + x rounds back to 1. Code uses the numbers in between far
/// more often than random ones: counts converted to floats, rounding ties
/// such as 2.5, which `@round` turns into 3 and rounding to even into 2, scale
/// factors such as 0.5, and constants such as 0.1 and 1_000. `floatAny` and
/// `floatBetween` almost never return them, so draw them as one kind among
/// `floatEdge` and the noise functions in the case's weights, as the file
/// header shows. Pass the result to `nudge` to get the floats right next to 1
/// or 2^p.
///
/// `F` must be f16, f32, or f64.
pub fn floatLiteral(prng: *Prng, comptime F: type) F {
    const magnitude_max: F = 1 << (std.math.floatFractionalBits(F) + 1);

    return prng.floatLiteralBetween(F, -magnitude_max, magnitude_max);
}

/// `floatLiteralBetween` returns an `F` in [`min`, `max`] that takes little
/// to write down: `min`, `max`, or 0, or a number made of a few digits, in
/// binary, such as 1, 0.5, 48, or 1/3 (0.010101… in binary), or in decimal,
/// such as 0.1, 2.5, 40, or 0.333333333:
///
/// ```text
/// [-80, 80]       -80  80  0  1  -1  -64  -40  -0.5  0.25  2.5  12  0.001 …
/// [0.001, 1000]   0.001  1000  1  0.01  0.1  0.5  2  64  100  500 …
/// ```
///
/// It picks values the way `intLiteralBetween` does: binary or decimal with
/// equal odds, then how many leading digits to keep. 0 digits gives `min`,
/// `max`, or 0 and is as likely as 1 digit; each further digit is rarer. The
/// digits after the kept ones become zeros or repeats of the kept digits, with
/// equal odds, up to the precision of `F`. In f32, 37.18 with 1 decimal digit
/// becomes 40 or 33.3333333, and with 2 binary digits, 32 or 42.6666…, which
/// is 101010.1010… in binary. Decimal results are rounded to `F`, such as 0.1
/// to 0.100000001 in f32.
///
/// The numbers it shapes have every order of magnitude about equally likely,
/// from the largest magnitude in the range down to 2^-p, where p is the
/// precision of `F` in bits and 1 + 2^-p rounds back to 1: for f32, down to
/// 2^-24, about 6e-8. In a range within (-1, 1), that lower end shrinks with
/// the range: for [0, 0.001], it is 0.001 · 2^-24. In f64 ranges beyond about
/// 1e285, it rises to stay within 2^1000 of the largest magnitude.
///
/// For the values in between, add `floatBetween` as another kind in the
/// case's weights, like `intAny` in the file header. `F` must be f16, f32, or
/// f64, `min` must not exceed `max`, and both must be finite.
pub fn floatLiteralBetween(prng: *Prng, comptime F: type, min: F, max: F) F {
    comptime assert(F == f16 or F == f32 or F == f64);
    assert(min <= max);
    assert(std.math.isFinite(min) and std.math.isFinite(max));

    return literalBetween(prng, F, min, max);
}

/// `shapeFloatLeadingDigits` applies `shapeLeadingDigits` to the first
/// `significand_digits_count` significant digits of `value` in `base`: with 9
/// decimal digits, 1 of them kept, 37.18 becomes 40 with zeros and 33.3333333
/// with repeats. The result is the shaped number rounded to the nearest f64.
/// `value` must not be 0.
fn shapeFloatLeadingDigits(
    value: f64,
    base: u8,
    significand_digits_count: u8,
    kept_digits_count: u8,
    fill: Fill,
) f64 {
    assert(value != 0);

    // `value` is `significand` · `base`^`exponent`, where `significand` is a
    // whole number of `significand_digits_count` digits, cut off after the
    // last one: 37.18 with 9 decimal digits is 371_800_000 · 10^-7. Right
    // next to a power of ten, `log10` can be off by one, which leaves the
    // significand a digit longer or shorter and changes nothing else.
    const exponent: i32 = if (base == 2)
        std.math.frexp(value).exponent - significand_digits_count
    else
        @as(i32, @intFromFloat(@floor(@log10(@abs(value))))) + 1 - significand_digits_count;

    const significand: i128 = @intFromFloat(@trunc(scaleByPower(value, base, -exponent)));
    const shaped = shapeLeadingDigits(significand, base, kept_digits_count, fill);

    return scaleByPower(@floatFromInt(shaped), base, exponent);
}

/// `scaleByPower` returns `value` · `base`^`exponent` rounded to the nearest
/// f64, for a `base` of 2 or 10.
fn scaleByPower(value: f64, base: u8, exponent: i32) f64 {
    // 10^exponent is 5^exponent · 2^exponent. The power of five, the product,
    // and `ldexp` run in f128, whose significand has 60 bits more than f64's,
    // and the cast rounds once at the end. In f64, the power of five would
    // round from 5^23 on, and the product would round again, which turns 1e-7
    // into 1.0000000000000001e-7. Dividing by 5^-exponent instead of
    // multiplying by 5^exponent keeps 3 · 10^-1 at 0.3 for the same reason:
    // 5^1 is exact, and 5^-1 is not.
    const power_of_five: f128 = if (base == 10)
        powers_of_five[@abs(exponent)]
    else
        1;

    const value_wide: f128 = value;

    const scaled = if (exponent < 0)
        value_wide / power_of_five
    else
        value_wide * power_of_five;

    return @floatCast(std.math.ldexp(scaled, exponent));
}

// `powers_of_five` holds 5^0 through 5^341, enough for every decimal exponent
// that `shapeFloatLeadingDigits` needs: splitting the smallest f64, 4.9e-324,
// into 17 digits takes 10^340, and `log10` can be off by one. f128 holds the
// powers exactly up to 5^48, and the larger ones within about 2^-105 of their
// exact value.
const powers_of_five: [342]f128 = table: {
    var powers: [342]f128 = undefined;
    powers[0] = 1;

    for (1..powers.len) |exponent| {
        powers[exponent] = powers[exponent - 1] * 5;
    }

    break :table powers;
};

// ─── Boundaries ─────────────────────────────────────────────────────────────

/// `nudge` returns `value` moved by up to `step_count_max` steps in either
/// direction, including not at all, with every step count from
/// -`step_count_max` to `step_count_max` equally likely. A step is 1 for
/// integers; for floats, a step moves to the adjacent float:
///
/// ```zig
/// // 254 or 255, because 256 does not fit in a `u8`.
/// const byte = prng.nudge(u8, 255, 1);
///
/// // -2, -1, 0, 1, or 2.
/// const small = prng.nudge(i32, 0, 2);
///
/// // 0.99999994, 1.0, or 1.00000012.
/// const around_one = prng.nudge(f32, 1.0, 1);
/// ```
///
/// Use `nudge` to probe a boundary, such as the largest input a function
/// accepts, from both sides. When a step would move past the end of `T`'s
/// range, `nudge` returns that end instead: `std.math.maxInt(T)` or
/// `std.math.minInt(T)` for integers, and infinity or negative infinity for
/// floats. Near an end, the end therefore comes up more often than the other
/// values: `prng.nudge(u8, 255, 1)` returns 255 in 2 draws of 3, and
/// `prng.nudge(f32, std.math.inf(f32), 1)` returns infinity in 2 draws of 3.
/// A NaN stays NaN. For integers, `T` must have at most 64 bits.
pub fn nudge(prng: *Prng, comptime T: type, value: T, step_count_max: u16) T {
    // `neighbor_index` in [0, 2 * `step_count_max`] stands for the signed step
    // count `neighbor_index - step_count_max`, from -`step_count_max` to
    // `step_count_max`, so every step count is equally likely. `is_below` and
    // `step_count` split it into a direction and a number of steps, which
    // both branches below use.
    const neighbor_index = prng.intBetween(u32, 0, 2 * @as(u32, step_count_max));
    const is_below = neighbor_index < step_count_max;

    const step_count = if (is_below)
        step_count_max - neighbor_index
    else
        neighbor_index - step_count_max;

    switch (@typeInfo(T)) {
        .int => {
            // `nudge` adds the signed step count to `value` in `i128`, which
            // holds any value of an integer type of at most 64 bits plus at
            // most `maxInt(u16)` steps, and clamps only the sum to `T`'s
            // range. Clamping the step count to `T` before adding it would
            // lose neighbors that fit in `T`: from -128, a step count of 200
            // does not fit in an `i8`, yet the neighbor -128 + 200 = 72 does.
            comptime assert(@bitSizeOf(T) <= 64);

            const signed_step_count: i128 = if (is_below)
                -@as(i128, step_count)
            else
                step_count;

            const neighbor_wide = @as(i128, value) + signed_step_count;

            return @intCast(std.math.clamp(neighbor_wide, std.math.minInt(T), std.math.maxInt(T)));
        },

        .float => {
            // Each `nextAfter` call moves to the adjacent float toward
            // `direction`, so every step has the spacing of the floats at the
            // current value, whatever its magnitude. `nextAfter` returns an
            // infinity unchanged when it moves toward that same infinity, so a
            // walk that reaches an infinity stays there.
            const direction: T = if (is_below)
                -std.math.inf(T)
            else
                std.math.inf(T);

            var neighbor = value;

            for (0..step_count) |_| {
                neighbor = std.math.nextAfter(T, neighbor, direction);
            }

            return neighbor;
        },

        else => @compileError("`nudge` needs an integer or float type"),
    }
}

// The tests use a fixed seed, as TigerBeetle's own PRNG tests do. Some checks
// are statistical, and with a random seed they would fail by chance now and
// then. With the fixed seed, every run sees the same draws, so a failure comes
// from a change to `Prng` or to std's generator. Each test starts its own
// `Prng`, so changing one test leaves the draws of the others alone.
const test_seed = 92;

// A histogram that counts how often a condition holds watches only `true`.
const true_only = [_]bool{true};

const Histogram = @import("histogram.zig").Histogram;
const withBothSigns = @import("histogram.zig").withBothSigns;

test "Prng.intBetween draws the same values for every integer type" {
    // `intBetween` draws in 64 bits whatever `T` is, so draws from the same
    // seed agree across types. `u32` stands in for a 32-bit `usize` on 64-bit
    // hosts, and the 32-bit test run checks `usize` itself, as the file header
    // promises.
    var prng_u8 = Prng.init(test_seed);
    var prng_u32 = Prng.init(test_seed);
    var prng_usize = Prng.init(test_seed);
    var prng_u64 = Prng.init(test_seed);
    var prng_i8 = Prng.init(test_seed);
    var prng_i64 = Prng.init(test_seed);

    for (0..1_000) |_| {
        const value: u64 = prng_u8.intBetween(u8, 0, 7);

        try std.testing.expectEqual(value, prng_u32.intBetween(u32, 0, 7));
        try std.testing.expectEqual(value, prng_usize.intBetween(usize, 0, 7));
        try std.testing.expectEqual(value, prng_u64.intBetween(u64, 0, 7));

        const signed_value: i64 = prng_i8.intBetween(i8, -4, 3);

        try std.testing.expectEqual(signed_value, prng_i64.intBetween(i64, -4, 3));
    }
}

test "Prng.intBetween stays in range" {
    // The ends of the widest types, which a draw of the wrong signedness would
    // overflow.
    try expectDrawsBetween(Prng.intBetween, u64, std.math.maxInt(u64) - 1, std.math.maxInt(u64));
    try expectDrawsBetween(Prng.intBetween, u64, 0, std.math.maxInt(u64));
    try expectDrawsBetween(Prng.intBetween, i64, std.math.minInt(i64), std.math.minInt(i64) + 1);
    try expectDrawsBetween(Prng.intBetween, i8, std.math.minInt(i8), std.math.maxInt(i8));
}

test "Prng.intExponential averages about mean - 0.5" {
    var prng = Prng.init(test_seed);

    const draw_count = 100_000;
    var sum: u64 = 0;

    for (0..draw_count) |_| {
        sum += prng.intExponential(u64, 16);
    }

    // Rounding exponential draws with mean 16 down gives integers whose exact
    // mean is 1 / (e^(1/16) - 1) = 15.505. The mean of 100,000 draws has a
    // standard error of about 0.05, so the tolerance of 0.25 is about five
    // standard errors.
    const mean = @as(f64, @floatFromInt(sum)) / draw_count;

    try std.testing.expectApproxEqAbs(15.505, mean, 0.25);
}

test "Prng.intExponential puts most draws below mean" {
    // About 63% of the draws fall below `mean`, and about 5% land at or above
    // three times `mean`: 1 - e^-1 and e^-3, exact for a whole-number `mean`.
    // Over 100,000 draws, the two shares have standard errors of about 0.0015
    // and 0.0007, so the tolerances of 0.01 and 0.005 are about seven
    // standard errors.
    var prng = Prng.init(test_seed);

    var below_mean: Histogram(bool, &true_only) = .{};
    var three_means_or_more: Histogram(bool, &true_only) = .{};

    for (0..100_000) |_| {
        const value = prng.intExponential(u64, 16);

        below_mean.add(value < 16);
        three_means_or_more.add(value >= 48);
    }

    try below_mean.expectShare(true, 1 - @exp(-1.0), 0.01);
    try three_means_or_more.expectShare(true, @exp(-3.0), 0.005);
}

test "Prng.intExponential stops at maxInt(T)" {
    // With a `mean` of 200, every `u8` result that would reach 255 or more
    // comes out as 255, which happens in e^(-255/200) of the draws, about
    // 28%. Over 100,000 draws, that share has a standard error of about
    // 0.0014, so the tolerance of 0.01 is about seven standard errors.
    var prng = Prng.init(test_seed);

    const max_only = [_]u8{255};
    var max_histogram: Histogram(u8, &max_only) = .{};

    for (0..100_000) |_| {
        max_histogram.add(prng.intExponential(u8, 200));
    }

    try max_histogram.expectShare(255, @exp(-255.0 / 200.0), 0.01);
}

test "Prng.intLogUniform stays in range" {
    // One-value ranges, ranges from 0, and the whole range of `u64`, where
    // the f64 math rounds at both ends.
    try expectDrawsBetween(Prng.intLogUniform, u64, 0, 0);
    try expectDrawsBetween(Prng.intLogUniform, u64, 1, 1);
    try expectDrawsBetween(Prng.intLogUniform, u8, 7, 7);
    try expectDrawsBetween(Prng.intLogUniform, u8, 0, 3);
    try expectDrawsBetween(Prng.intLogUniform, u8, 1, 255);
    try expectDrawsBetween(Prng.intLogUniform, u64, 3, 1_000_000);
    try expectDrawsBetween(Prng.intLogUniform, u64, 0, std.math.maxInt(u64));
    try expectDrawsBetween(Prng.intLogUniform, u64, 1, std.math.maxInt(u64));
}

test "Prng.intLogUniform spreads evenly across orders of magnitude" {
    // 1 to 9, 10 to 99, and 100 to 999 each get a third of the draws from
    // [1, 999], and half of the draws over all of `u64` lie above 2^32. Over
    // 100,000 draws, each share has a standard error of about 0.0015, so the
    // tolerance of 0.01 is about seven standard errors.
    var prng = Prng.init(test_seed);

    const digit_counts = [_]u8{ 1, 2, 3 };
    var digit_count_histogram: Histogram(u8, &digit_counts) = .{};
    var above_2_32: Histogram(bool, &true_only) = .{};

    for (0..100_000) |_| {
        const digit_count: u8 = switch (prng.intLogUniform(u32, 1, 999)) {
            1...9 => 1,
            10...99 => 2,
            else => 3,
        };

        digit_count_histogram.add(digit_count);

        above_2_32.add(prng.intLogUniform(u64, 1, std.math.maxInt(u64)) > 1 << 32);
    }

    for (digit_counts) |digit_count| {
        try digit_count_histogram.expectShare(digit_count, 1.0 / 3.0, 0.01);
    }

    try above_2_32.expectShare(true, 0.5, 0.01);
}

test "Prng.intLogUniform gives small values their exact shares" {
    // Within one order of magnitude, rounding e^u down instead of to the
    // nearest integer matters. For [1, 3], the shares are
    // ln(2) / ln(4) = 0.5, ln(3/2) / ln(4) = 0.292, and
    // ln(4/3) / ln(4) = 0.208. Rounding to the nearest would give 1 a share of
    // ln(1.5) / ln(4) = 0.29 instead.
    //
    // 0 comes up as often as 1. For [0, 3], the log scale runs from ln(1/2)
    // to ln(4), a width of ln(8). 0 and 1 each get ln(2) / ln(8) = 1/3, 2
    // gets ln(3/2) / ln(8) = 0.195, and 3 gets ln(4/3) / ln(8) = 0.138.
    //
    // Over 100,000 draws, each share has a standard error of at most 0.0016,
    // so the tolerance of 0.01 is about six standard errors. A `min` of 1
    // must never return 0.
    var prng = Prng.init(test_seed);

    const small_values = [_]u8{ 0, 1, 2, 3 };
    var from_one_histogram: Histogram(u8, &small_values) = .{};
    var from_zero_histogram: Histogram(u8, &small_values) = .{};

    for (0..100_000) |_| {
        from_one_histogram.add(prng.intLogUniform(u8, 1, 3));
        from_zero_histogram.add(prng.intLogUniform(u8, 0, 3));
    }

    try from_one_histogram.expectShare(0, 0, 0);
    try from_one_histogram.expectShare(1, 0.5, 0.01);
    try from_one_histogram.expectShare(2, 0.292, 0.01);
    try from_one_histogram.expectShare(3, 0.208, 0.01);

    try from_zero_histogram.expectShare(0, 1.0 / 3.0, 0.01);
    try from_zero_histogram.expectShare(1, 1.0 / 3.0, 0.01);
    try from_zero_histogram.expectShare(2, 0.195, 0.01);
    try from_zero_histogram.expectShare(3, 0.138, 0.01);
}

test "Prng.intLogUniform keeps narrow ranges of large numbers" {
    // Every value of a narrow range far above 2^53 comes up, and nothing
    // else does. Each of the 101 values comes up in about 1 draw of 101, so
    // 20,000 draws miss one with probability about e^-198.
    var prng = Prng.init(test_seed);

    const range_min: u64 = 1 << 60;

    const range_values = comptime values: {
        var values: [101]u64 = undefined;
        for (&values, 0..) |*value, value_index| {
            value.* = range_min + value_index;
        }

        break :values values;
    };

    var range_histogram: Histogram(u64, &range_values) = .{};

    for (0..20_000) |_| {
        range_histogram.add(prng.intLogUniform(u64, range_min, range_min + 100));
    }

    try range_histogram.expectAllSeen();
    try range_histogram.expectNothingElse();
}

test "Prng.intEdge returns exactly its listed values" {
    const u8_edges = [_]u8{ 0, 1, 127, 255 };
    const i8_edges = [_]i8{ 0, 1, -1, -128, 127 };
    const u12_edges = [_]u12{ 0, 1, 127, 255, 4_095 };

    const i32_edges = [_]i32{
        0,      1,              -1,            -128, 127, 255, -32_768, 32_767,
        65_535, -2_147_483_648, 2_147_483_647,
    };

    const u64_edges = [_]u64{
        0,                         1,                          127,           255,
        32_767,                    65_535,                     2_147_483_647, 4_294_967_295,
        9_223_372_036_854_775_807, 18_446_744_073_709_551_615,
    };

    try expectDrawsExactly(Prng.intEdge, u8, &u8_edges);
    try expectDrawsExactly(Prng.intEdge, i8, &i8_edges);
    try expectDrawsExactly(Prng.intEdge, u12, &u12_edges);
    try expectDrawsExactly(Prng.intEdge, i32, &i32_edges);
    try expectDrawsExactly(Prng.intEdge, u64, &u64_edges);
}

test "Prng.intEdge gives every edge the same odds" {
    // Each of the 10 edges of `u64` comes up in about 1 draw of 10, although
    // 0 and the largest `u64` also appear among the candidates as the minimum
    // and maximum of `T`. Over 20,000 draws, a share of 1/10 has a standard
    // error of about 0.002, so the tolerance of 0.015 is about seven standard
    // errors.
    var prng = Prng.init(test_seed);

    const edges = [_]u64{
        0,                         1,                          127,           255,
        32_767,                    65_535,                     2_147_483_647, 4_294_967_295,
        9_223_372_036_854_775_807, 18_446_744_073_709_551_615,
    };

    var edge_histogram: Histogram(u64, &edges) = .{};

    for (0..20_000) |_| {
        edge_histogram.add(prng.intEdge(u64));
    }

    try edge_histogram.expectNothingElse();

    for (edges) |edge| {
        try edge_histogram.expectShare(edge, 0.1, 0.015);
    }
}

test "Prng.intLiteral reaches every value of small types" {
    // `u4` and `i4` are small enough to cover completely.
    const u4_values = [_]u4{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    const i4_values = [_]i4{ -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7 };

    try expectDrawsExactly(Prng.intLiteral, u4, &u4_values);
    try expectDrawsExactly(Prng.intLiteral, i4, &i4_values);
}

test "Prng.intLiteral returns the ends of the type and both signs" {
    // The minimum, the maximum, and 0 of `i64` come up, and about half the
    // nonzero `i32` results are negative. About 18,700 of the 20,000 draws
    // are nonzero, so the negative share has a standard error of about 0.004,
    // and the tolerance of 0.02 is about five standard errors.
    var prng = Prng.init(test_seed);

    const i64_named_values = [_]i64{ std.math.minInt(i64), std.math.maxInt(i64), 0 };
    var i64_histogram: Histogram(i64, &i64_named_values) = .{};
    var negative: Histogram(bool, &true_only) = .{};

    for (0..20_000) |_| {
        i64_histogram.add(prng.intLiteral(i64));

        const value = prng.intLiteral(i32);

        if (value != 0) {
            negative.add(value < 0);
        }
    }

    try i64_histogram.expectAllSeen();
    try negative.expectShare(true, 0.5, 0.02);
}

test "Prng.intLiteralBetween stays in range" {
    // One-value ranges, ranges on either side of 0 and across it, the whole
    // range of a type, and a narrow range of large numbers, where rounding to
    // few digits always leaves the range.
    try expectDrawsBetween(Prng.intLiteralBetween, u8, 5, 5);
    try expectDrawsBetween(Prng.intLiteralBetween, i32, 0, 0);
    try expectDrawsBetween(Prng.intLiteralBetween, i32, -80, 80);
    try expectDrawsBetween(Prng.intLiteralBetween, i32, -1_234, -37);
    try expectDrawsBetween(Prng.intLiteralBetween, i16, -1, 1_000);
    try expectDrawsBetween(Prng.intLiteralBetween, u64, 1 << 60, (1 << 60) + 100);
    try expectDrawsBetween(Prng.intLiteralBetween, i8, std.math.minInt(i8), std.math.maxInt(i8));
    try expectDrawsBetween(Prng.intLiteralBetween, i64, std.math.minInt(i64), std.math.maxInt(i64));
    try expectDrawsBetween(Prng.intLiteralBetween, u64, 0, std.math.maxInt(u64));
}

test "Prng.intLiteralBetween gives min, max, and 0 the top shares" {
    // The shares that the doc comment lists for [-80, 80]: about 11% each for
    // -80, 0, and 80, and about 5% each for -1 and 1, so that 0 and ±1 are
    // not counted twice. The bounds also come up for [37, 1_234], which holds
    // no 0.
    //
    // The shares come from the rules in the doc comment: 0 digits in about
    // 32% of the draws, split three ways, plus the round numbers that land on
    // the bounds. Over 100,000 draws, a share of 11% has a standard error of
    // about 0.001, so the tolerance of 0.015 leaves room for the rounding in
    // the listed shares.
    var prng = Prng.init(test_seed);

    const common_values = [_]i32{ -80, 0, 80, -1, 1 };
    const bounds = [_]u32{ 37, 1_234 };

    var common_histogram: Histogram(i32, &common_values) = .{};
    var bound_histogram: Histogram(u32, &bounds) = .{};

    for (0..100_000) |_| {
        common_histogram.add(prng.intLiteralBetween(i32, -80, 80));
        bound_histogram.add(prng.intLiteralBetween(u32, 37, 1_234));
    }

    try common_histogram.expectShare(-80, 0.11, 0.015);
    try common_histogram.expectShare(0, 0.11, 0.015);
    try common_histogram.expectShare(80, 0.11, 0.015);
    try common_histogram.expectShare(-1, 0.054, 0.015);
    try common_histogram.expectShare(1, 0.054, 0.015);
    try bound_histogram.expectAllSeen();
}

test "Prng.intLiteralBetween counts a bound of 0 once" {
    // For [0, 999], 0 is both `min` and a named value, yet it comes up as
    // often as 999, in about 14% of the draws each. Counting 0 twice would
    // give it two thirds of the named share, about 19%. Over 20,000 draws, a
    // share of 14% has a standard error of about 0.0025, so the tolerance of
    // 0.015 is about six standard errors.
    var prng = Prng.init(test_seed);

    const bounds = [_]usize{ 0, 999 };
    var bound_histogram: Histogram(usize, &bounds) = .{};

    for (0..20_000) |_| {
        bound_histogram.add(prng.intLiteralBetween(usize, 0, 999));
    }

    try bound_histogram.expectShare(0, 0.14, 0.015);
    try bound_histogram.expectShare(999, 0.14, 0.015);
}

test "Prng.intLiteralBetween weighs the sides of 0 by their orders of magnitude" {
    // For [-1, 1_000], the negative side holds only -1 and covers a tenth as
    // much of the log scale as the positive side, so -1 comes up in about
    // 15.5% of the draws: its share as `min`, plus a small share of the shaped
    // draws. A coin flip between the sides would give it about 46%. Over
    // 20,000 draws, a share of 15.5% has a standard error of about 0.0026, so
    // the tolerance of 0.02 is about eight standard errors.
    var prng = Prng.init(test_seed);

    const min_only = [_]i32{-1};
    var min_histogram: Histogram(i32, &min_only) = .{};

    for (0..20_000) |_| {
        min_histogram.add(prng.intLiteralBetween(i32, -1, 1_000));
    }

    try min_histogram.expectShare(-1, 0.155, 0.02);
}

test "Prng.intLiteralBetween favors few digits at every scale" {
    // Most results over [0, 1_000_000] consist of at most 2 leading digits in
    // binary or decimal followed by zeros or repeats, which `intBetween`
    // would almost never give, and results below 1_000 are common although
    // they make up only 0.1% of the range.
    //
    // Keeping at most 2 digits already makes up about half the draws, and
    // small starting numbers keep few digits even with more digits kept, so
    // about 74% of the results have few digits, and about 50% lie below
    // 1_000. Over 2,000 draws, a share has a standard error of about 0.011,
    // so both shares lie more than 20 standard errors above their
    // thresholds.
    var prng = Prng.init(test_seed);

    var few_digits: Histogram(bool, &true_only) = .{};
    var below_1_000: Histogram(bool, &true_only) = .{};

    for (0..2_000) |_| {
        const value = prng.intLiteralBetween(u32, 0, 1_000_000);

        few_digits.add(hasFewDigits(value, 2, 2) or hasFewDigits(value, 10, 2));
        below_1_000.add(value < 1_000);
    }

    try std.testing.expect(few_digits.share(true) > 0.5);
    try std.testing.expect(below_1_000.share(true) > 0.2);
}

test "Prng.intLiteralBetween reaches the ends and round positions of a row" {
    // As indexes into a row of 51_864 values, the results reach both ends and
    // the first chunk boundaries at 8, 16, and 32. The rarest of them, 32,
    // comes up in about 1 draw of 180, so 5,000 draws miss it with
    // probability about e^-28.
    var prng = Prng.init(test_seed);

    const round_indexes = [_]usize{ 0, 8, 16, 32, 51_863 };
    var index_histogram: Histogram(usize, &round_indexes) = .{};

    for (0..5_000) |_| {
        index_histogram.add(prng.intLiteralBetween(usize, 0, 51_863));
    }

    try index_histogram.expectAllSeen();
}

test "Prng.intLiteralBetween repeats digits about as often as it zeros them" {
    // Over all of `u16`, 11_111 and 0x5555 (101010101010101 in binary), which
    // only repeats give, come up. 11_111 comes up in about 1 draw of 210, and
    // 10_000, which only zeros give, in about 1 draw of 330. With repeats in 1
    // draw of 10 instead of half, 11_111 would come up 5 times less often, and
    // without repeats, never. 0x5555, the rarest, comes up in about 1 draw of
    // 820, so 50,000 draws miss it with probability about e^-60.
    //
    // About 240 draws give 11_111 and 150 give 10_000, so their ratio, about
    // 1.6, has a standard error of about 0.17, far inside [1/2, 3].
    var prng = Prng.init(test_seed);

    const watched_values = [_]u16{ 11_111, 10_000, 0x5555 };
    var value_histogram: Histogram(u16, &watched_values) = .{};

    for (0..50_000) |_| {
        value_histogram.add(prng.intLiteralBetween(u16, 0, std.math.maxInt(u16)));
    }

    const repeated_ones_count = value_histogram.count(11_111);
    const ten_thousand_count = value_histogram.count(10_000);

    try value_histogram.expectAllSeen();
    try std.testing.expect(repeated_ones_count * 2 > ten_thousand_count);
    try std.testing.expect(repeated_ones_count < ten_thousand_count * 3);
}

test "Prng.chance returns true with its probability" {
    // Never true for a numerator of 0, always true for a numerator equal to
    // the denominator, and true in about numerator / denominator of the draws
    // in between. Over 40,000 draws, the share of 1/4 has a standard error of
    // about 0.002, so the tolerance of 0.02 is about nine standard errors.
    var prng = Prng.init(test_seed);

    var one_in_four: Histogram(bool, &true_only) = .{};

    for (0..40_000) |_| {
        try std.testing.expectEqual(false, prng.chance(0, 4));
        try std.testing.expectEqual(true, prng.chance(4, 4));

        one_in_four.add(prng.chance(1, 4));
    }

    try one_in_four.expectShare(true, 0.25, 0.02);
}

test "Prng.enumWeighted follows the weights" {
    // A tag whose weight is 0 never comes up, and the other tags come up in
    // proportion to their weights: red in 1/4 of the draws and blue in 3/4.
    // Over 40,000 draws, each share has a standard error of about 0.002, so
    // the tolerance of 0.02 is about nine standard errors.
    const Color = enum { red, green, blue };

    var prng = Prng.init(test_seed);

    // Green sits between the two enabled tags, so a mix-up between the
    // weights and the tags they belong to would make green come up.
    const weights: EnumWeightsType(Color) = .{ .red = 1, .green = 0, .blue = 3 };
    const colors = [_]Color{ .red, .green, .blue };

    var color_histogram: Histogram(Color, &colors) = .{};

    for (0..40_000) |_| {
        color_histogram.add(prng.enumWeighted(Color, weights));
    }

    try color_histogram.expectShare(.red, 0.25, 0.02);
    try color_histogram.expectShare(.green, 0, 0);
    try color_histogram.expectShare(.blue, 0.75, 0.02);
}

test "Prng.enumSet gives every count the same odds and favors no tag" {
    // `enumSet` returns only tags of `allowed`, never an empty set, every
    // size about equally often, and no tag more often than another. Each size
    // comes up in a third of the draws, and no tag is favored within a size:
    // each single tag and each pair come up in 1/9 of the draws, and all three
    // in 1/3. Over 10,000 draws, a share of 1/9 has a standard error of about
    // 0.003, so the tolerance of 0.02 is about six standard errors. Without
    // the shuffle in `enumSet`, every single-tag draw would return `.a`.
    const Tag = enum { a, b, c, d, e };

    var prng = Prng.init(test_seed);

    const allowed: std.EnumSet(Tag) = .initMany(&.{ .a, .c, .e });

    // Bit 0 stands for `.a`, bit 1 for `.c`, and bit 2 for `.e`. The empty
    // set, 0, is not watched, so `expectNothingElse` checks that no draw
    // returns it.
    const nonempty_subsets = [_]usize{ 1, 2, 3, 4, 5, 6, 7 };
    var subset_histogram: Histogram(usize, &nonempty_subsets) = .{};

    for (0..10_000) |_| {
        const subset = prng.enumSet(Tag, allowed);

        try std.testing.expect(subset.subsetOf(allowed));

        const subset_bits = @as(usize, @intFromBool(subset.contains(.a))) |
            @as(usize, @intFromBool(subset.contains(.c))) << 1 |
            @as(usize, @intFromBool(subset.contains(.e))) << 2;

        subset_histogram.add(subset_bits);
    }

    try subset_histogram.expectNothingElse();

    for (nonempty_subsets) |subset_bits| {
        const expected_share: f64 = if (@popCount(subset_bits) == 3)
            1.0 / 3.0
        else
            1.0 / 9.0;

        try subset_histogram.expectShare(subset_bits, expected_share, 0.02);
    }
}

test "Prng.enumWeights enables only tags of allowed" {
    const Tag = enum { a, b, c, d, e };

    var prng = Prng.init(test_seed);

    const allowed: std.EnumSet(Tag) = .initMany(&.{ .a, .c, .e });

    for (0..10_000) |_| {
        const weights = prng.enumWeights(Tag, allowed);

        var enabled_count: usize = 0;

        inline for (comptime std.enums.values(Tag)) |tag| {
            if (@field(weights, @tagName(tag)) > 0) {
                try std.testing.expect(allowed.contains(tag));

                enabled_count += 1;
            }
        }

        try std.testing.expect(enabled_count >= 1);
    }
}

test "Prng.enumWeights enables one to all tags" {
    // Every weight lies in [0, 1000], and every count of enabled tags from 1
    // to all 10 comes up in about 1 call of 10, as with TigerBeetle's
    // `enum_weights`. Over 10,000 calls, a share of 1/10 has a standard error
    // of 0.003, so the tolerance of 0.02 is almost seven standard errors.
    const Tag = enum { a, b, c, d, e, f, g, h, i, j };

    var prng = Prng.init(test_seed);

    // 0 is not watched, so `expectNothingElse` checks that no call enables 0
    // tags.
    const enabled_tag_counts = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var enabled_tag_count_histogram: Histogram(usize, &enabled_tag_counts) = .{};

    for (0..10_000) |_| {
        const weights = prng.enumWeights(Tag, .full);

        var enabled_tag_count: usize = 0;

        inline for (comptime std.enums.values(Tag)) |tag| {
            const weight = @field(weights, @tagName(tag));

            try std.testing.expect(weight <= 1000);

            if (weight > 0) {
                enabled_tag_count += 1;
            }
        }

        enabled_tag_count_histogram.add(enabled_tag_count);
    }

    try enabled_tag_count_histogram.expectNothingElse();

    for (enabled_tag_counts) |enabled_tag_count| {
        try enabled_tag_count_histogram.expectShare(enabled_tag_count, 0.1, 0.02);
    }
}

test "Prng.enumWeights draws lopsided weights" {
    // About 37% of the enabled weights lie below 10. When two tags are both
    // enabled, one outweighs the other by ten times or more in about 55% of
    // the calls, and both get the same weight in about 7%, as the
    // documentation of `enumWeights` says. Giving every enabled tag of a call
    // the same weight would fail both of these.
    //
    // `intLiteralBetween(u64, 1, 1000)` returns less than 10 in about 37% of
    // the draws, 1 alone in about 21%; with weights drawn evenly from 1 to 100
    // it would be 9%. The 10,000 calls give about 55,000 weights, so the share
    // has a standard error of about 0.002, and the tolerance of 0.03 also
    // leaves room for the rounding in the 37%. Both tags are enabled in about
    // 3,700 of the calls. There, the tenfold share has a standard error of
    // about 0.008 and the equal share one of about 0.004, so the tolerances of
    // 0.05 and 0.025 are about six standard errors.
    const Tag = enum { a, b, c, d, e, f, g, h, i, j };

    var prng = Prng.init(test_seed);

    var small_weight: Histogram(bool, &true_only) = .{};
    var tenfold_pair: Histogram(bool, &true_only) = .{};
    var equal_pair: Histogram(bool, &true_only) = .{};

    for (0..10_000) |_| {
        const weights = prng.enumWeights(Tag, .full);

        inline for (comptime std.enums.values(Tag)) |tag| {
            const weight = @field(weights, @tagName(tag));

            if (weight > 0) {
                small_weight.add(weight < 10);
            }
        }

        // The first two tags stand for every pair of tags.
        if (weights.a > 0 and weights.b > 0) {
            tenfold_pair.add(weights.a >= 10 * weights.b or weights.b >= 10 * weights.a);
            equal_pair.add(weights.a == weights.b);
        }
    }

    try small_weight.expectShare(true, 0.37, 0.03);
    try tenfold_pair.expectShare(true, 0.55, 0.05);
    try equal_pair.expectShare(true, 0.07, 0.025);
}

test "Prng.indexEdge returns the ends" {
    // Only the first two and the last two indexes come up, each a quarter of
    // the time, and the ends overlap in short slices: the middle of 3 items
    // comes up half the time, and 1 item always gives 0. Over 10,000 draws, a
    // share of 1/4 or 1/2 has a standard error of at most 0.005, so the
    // tolerance of 0.03 is six standard errors.
    var prng = Prng.init(test_seed);

    const ten_items = [_]u8{0} ** 10;
    const three_items = [_]u8{0} ** 3;
    const one_item = [_]u8{0};

    // The middle indexes of ten items are not watched, so
    // `expectNothingElse` checks that they never come up.
    const ten_items_ends = [_]usize{ 0, 1, 8, 9 };
    const three_items_indexes = [_]usize{ 0, 1, 2 };

    var ten_items_histogram: Histogram(usize, &ten_items_ends) = .{};
    var three_items_histogram: Histogram(usize, &three_items_indexes) = .{};

    for (0..10_000) |_| {
        ten_items_histogram.add(prng.indexEdge(&ten_items));
        three_items_histogram.add(prng.indexEdge(&three_items));

        try std.testing.expectEqual(0, prng.indexEdge(&one_item));
    }

    try ten_items_histogram.expectNothingElse();

    for (ten_items_ends) |item_index| {
        try ten_items_histogram.expectShare(item_index, 0.25, 0.03);
    }

    try three_items_histogram.expectShare(0, 0.25, 0.03);
    try three_items_histogram.expectShare(1, 0.5, 0.03);
    try three_items_histogram.expectShare(2, 0.25, 0.03);
}

test "Prng.fillPattern uses only the pool" {
    // Every pattern, on slices of several lengths including an empty one:
    // every item comes from the pool, and a pool of one value gives a
    // constant slice.
    var prng = Prng.init(test_seed);

    const pool = [_]u8{ 10, 20, 30, 40 };

    var storage: [1_000]u8 = undefined;
    for ([_]usize{ 0, 1, 2, 3, 7, 64, 1_000 }) |items_count| {
        const items = storage[0..items_count];

        inline for (comptime std.enums.values(Pattern)) |pattern| {
            for (0..100) |_| {
                prng.fillPattern(u8, items, pattern, &pool);

                for (items) |item| {
                    try std.testing.expect(std.mem.indexOfScalar(u8, &pool, item) != null);
                }

                prng.fillPattern(u8, items, pattern, &.{7});

                for (items) |item| {
                    try std.testing.expectEqual(7, item);
                }
            }
        }
    }
}

test "Prng.fillPattern constant repeats one pool value" {
    // Every item equals the first, and each pool value comes up as the
    // constant.
    var prng = Prng.init(test_seed);

    const pool = [_]u8{ 10, 20, 30, 40 };

    var constant_histogram: Histogram(u8, &pool) = .{};

    var items: [64]u8 = undefined;
    for (0..1_000) |_| {
        prng.fillPattern(u8, &items, .constant, &pool);

        for (items) |item| {
            try std.testing.expectEqual(items[0], item);
        }

        constant_histogram.add(items[0]);
    }

    try constant_histogram.expectAllSeen();
    try constant_histogram.expectNothingElse();
}

test "Prng.fillPattern mostly_constant keeps one fill value with a few odd items" {
    // Every slice has at least one odd item, because outliers take another
    // pool position than the fill value, and at most a quarter of the items.
    // Both a single outlier and more than 10 come up, a single outlier lands
    // on the first or last item in about 30% of the slices, and it also lands
    // on round positions such as 8, 16, 32, and 64. The count of odd items can
    // be lower than the count drawn, because two outliers can land on the
    // same item.
    var prng = Prng.init(test_seed);

    const pool = [_]u8{ 10, 20, 30, 40 };

    var single_outlier_at_end: Histogram(bool, &true_only) = .{};
    var saw_many_outliers = false;
    var saw_round_position = false;

    var items: [1_000]u8 = undefined;
    for (0..10_000) |_| {
        prng.fillPattern(u8, &items, .mostly_constant, &pool);

        // The fill value is the pool value at most positions.
        var items_count_by_pool_index = [_]usize{0} ** pool.len;

        for (items) |item| {
            const pool_index = std.mem.indexOfScalar(u8, &pool, item) orelse {
                return error.TestUnexpectedResult;
            };

            items_count_by_pool_index[pool_index] += 1;
        }

        const fill_items_count = std.mem.max(usize, &items_count_by_pool_index);
        const outlier_count = items.len - fill_items_count;

        try std.testing.expect(outlier_count >= 1);
        try std.testing.expect(outlier_count <= items.len / 4);

        if (outlier_count > 10) {
            saw_many_outliers = true;
        }

        if (outlier_count == 1) {
            const fill_pool_index = std.mem.indexOfScalar(
                usize,
                &items_count_by_pool_index,
                fill_items_count,
            ) orelse {
                return error.TestUnexpectedResult;
            };

            const fill_value = pool[fill_pool_index];

            var outlier_index: usize = 0;

            for (items, 0..) |item, item_index| {
                if (item != fill_value) {
                    outlier_index = item_index;
                }
            }

            single_outlier_at_end.add(outlier_index == 0 or outlier_index == items.len - 1);

            if (std.mem.indexOfScalar(usize, &.{ 8, 16, 32, 64 }, outlier_index) != null) {
                saw_round_position = true;
            }
        }
    }

    try std.testing.expect(saw_many_outliers);
    try std.testing.expect(saw_round_position);

    // `intLiteralBetween(usize, 0, 999)` returns 0 and 999 about 14% of the
    // time each. About 1,300 of the 10,000 slices have a single outlier, so
    // the share has a standard error of about 0.013, and the tolerance of
    // 0.08 is about six standard errors.
    try single_outlier_at_end.expectShare(true, 0.30, 0.08);
}

test "Prng.fillPattern periodic cycles through 2 to all pool values" {
    // Each slice repeats with a period from 2 to the pool size of 4, every
    // such period comes up, and the cycle starts at every pool value. The pool
    // values are distinct, so a cycle of p values repeats every p items and no
    // sooner. The first item is the pool value at the cycle's start, and the
    // last pool value needs a period of 4 and a start of 3, 1 slice in 12, so
    // 1,000 slices miss it with probability about e^-87.
    var prng = Prng.init(test_seed);

    const pool = [_]u8{ 10, 20, 30, 40 };

    // Periods of 1 or above the pool size are not watched, so
    // `expectNothingElse` checks that none comes up.
    const periods = [_]usize{ 2, 3, 4 };

    var period_histogram: Histogram(usize, &periods) = .{};
    var first_item_histogram: Histogram(u8, &pool) = .{};

    var items: [64]u8 = undefined;
    for (0..1_000) |_| {
        prng.fillPattern(u8, &items, .periodic, &pool);

        // The smallest shift that maps the slice onto itself is its period.
        // A shift by the whole length always does, so the loop stops.
        var period: usize = 1;

        while (!std.mem.eql(u8, items[period..], items[0 .. items.len - period])) {
            period += 1;
        }

        period_histogram.add(period);
        first_item_histogram.add(items[0]);
    }

    try period_histogram.expectAllSeen();
    try period_histogram.expectNothingElse();
    try first_item_histogram.expectAllSeen();
}

test "Prng.fillPattern runs forms runs of round lengths" {
    // Every slice of 2 items or more has at least 2 runs, single items and
    // runs over more than half the slice both come up, and round lengths come
    // up far more often than their neighbors. Neighboring runs take different
    // pool values, so each stretch of equal items is one run.
    var prng = Prng.init(test_seed);

    const pool = [_]u8{ 10, 20, 30, 40 };
    const watched_lengths = [_]usize{ 1, 16, 17, 64, 65 };

    var length_histogram: Histogram(usize, &watched_lengths) = .{};
    var over_half: Histogram(bool, &true_only) = .{};

    var items: [1_000]u8 = undefined;
    for (0..1_000) |_| {
        prng.fillPattern(u8, &items, .runs, &pool);

        var run_start_index: usize = 0;
        var slice_run_count: usize = 0;

        for (1..items.len + 1) |run_end_index| {
            if (run_end_index < items.len and items[run_end_index] == items[run_start_index]) {
                continue;
            }

            const run_item_count = run_end_index - run_start_index;

            length_histogram.add(run_item_count);
            over_half.add(run_item_count > items.len / 2);

            slice_run_count += 1;
            run_start_index = run_end_index;
        }

        try std.testing.expect(slice_run_count >= 2);
    }

    var two_items: [2]u8 = undefined;
    for (0..1_000) |_| {
        prng.fillPattern(u8, &two_items, .runs, &pool);

        try std.testing.expect(two_items[0] != two_items[1]);
    }

    try std.testing.expect(length_histogram.count(1) > 0);
    try over_half.expectAllSeen();

    // About 80 runs have 16 items and 20 have 17, and about 55 have 64 items
    // and 4 have 65. Run lengths drawn evenly would give neighbors about the
    // same counts.
    try std.testing.expect(length_histogram.count(16) > 2 * length_histogram.count(17));
    try std.testing.expect(length_histogram.count(64) > 2 * length_histogram.count(65));
}

test "Prng.fillPattern scattered picks every item independently" {
    // A long slice contains every pool value, and about 1 in 4 pairs of
    // neighboring items are equal, as independent picks from 4 values give.
    // Each item misses a given value with probability 3/4, so all 1,000 items
    // of a slice miss it with probability (3/4)^1_000, about 1e-125. The 10
    // slices give 9,990 pairs, so the equal share has a standard error of
    // about 0.0043, and the tolerance of 0.03 is about seven standard errors.
    var prng = Prng.init(test_seed);

    const pool = [_]u8{ 10, 20, 30, 40 };

    var equal_pair: Histogram(bool, &true_only) = .{};

    var items: [1_000]u8 = undefined;
    for (0..10) |_| {
        prng.fillPattern(u8, &items, .scattered, &pool);

        var item_histogram: Histogram(u8, &pool) = .{};

        for (items) |item| {
            item_histogram.add(item);
        }

        try item_histogram.expectAllSeen();

        for (items[0 .. items.len - 1], items[1..]) |item, next_item| {
            equal_pair.add(item == next_item);
        }
    }

    try equal_pair.expectShare(true, 0.25, 0.03);
}

test "Prng.floatFinite returns only finite values with every exponent equally likely" {
    // The test draws f16 values, because far more of their bit patterns are
    // not finite than for f32. About 3% of the f16 bit patterns are NaNs, and
    // 2 of the 65,536 are the infinities, so 300,000 draws hit an infinity
    // about 9 times. With fewer draws, a `floatFinite` that let infinities
    // through could pass.
    //
    // Each of the 31 finite exponents of f16 comes up in about 1 draw of 31.
    // Over 300,000 draws, a share of 1/31 has a standard error of about
    // 0.0003, so the tolerance of 0.003 is about nine standard errors.
    var prng = Prng.init(test_seed);

    // The 5 exponent bits of f16 sit above its 10 fraction bits. The
    // all-ones exponent, 31, encodes the infinities and NaN, and it is not
    // watched, so `expectNothingElse` checks that no draw is one of them.
    const finite_exponents: [31]u16 = comptime std.simd.iota(u16, 31);
    var exponent_histogram: Histogram(u16, &finite_exponents) = .{};

    for (0..300_000) |_| {
        exponent_histogram.add(@as(u16, @bitCast(prng.floatFinite(f16))) >> 10 & 0x1f);
    }

    try exponent_histogram.expectNothingElse();

    for (finite_exponents) |exponent| {
        try exponent_histogram.expectShare(exponent, 1.0 / 31.0, 0.003);
    }
}

test "Prng.floatBetween stays in range" {
    try expectDrawsBetween(Prng.floatBetween, f32, -3, 5);
    try expectDrawsBetween(Prng.floatBetween, f16, -3, 5);

    // In f16, 40_000 - -40_000 overflows to infinity, so this range works
    // only because the bounds are checked in f32.
    try expectDrawsBetween(Prng.floatBetween, f16, -40_000, 40_000);
}

test "Prng.floatBetween spreads evenly" {
    // About half the draws from [-3, 5] fall below 1, the middle of the range,
    // for f32 and for f16, which draws in f32 and rounds. Over 10,000 draws, a
    // share of 1/2 has a standard error of 0.005, so the tolerance of 0.03 is
    // six standard errors.
    var prng = Prng.init(test_seed);

    var below_middle_f32: Histogram(bool, &true_only) = .{};
    var below_middle_f16: Histogram(bool, &true_only) = .{};

    for (0..10_000) |_| {
        below_middle_f32.add(prng.floatBetween(f32, -3, 5) < 1);
        below_middle_f16.add(prng.floatBetween(f16, -3, 5) < 1);
    }

    try below_middle_f32.expectShare(true, 0.5, 0.03);
    try below_middle_f16.expectShare(true, 0.5, 0.03);
}

test "Prng.floatLogUniform stays in range" {
    // Each type, a one-value range, and the widest f32 range, from the
    // smallest subnormal to the largest finite value.
    try expectDrawsBetween(Prng.floatLogUniform, f32, 2, 2);
    try expectDrawsBetween(Prng.floatLogUniform, f16, 0.001, 1000);
    try expectDrawsBetween(Prng.floatLogUniform, f32, 0.001, 1000);

    try expectDrawsBetween(
        Prng.floatLogUniform,
        f32,
        std.math.floatTrueMin(f32),
        std.math.floatMax(f32),
    );

    try expectDrawsBetween(Prng.floatLogUniform, f64, 1e-150, 1e150);
}

test "Prng.floatLogUniform spreads evenly across orders of magnitude" {
    // Each of the six tenfold ranges from 0.001 to 1000 gets a sixth of the
    // draws. Over 100,000 draws, each share has a standard error of about
    // 0.0012, so the tolerance of 0.01 is about eight standard errors.
    var prng = Prng.init(test_seed);

    const decade_indexes = [_]usize{ 0, 1, 2, 3, 4, 5 };
    var decade_histogram: Histogram(usize, &decade_indexes) = .{};

    for (0..100_000) |_| {
        const value = prng.floatLogUniform(f64, 0.001, 1000);

        // `log10` can land a hair outside [-3, 3] at the ends, and a draw of
        // exactly 1000 belongs to the last range, so the clamp keeps every
        // draw in one of the six ranges.
        const decade_index = std.math.clamp(@floor(std.math.log10(value) + 3), 0, 5);

        decade_histogram.add(@intFromFloat(decade_index));
    }

    for (decade_indexes) |decade_index| {
        try decade_histogram.expectShare(decade_index, 1.0 / 6.0, 0.01);
    }
}

test "Prng.floatLogUniform keeps narrow ranges of large numbers" {
    // A range of about 70 f64 values around 1e30 keeps its spread. The
    // logarithms of both ends round to the same f64 or to neighbors, so
    // drawing from ln(min) to ln(max) would return only the two ends. The
    // log-uniform spread over such a narrow range is almost even, so about
    // half the draws land in the middle half of the range. Over 10,000 draws,
    // that share has a standard error of 0.005, so the tolerance of 0.05 is
    // ten standard errors.
    var prng = Prng.init(test_seed);

    const range_min: f64 = 1e30;
    const range_max = range_min * (1 + 1e-14);
    const range_width = range_max - range_min;

    var in_middle_half: Histogram(bool, &true_only) = .{};

    for (0..10_000) |_| {
        const value = prng.floatLogUniform(f64, range_min, range_max);

        in_middle_half.add(
            value > range_min + range_width / 4 and value < range_max - range_width / 4,
        );
    }

    try in_middle_half.expectShare(true, 0.5, 0.05);
}

test "Prng.floatGaussian keeps the mean and the standard deviation" {
    // f16 draws average to `mean`. With a standard deviation of 0.5, the
    // average of 10,000 draws has a standard error of 0.005, so the tolerance
    // of 0.05 is ten standard errors. Swapping `mean` and
    // `standard_deviation` would put the average at 0.5.
    //
    // About 68% of the f32 draws lie within one `standard_deviation` of
    // `mean`. Over 10,000 draws, that share has a standard error of about
    // 0.0047, so the tolerance of 0.03 is about six standard errors.
    // Returning `mean` itself would put every draw within.
    var prng = Prng.init(test_seed);

    const draw_count = 10_000;

    var sum: f64 = 0;
    var within_one_deviation: Histogram(bool, &true_only) = .{};

    for (0..draw_count) |_| {
        sum += prng.floatGaussian(f16, 4, 0.5);

        within_one_deviation.add(@abs(prng.floatGaussian(f32, 0, 1)) < 1);
    }

    try std.testing.expectApproxEqAbs(4, sum / draw_count, 0.05);
    try within_one_deviation.expectShare(true, 0.6827, 0.03);
}

test "Prng.floatEdge returns exactly its listed values" {
    // The magnitudes its documentation lists for each type, NaN included,
    // each with both signs.
    //
    // The integer edges from 2^16 up exceed the largest f16, 65_504, and f16
    // has no narrower type.
    const f16_edges = comptime withBothSigns(f16, &.{
        0,                      std.math.floatTrueMin(f16), std.math.floatMin(f16), 1,
        std.math.floatMax(f16), std.math.inf(f16),          std.math.nan(f16),      0x1p7,
        0x1p8,                  0x1p11,                     0x1p15,
    });

    // f32's edges, then f16's edges and the two values that a cast to f16
    // turns into 0 and infinity, then the integer edges.
    const f32_edges = comptime withBothSigns(f32, &.{
        0,                          std.math.floatTrueMin(f32), std.math.floatMin(f32), 1,
        std.math.floatMax(
            f32,
        ),
        std.math.inf(
            f32,
        ),
        std.math.nan(
            f32,
        ),
        std.math.floatTrueMin(f16), std.math.floatMin(f16),     std.math.floatMax(f16), 0x1p-25,
        65_520,                     0x1p7,                      0x1p8,                  0x1p11,
        0x1p15,                     0x1p16,                     0x1p24,                 0x1p31,
        0x1p32,                     0x1p53,                     0x1p63,                 0x1p64,
    });

    // As for f32, plus f32's edges. A cast to f32 turns 2^-150 into 0, and
    // 0x1.ffffffp127, halfway between the largest f32 and 2^128, into
    // infinity.
    const f64_edges = comptime withBothSigns(f64, &.{
        0,                      std.math.floatTrueMin(f64), std.math.floatMin(f64),     1,
        std.math.floatMax(
            f64,
        ),
        std.math.inf(
            f64,
        ),
        std.math.nan(f64),      std.math.floatTrueMin(f16), std.math.floatMin(f16),
        std.math.floatMax(
            f16,
        ),
        0x1p-25,                65_520,                     std.math.floatTrueMin(f32),
        std.math.floatMin(
            f32,
        ),
        std.math.floatMax(f32), 0x1p-150,                   0x1.ffffffp127,             0x1p7,
        0x1p8,                  0x1p11,                     0x1p15,                     0x1p16,
        0x1p24,                 0x1p31,                     0x1p32,                     0x1p53,
        0x1p63,                 0x1p64,
    });

    try expectDrawsExactly(Prng.floatEdge, f16, f16_edges);
    try expectDrawsExactly(Prng.floatEdge, f32, f32_edges);
    try expectDrawsExactly(Prng.floatEdge, f64, f64_edges);
}

test "Prng.floatEdgeFinite returns exactly the finite edges" {
    // The magnitudes of `floatEdge`, without infinity and NaN, each with both
    // signs.
    const f16_edges = comptime withBothSigns(f16, &.{
        0,                      std.math.floatTrueMin(f16), std.math.floatMin(f16), 1,
        std.math.floatMax(f16), 0x1p7,                      0x1p8,                  0x1p11,
        0x1p15,
    });

    const f32_edges = comptime withBothSigns(f32, &.{
        0,                          std.math.floatTrueMin(f32), std.math.floatMin(f32), 1,
        std.math.floatMax(
            f32,
        ),
        std.math.floatTrueMin(f16), std.math.floatMin(f16),     std.math.floatMax(f16), 0x1p-25,
        65_520,                     0x1p7,                      0x1p8,                  0x1p11,
        0x1p15,                     0x1p16,                     0x1p24,                 0x1p31,
        0x1p32,                     0x1p53,                     0x1p63,                 0x1p64,
    });

    try expectDrawsExactly(Prng.floatEdgeFinite, f16, f16_edges);
    try expectDrawsExactly(Prng.floatEdgeFinite, f32, f32_edges);
}

test "Prng.floatEdge and floatEdgeFinite pick each group of edges a third of the time" {
    // For f32, the 11 integer edges come up in about a third of the draws, as
    // one of three groups. Picking from one list of all 23 edges would give
    // them 11/23, about 48%, and a `floatEdgeFinite` that drew `floatEdge`
    // again for infinity and NaN would give them 7/19, about 37%. Over 20,000
    // draws, a share of 1/3 has a standard error of about 0.0033, so the
    // tolerance of 0.02 is about six standard errors.
    const integer_edges = [_]f32{
        0x1p7,  0x1p8,  0x1p11, 0x1p15, 0x1p16, 0x1p24,
        0x1p31, 0x1p32, 0x1p53, 0x1p63, 0x1p64,
    };

    inline for (.{ Prng.floatEdge, Prng.floatEdgeFinite }) |draw| {
        var prng = Prng.init(test_seed);

        var integer_edge: Histogram(bool, &true_only) = .{};

        for (0..20_000) |_| {
            const magnitude = @abs(draw(&prng, f32));

            integer_edge.add(std.mem.indexOfScalar(f32, &integer_edges, magnitude) != null);
        }

        try integer_edge.expectShare(true, 1.0 / 3.0, 0.02);
    }
}

test "Prng.floatLiteral stays within 2^p" {
    // [-2_048, 2_048] for f16 and [-2^24, 2^24] for f32. A NaN would fail
    // the comparison too.
    var prng = Prng.init(test_seed);

    for (0..10_000) |_| {
        try std.testing.expect(@abs(prng.floatLiteral(f16)) <= 2_048);
        try std.testing.expect(@abs(prng.floatLiteral(f32)) <= 0x1p24);
    }
}

test "Prng.floatLiteral returns common literals" {
    // Both bounds of f32 and literals such as 1, 0.1, and 1/3 come up. The
    // rarest of them, 1/3, comes up in about 1 draw of 4_000, so 100,000
    // draws miss it with probability about e^-25.
    var prng = Prng.init(test_seed);

    const common_literals = [_]f32{ 0x1p24, -0x1p24, 0, 1, -1, 0.5, 3, 0.1, 1_000, 1.0 / 3.0 };
    var literal_histogram: Histogram(f32, &common_literals) = .{};

    for (0..100_000) |_| {
        literal_histogram.add(prng.floatLiteral(f32));
    }

    try literal_histogram.expectAllSeen();
}

test "Prng.floatLiteralBetween stays in range" {
    // Each type, a range of one value, ranges across 0 with a short and a
    // long side, a narrow range of large numbers, and the whole finite range
    // of f32.
    try expectDrawsBetween(Prng.floatLiteralBetween, f32, 2, 2);
    try expectDrawsBetween(Prng.floatLiteralBetween, f16, 0.001, 1000);
    try expectDrawsBetween(Prng.floatLiteralBetween, f32, -80, 80);
    try expectDrawsBetween(Prng.floatLiteralBetween, f32, -0.001, 5);
    try expectDrawsBetween(Prng.floatLiteralBetween, f64, 1e30, 1.0001e30);

    try expectDrawsBetween(
        Prng.floatLiteralBetween,
        f32,
        -std.math.floatMax(f32),
        std.math.floatMax(f32),
    );

    // These f64 ranges need the limits on the floor of the magnitudes: 2^-53
    // lies more than 2^1000 below the largest f64, and 1e-310 · 2^-53 rounds
    // to 0.
    try expectDrawsBetween(
        Prng.floatLiteralBetween,
        f64,
        -std.math.floatMax(f64),
        std.math.floatMax(f64),
    );

    try expectDrawsBetween(Prng.floatLiteralBetween, f64, 0, 1e-310);
}

test "Prng.floatLiteralBetween returns the bounds, 0, and common literals" {
    // For [-80, 80] in f32. The rarest watched value, 2.5, comes up in about
    // 1 draw of 5_500, so 100,000 draws miss it with probability about
    // e^-18.
    var prng = Prng.init(test_seed);

    const common_literals = [_]f32{ -80, 80, 0, 1, -1, 0.5, 2.5, 0.1 };
    var literal_histogram: Histogram(f32, &common_literals) = .{};

    for (0..100_000) |_| {
        literal_histogram.add(prng.floatLiteralBetween(f32, -80, 80));
    }

    try literal_histogram.expectAllSeen();
}

test "Prng.floatLiteralBetween favors few binary digits" {
    // For [-80, 80] in f32, at least a quarter of the results have a
    // significand of at most 4 leading binary digits followed by zeros or
    // repeats, which a uniform draw almost never gives, and significands of
    // all ones, such as 63.9999962 (111111.111… in binary), come up. About
    // 44% of the results have few digits, and 1 in 17 has all ones. Over 2,000
    // draws, the share has a standard error of about 0.011, far above the
    // threshold of 0.25, and all 2,000 draws miss all ones with probability
    // about e^-120.
    var prng = Prng.init(test_seed);

    const draw_count = 2_000;

    var few_digits: Histogram(bool, &true_only) = .{};
    var all_ones: Histogram(bool, &true_only) = .{};

    for (0..draw_count) |_| {
        // 0 takes no digits at all. Every other result is at least about 5e-8
        // in magnitude, far above the smallest normal f32, 1.2e-38, so its
        // significand is the 23 stored bits with the leading 1 in front.
        const value = prng.floatLiteralBetween(f32, -80, 80);
        if (value == 0) {
            few_digits.add(true);

            continue;
        }

        const significand = @as(u32, @bitCast(value)) & 0x7f_ffff | 0x80_0000;

        few_digits.add(hasFewDigits(significand, 2, 4));
        all_ones.add(significand == 0xff_ffff);
    }

    try std.testing.expect(few_digits.share(true) > 0.25);
    try all_ones.expectAllSeen();
}

test "Prng.floatLiteralBetween repeats decimal digits" {
    // For [-80, 80] in f32, the f32s nearest to 0.333333333, 3.33333333, and
    // 33.3333333, a decimal digit repeated to the 9 digits that f32 keeps,
    // come up. Each comes up in about 1 draw of 1_200, so 40,000 draws miss
    // one of them with probability below e^-32.
    var prng = Prng.init(test_seed);

    const repeated_threes = [_]f32{ 0.333333333, 3.33333333, 33.3333333 };
    var magnitude_histogram: Histogram(f32, &repeated_threes) = .{};

    for (0..40_000) |_| {
        magnitude_histogram.add(@abs(prng.floatLiteralBetween(f32, -80, 80)));
    }

    try magnitude_histogram.expectAllSeen();
}

test "Prng.floatLiteralBetween reaches tiny and large magnitudes" {
    // For [-80, 80] in f32, magnitudes below 0.001 and above 10 both come up.
    // Tiny magnitudes come up in about 37% of the draws and large ones in
    // about 21%, so 200 draws miss either with probability below 1e-20.
    var prng = Prng.init(test_seed);

    var tiny: Histogram(bool, &true_only) = .{};
    var large: Histogram(bool, &true_only) = .{};

    for (0..200) |_| {
        const magnitude = @abs(prng.floatLiteralBetween(f32, -80, 80));

        tiny.add(magnitude != 0 and magnitude < 0.001);
        large.add(magnitude > 10);
    }

    try tiny.expectAllSeen();
    try large.expectAllSeen();
}

test "Prng.scaleByPower returns the nearest f64" {
    // Decimal powers far from 1 still round once: 1e-7, 1e-10, and 1e100
    // come out as the f64s nearest to them, where f64 math alone gives
    // 1.0000000000000001e-7, 9.999999999999999e-11, and
    // 1.0000000000000002e100. 3 · 10^-1 gives 0.3, not 0.30000000000000004,
    // and 49 · 10^-325 gives the smallest subnormal f64, 4.9e-324.
    try std.testing.expectEqual(0.3, scaleByPower(3, 10, -1));
    try std.testing.expectEqual(1e-7, scaleByPower(1, 10, -7));
    try std.testing.expectEqual(1e-10, scaleByPower(1, 10, -10));
    try std.testing.expectEqual(1e100, scaleByPower(1, 10, 100));
    try std.testing.expectEqual(std.math.floatTrueMin(f64), scaleByPower(49, 10, -325));
}

test "Prng.nudge stays within step_count_max" {
    // `nudge` returns only `value` and its neighbors within
    // `step_count_max`, and it returns the end of `T`'s range in place of
    // neighbors past that end.
    var prng = Prng.init(test_seed);

    var positive_count: usize = 0;

    for (0..10_000) |_| {
        // From 255, the offset +1 passes the largest `u8`, so `nudge` returns
        // 255 for it, and the only other result is 254.
        try std.testing.expect(prng.nudge(u8, 255, 1) >= 254);

        // From -128, the offsets in [-200, 200] reach at most
        // -128 + 200 = 72, and every negative offset stops at -128.
        const nudged_i8_min = prng.nudge(i8, -128, 200);

        try std.testing.expect(nudged_i8_min <= 72);

        // If `nudge` clamped the offset to the `i8` maximum of 127 before
        // adding it, no result would exceed -128 + 127 = -1. A positive result
        // shows that `nudge` adds offsets that do not fit in an `i8`.
        if (nudged_i8_min > 0) {
            positive_count += 1;
        }

        // The floats directly around 1.0 are 0.99999994, which is 1 - 2^-24,
        // and 1.00000012, which is 1 + 2^-23. The step below 1.0 is smaller,
        // because 1.0 is the smallest float with its exponent. No other float
        // lies between these two neighbors.
        const nudged_one = prng.nudge(f32, 1.0, 1);

        try std.testing.expect(nudged_one >= 0.99999994);
        try std.testing.expect(nudged_one <= 1.00000012);

        // From infinity, one step down reaches the largest finite `f32`, and
        // one step up stays at infinity.
        try std.testing.expect(prng.nudge(f32, std.math.inf(f32), 1) >= std.math.floatMax(f32));
    }

    try std.testing.expect(positive_count > 0);
}

test "Prng.nudge gives every step count the same odds" {
    // With a `step_count_max` of 2, `nudge` returns `value` and each of the
    // two neighbors on each side in about 1 draw of 5, for integers and for
    // floats. Over 10,000 draws, a share of 1/5 has a standard error of
    // 0.004, so the tolerance of 0.02 is five standard errors.
    var prng = Prng.init(test_seed);

    const int_neighbors = [_]i32{ -2, -1, 0, 1, 2 };

    const float_neighbors = comptime neighbors: {
        const one_below = std.math.nextAfter(f32, 1.0, 0);
        const one_above = std.math.nextAfter(f32, 1.0, 2);

        break :neighbors [_]f32{
            std.math.nextAfter(f32, one_below, 0),
            one_below,
            1.0,
            one_above,
            std.math.nextAfter(f32, one_above, 2),
        };
    };

    var int_histogram: Histogram(i32, &int_neighbors) = .{};
    var float_histogram: Histogram(f32, &float_neighbors) = .{};

    for (0..10_000) |_| {
        int_histogram.add(prng.nudge(i32, 0, 2));
        float_histogram.add(prng.nudge(f32, 1.0, 2));
    }

    try int_histogram.expectNothingElse();
    try float_histogram.expectNothingElse();

    for (int_neighbors, float_neighbors) |int_neighbor, float_neighbor| {
        try int_histogram.expectShare(int_neighbor, 0.2, 0.02);
        try float_histogram.expectShare(float_neighbor, 0.2, 0.02);
    }
}

/// `expectDrawsBetween` draws `draw(prng, T, min, max)` 10,000 times and
/// checks that every result lies in [`min`, `max`]. The draws cover each
/// range's ordinary results and catch assertions that trip for it. They can't
/// reach the rounding at a bound that the clamps in the functions guard
/// against, which needs a uniform draw within a few ulps of 1.
fn expectDrawsBetween(comptime draw: anytype, comptime T: type, min: T, max: T) !void {
    var prng = Prng.init(test_seed);

    for (0..10_000) |_| {
        const value = draw(&prng, T, min, max);

        // A NaN fails both comparisons.
        const is_in_range = min <= value and value <= max;

        if (!is_in_range) {
            std.debug.print("{any} lies outside [{any}, {any}] for {s}\n", .{
                value,
                min,
                max,
                @typeName(T),
            });

            return error.TestUnexpectedResult;
        }
    }
}

/// `expectDrawsExactly` draws `draw(prng, T)` 20,000 times and checks that
/// the results are exactly `values`: every one of them comes up, and nothing
/// else does.
fn expectDrawsExactly(comptime draw: anytype, comptime T: type, comptime values: []const T) !void {
    var prng = Prng.init(test_seed);

    var value_histogram: Histogram(T, values) = .{};

    // The rarest value that the tests check comes up in about 1 draw of 167:
    // 13 from `intLiteral(u4)`, which takes 4 binary digits or 2 decimal ones.
    // The 20,000 draws then miss it with probability (1 - 1/167)^20_000,
    // about 1e-52, so a missing value comes from a bug rather than chance.
    for (0..20_000) |_| {
        value_histogram.add(draw(&prng, T));
    }

    try value_histogram.expectAllSeen();
    try value_histogram.expectNothingElse();
}

/// `hasFewDigits` returns whether `value`, written in `base`, consists of at
/// most `block_digits_count_max` leading digits followed by zeros, or of a
/// block of at most that many digits repeated and cut off after any digit. In
/// decimal with 2 digits, 1_200, 7, 999, and 12_121 do, and 1_234 doesn't.
/// It works on the written digits, so it checks the literal functions without
/// sharing their arithmetic.
fn hasFewDigits(value: u64, base: u8, block_digits_count_max: usize) bool {
    var digit_storage: [64]u8 = undefined;
    const digits = digit_storage[0..std.fmt.printInt(&digit_storage, value, base, .lower, .{})];
    if (std.mem.trimEnd(u8, digits, "0").len <= block_digits_count_max) {
        return true;
    }

    // A block of k digits repeats when shifting the digits by k leaves them
    // unchanged. A value of at most `block_digits_count_max` digits returned
    // above, so every shift leaves some digits to compare.
    for (1..block_digits_count_max + 1) |block_digits_count| {
        if (std.mem.eql(
            u8,
            digits[block_digits_count..],
            digits[0 .. digits.len - block_digits_count],
        )) {
            return true;
        }
    }

    return false;
}
