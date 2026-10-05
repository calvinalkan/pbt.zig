//! `Exhaustive` enumerates every sequence of choices that a test body can
//! make, so the body covers a small input space completely instead of
//! sampling it. A `while (!exhaustive.done())` loop runs the body once per
//! sequence, and each run is one case. The body makes its choices through
//! methods that `Prng` also has, such as `intBetween`, `index`, and `pick`,
//! and the bounds of a choice may depend on the earlier choices of the same
//! case:
//!
//! ```zig
//! var exhaustive: Exhaustive = .{};
//! var cases_count: usize = 0;
//!
//! while (!exhaustive.done()) : (cases_count += 1) {
//!     const length = exhaustive.intBetween(usize, 0, 3);
//!     const position = exhaustive.intBetween(usize, 0, length);
//!     // ...
//! }
//!
//! // Each `length` in [0, 3] runs with every `position` in [0, `length`], so
//! // the loop runs `length + 1` cases per length.
//! try std.testing.expectEqual(1 + 2 + 3 + 4, cases_count);
//! ```
//!
//! Each case may make at most 32 choices: `shuffle` makes one choice per
//! item, and every other method makes one choice per call.
//!
//! Everything a case does must follow from its own choices. Whether the body
//! makes a choice at all, and the bounds of each choice, such as `min` and
//! `max` for `intBetween` or `items.len` for `pick`, may depend only on the
//! earlier choices of the same case. So may the data the case starts from:
//! reset anything the body changes, such as a slice it shuffles, at the start
//! of each case. A bound that depends on anything else, such as a `Prng` draw,
//! can make the cases skip or repeat sequences, or trip an assertion. Data
//! left over from the previous case can make different cases give the same
//! result: shuffling the two items 0 and 1 without resetting them gives 1, 0
//! in both cases.
//!
//! Adapted from TigerBeetle's `src/testing/exhaustigen.zig`
//! (https://github.com/tigerbeetle/tigerbeetle/blob/47aeb2212a255273dda508288412e537d11e4b7c/src/testing/exhaustigen.zig),
//! which implements the technique from matklad's "Generate All the Things"
//! (https://matklad.github.io/2021/11/07/generate-all-the-things.html).
//! TigerBeetle is licensed under the Apache License, Version 2.0
//! (https://github.com/tigerbeetle/tigerbeetle/blob/47aeb2212a255273dda508288412e537d11e4b7c/LICENSE).
//! Changes from the original:
//!
//! - Names match `Prng`: `intBetween` replaces `range_inclusive`, and
//!   `enumTag` replaces `enum_value`.
//! - `int_inclusive` is removed, because `intBetween` with a `min` of 0 does
//!   the same.
//! - The fields have descriptive names, and `gen` is renamed `choose`.
//! - `choose` asserts that a choice recorded from the previous case fits the
//!   bound this case asks for.
//! - `pick` and `boolean` are added.
//! - The permutation and shuffle tests check that every order comes up
//!   exactly once instead of only counting the cases. The tests for dependent
//!   bounds, for cases with different numbers of choices, and for `pick`,
//!   `boolean`, and `enumTag` are new.
const Exhaustive = @This();
const std = @import("std");
const assert = std.debug.assert;

const choices_count_max = 32;

started: bool = false,
choices: [choices_count_max]Choice = undefined,
choice_index: usize = 0,
choices_count: usize = 0,

const Choice = struct { value: u32, bound: u32 };

/// `done` returns false before each case, including the first, and returns
/// true once every sequence of choices has run.
pub fn done(exhaustive: *Exhaustive) bool {
    if (!exhaustive.started) {
        exhaustive.started = true;

        return false;
    }

    // `choices[0..choices_count]` holds the case that just ran: the value of
    // each choice and the bound the case asked for it. For example:
    //
    //   value:  3 1 4 4
    //   bound:  5 4 4 4
    //
    // The next case takes the next sequence in lexicographic order that stays
    // within the bounds. The last two 4s already equal their bounds, so the
    // rightmost choice that can grow is the 1. The loop increments it to 2
    // and drops every choice after it, which leaves 3 2. In the next case,
    // `choose` returns 3 and 2 for the first two choices and starts every
    // later choice at 0. The bounds of those later choices may differ from
    // this case's, because they may depend on the 2.
    var position = exhaustive.choices_count;

    while (position > 0) {
        position -= 1;

        const choice = &exhaustive.choices[position];
        if (choice.value < choice.bound) {
            choice.value += 1;
            exhaustive.choices_count = position + 1;
            exhaustive.choice_index = 0;

            return false;
        }
    }

    return true;
}

/// `intBetween` returns this case's choice in [`min`, `max`]. `T` must be
/// unsigned, `min` must not exceed `max`, and `max - min` must fit in a
/// `u32`.
pub fn intBetween(exhaustive: *Exhaustive, comptime T: type, min: T, max: T) T {
    comptime assert(@typeInfo(T).int.signedness == .unsigned);
    assert(min <= max);

    const offset = exhaustive.choose(@intCast(max - min));

    return min + @as(T, @intCast(offset));
}

/// `index` returns this case's index in [0, `items.len` - 1]; `items` must not
/// be empty.
pub fn index(exhaustive: *Exhaustive, items: anytype) usize {
    assert(items.len > 0);

    return exhaustive.intBetween(usize, 0, items.len - 1);
}

/// `pick` returns this case's element of `items`, which must not be empty.
pub fn pick(exhaustive: *Exhaustive, comptime T: type, items: []const T) T {
    return items[exhaustive.index(items)];
}

/// `boolean` returns this case's choice of false or true.
pub fn boolean(exhaustive: *Exhaustive) bool {
    return exhaustive.intBetween(u1, 0, 1) == 1;
}

/// `enumTag` returns this case's tag of `E`.
pub fn enumTag(exhaustive: *Exhaustive, comptime E: type) E {
    const tags = comptime std.enums.values(E);

    return tags[exhaustive.index(tags)];
}

/// `shuffle` puts `items` in this case's order. For n items that start in the
/// same order in every case, the cases reach all n! orders, so a body whose
/// only choices come from one `shuffle` call runs n! cases.
pub fn shuffle(exhaustive: *Exhaustive, comptime T: type, items: []T) void {
    // This is the inside-out Fisher-Yates shuffle. The element at `position`
    // either stays or swaps with one of the `position` elements before it,
    // which gives `position + 1` choices. Different sequences of those
    // choices give different orders, so the 1 * 2 * ... * n sequences give
    // each of the n! orders exactly once.
    for (0..items.len) |position| {
        const other = exhaustive.intBetween(usize, 0, position);

        std.mem.swap(T, &items[position], &items[other]);
    }
}

// `choose` returns this case's choice in [0, `bound`] and records `bound`,
// so that `done` knows how far the choice can still grow.
fn choose(exhaustive: *Exhaustive, bound: u32) u32 {
    assert(exhaustive.choice_index < choices_count_max);

    if (exhaustive.choice_index == exhaustive.choices_count) {
        // This case has already made all `choices_count` recorded choices, so
        // this choice is new and starts at 0.
        exhaustive.choices[exhaustive.choice_index] = .{ .value = 0, .bound = 0 };
        exhaustive.choices_count += 1;
    }

    const choice = &exhaustive.choices[exhaustive.choice_index];

    exhaustive.choice_index += 1;
    choice.bound = bound;

    // A recorded choice has the value it had in the previous case, or one
    // more if `done` incremented it. The choices before it are the same as in
    // the previous case, so a `bound` that depends only on them is the same
    // too, and the value fits it. A `bound` that also depends on anything
    // else can fall below the value; the assertion catches that instead of
    // returning a choice outside [0, `bound`].
    assert(choice.value <= bound);

    return choice.value;
}

test "Exhaustive allows bounds that depend on earlier choices" {
    // The bound of a choice may depend on an earlier choice of the same case,
    // and the cases run in lexicographic order.
    const Case = struct { length: usize, position: usize };

    // Each `length` in [0, 3] runs with every `position` in [0, `length`]. The
    // first choice changes slowest, so the cases come in this order.
    const expected = [_]Case{
        .{ .length = 0, .position = 0 },
        .{ .length = 1, .position = 0 },
        .{ .length = 1, .position = 1 },
        .{ .length = 2, .position = 0 },
        .{ .length = 2, .position = 1 },
        .{ .length = 2, .position = 2 },
        .{ .length = 3, .position = 0 },
        .{ .length = 3, .position = 1 },
        .{ .length = 3, .position = 2 },
        .{ .length = 3, .position = 3 },
    };

    var actual_buffer: [64]Case = undefined;
    var actual: std.ArrayList(Case) = .initBuffer(&actual_buffer);

    var exhaustive: Exhaustive = .{};

    while (!exhaustive.done()) {
        const length = exhaustive.intBetween(usize, 0, 3);
        const position = exhaustive.intBetween(usize, 0, length);

        actual.appendAssumeCapacity(.{ .length = length, .position = position });
    }

    try std.testing.expectEqualSlices(Case, &expected, actual.items);
}

test "Exhaustive runs cases with different numbers of choices" {
    // An earlier choice decides how many choices come after it: a length in
    // [0, 3], then that many booleans, written as 0s and 1s. The cases cover
    // all 1 + 2 + 4 + 8 = 15 sequences exactly once, in lexicographic order.
    // That takes `done` dropping the choices after the one it increments, and
    // the next case recording new choices in their place.
    const expected = [_][]const u8{
        "",
        "0",
        "1",
        "00",
        "01",
        "10",
        "11",
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111",
    };

    var cases_count: usize = 0;

    var exhaustive: Exhaustive = .{};

    while (!exhaustive.done()) : (cases_count += 1) {
        const length = exhaustive.intBetween(usize, 0, 3);

        var flags: [3]u8 = undefined;
        for (flags[0..length]) |*flag| {
            flag.* = if (exhaustive.boolean()) '1' else '0';
        }

        try std.testing.expect(cases_count < expected.len);
        try std.testing.expectEqualStrings(expected[cases_count], flags[0..length]);
    }

    try std.testing.expectEqual(expected.len, cases_count);
}

test "Exhaustive.intBetween covers a range that starts above 0" {
    // `intBetween` adds `min` to the choice it records, so a range that starts
    // above 0 gives exactly its values, in order.
    var actual_buffer: [64]u8 = undefined;
    var actual: std.ArrayList(u8) = .initBuffer(&actual_buffer);

    var exhaustive: Exhaustive = .{};

    while (!exhaustive.done()) {
        const value = exhaustive.intBetween(u8, 5, 7);

        actual.appendAssumeCapacity(value);
    }

    try std.testing.expectEqualSlices(u8, &.{ 5, 6, 7 }, actual.items);
}

test "Exhaustive.index gives every permutation in order" {
    // Taking each letter with `index` from the letters that are left gives
    // every order of "abcd" exactly once, in lexicographic order.
    const expected = [_][4]u8{
        "abcd".*, "abdc".*, "acbd".*, "acdb".*, "adbc".*, "adcb".*,
        "bacd".*, "badc".*, "bcad".*, "bcda".*, "bdac".*, "bdca".*,
        "cabd".*, "cadb".*, "cbad".*, "cbda".*, "cdab".*, "cdba".*,
        "dabc".*, "dacb".*, "dbac".*, "dbca".*, "dcab".*, "dcba".*,
    };

    var actual_buffer: [64][4]u8 = undefined;
    var actual: std.ArrayList([4]u8) = .initBuffer(&actual_buffer);

    var exhaustive: Exhaustive = .{};

    while (!exhaustive.done()) {
        // `left` holds the letters that `order` has not taken yet, in
        // alphabetical order, and `left_count` says how many there are.
        var left: [4]u8 = "abcd".*;
        var left_count: usize = 4;

        var order: [4]u8 = undefined;
        for (0..4) |position| {
            const left_index = exhaustive.index(left[0..left_count]);

            order[position] = left[left_index];

            // Every letter after the taken one moves one slot to the left, so
            // `left` stays in alphabetical order. Taking the first letter that
            // is left in every case gives "abcd", and taking the last one
            // gives "dcba".
            for (left_index..left_count - 1) |slot| {
                left[slot] = left[slot + 1];
            }

            left_count -= 1;
        }

        actual.appendAssumeCapacity(order);
    }

    try std.testing.expectEqualSlices([4]u8, &expected, actual.items);
}

test "Exhaustive.shuffle reaches every order exactly once" {
    // `shuffle` puts n items in each of their n! orders exactly once, for
    // every n in [0, 4].

    // The number of orders of 0, 1, 2, 3, and 4 items.
    const expected_orders_counts = [_]usize{ 1, 1, 2, 6, 24 };

    for (0..5) |items_count| {
        var orders_buffer: [64][4]u8 = undefined;
        var orders: std.ArrayList([4]u8) = .initBuffer(&orders_buffer);

        var exhaustive: Exhaustive = .{};

        while (!exhaustive.done()) {
            // Every case starts from 0, 1, 2, and 3, as `shuffle` requires.
            // The items are distinct, so two different orders never look the
            // same.
            var items: [4]u8 = .{ 0, 1, 2, 3 };

            exhaustive.shuffle(u8, items[0..items_count]);

            orders.appendAssumeCapacity(items);
        }

        try std.testing.expectEqual(expected_orders_counts[items_count], orders.items.len);

        for (0..orders.items.len) |first| {
            const first_order = orders.items[first][0..items_count];

            // Each order holds every item exactly once.
            for (0..items_count) |item| {
                var occurrences_count: usize = 0;

                for (first_order) |value| {
                    if (value == item) {
                        occurrences_count += 1;
                    }
                }

                try std.testing.expectEqual(1, occurrences_count);
            }

            // No two cases give the same order. Together with the count
            // above, this means that every order came up exactly once.
            for (first + 1..orders.items.len) |second| {
                const second_order = orders.items[second][0..items_count];

                try std.testing.expectEqual(false, std.mem.eql(u8, first_order, second_order));
            }
        }
    }
}

test "Exhaustive.pick, boolean, and enumTag cover every combination" {
    // `pick`, `boolean`, and `enumTag` return every combination of letter,
    // boolean, and tag exactly once, in lexicographic order.
    const Tag = enum { x, y };
    const Case = struct { letter: u8, flag: bool, tag: Tag };

    // The first choice, the letter, changes slowest, and the last choice, the
    // tag, changes fastest. `boolean` returns false before true, and
    // `enumTag` returns the tags in declaration order.
    const expected = [_]Case{
        .{ .letter = 'a', .flag = false, .tag = .x },
        .{ .letter = 'a', .flag = false, .tag = .y },
        .{ .letter = 'a', .flag = true, .tag = .x },
        .{ .letter = 'a', .flag = true, .tag = .y },
        .{ .letter = 'b', .flag = false, .tag = .x },
        .{ .letter = 'b', .flag = false, .tag = .y },
        .{ .letter = 'b', .flag = true, .tag = .x },
        .{ .letter = 'b', .flag = true, .tag = .y },
        .{ .letter = 'c', .flag = false, .tag = .x },
        .{ .letter = 'c', .flag = false, .tag = .y },
        .{ .letter = 'c', .flag = true, .tag = .x },
        .{ .letter = 'c', .flag = true, .tag = .y },
    };

    var actual_buffer: [64]Case = undefined;
    var actual: std.ArrayList(Case) = .initBuffer(&actual_buffer);

    var exhaustive: Exhaustive = .{};

    while (!exhaustive.done()) {
        const letter = exhaustive.pick(u8, "abc");
        const flag = exhaustive.boolean();
        const tag = exhaustive.enumTag(Tag);

        actual.appendAssumeCapacity(.{ .letter = letter, .flag = flag, .tag = tag });
    }

    try std.testing.expectEqualSlices(Case, &expected, actual.items);
}
