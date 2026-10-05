const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Module ──
    //
    // The module sets no target or optimization mode, so it takes the ones of
    // the module that imports it.
    _ = b.addModule("pbt", .{ .root_source_file = b.path("src/pbt.zig") });

    const build_is_top_level = b.dep_prefix.len == 0;
    if (!build_is_top_level) {
        // ── Stop In A Dependency Build ──
        //
        // A project that depends on pbt.zig runs this `build` function as part
        // of its own build, and uses only the module above. The lint and test
        // steps below serve only pbt.zig's own development.
        //
        // Zig gives only the top-level build an empty `dep_prefix`. A dependency's
        // `dep_prefix` is its name in the other project's build.zig.zon followed by
        // a dot, such as `pbt.`.
        return;
    }

    // ── Lint ──
    //
    // ziglint comes from PATH.

    const lint_step = b.step("lint", "Check the source's formatting and lint findings");
    const lint_fix_step = b.step("lint:fix", "Format the source and fix lint findings");

    for ([_]struct { *std.Build.Step, []const []const u8 }{
        .{ lint_step, &.{ "ziglint", "." } },
        .{ lint_fix_step, &.{ "ziglint", "--fix", "." } },
    }) |entry| {
        const step, const argv = entry;

        const ziglint_run = b.addSystemCommand(argv);
        ziglint_run.setCwd(b.path("."));

        step.dependOn(&ziglint_run.step);
    }

    // ── Test ──

    const test_executable = b.addTest(.{
        .name = "pbt-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/pbt.zig"),
            .target = target,
            .optimize = optimize,
        }),
        // `zig build test -- foo bar` runs only the tests whose names contain
        // `foo` or `bar`. Without arguments after `--`, every test runs.
        .filters = b.args orelse &.{},
    });

    const test_step = b.step("test", "Lint and run the tests");
    test_step.dependOn(lint_step);
    test_step.dependOn(&b.addRunArtifact(test_executable).step);
}
