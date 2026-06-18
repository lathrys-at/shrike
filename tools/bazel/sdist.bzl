"""//shrike-py:sdist — a hermetic, stamped Python source distribution (#245).

hatchling builds the sdist from the explicit [tool.hatch.build.targets.sdist] include
(git-independent — see pyproject.toml), with the version stamped from STABLE_VERSION
(the git tag, via tools/workspace_status.sh) so the Bazel sdist matches the pip path.
The output name is fixed (the version isn't known at analysis time, like py_wheel);
tools/build-sdist.sh renames it to the versioned shrike_mcp-<version>.tar.gz.
"""

def _py_sdist_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".tar.gz")

    args = ctx.actions.args()
    args.add("--version-file", ctx.info_file)
    args.add("--out", out)
    args.add("--project-subdir", ctx.attr.project_subdir)
    args.add_all(ctx.files.srcs)
    args.use_param_file("@%s", use_always = True)
    args.set_param_file_format("multiline")

    ctx.actions.run(
        executable = ctx.executable._builder,
        arguments = [args],
        # ctx.info_file (stable-status.txt) carries STABLE_VERSION; depending on it
        # reruns the build when the tag-derived version changes.
        inputs = depset(ctx.files.srcs + [ctx.info_file]),
        outputs = [out],
        mnemonic = "PySdist",
        progress_message = "Building Python sdist for %{label}",
    )
    return [DefaultInfo(files = depset([out]))]

py_sdist = rule(
    implementation = _py_sdist_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
            doc = "Files staged for the sdist build — must cover the pyproject include set.",
        ),
        "project_subdir": attr.string(
            default = "",
            doc = "Directory within the staged tree that holds pyproject.toml (the " +
                  "project root). Empty = the stage root. Set to the unit dir " +
                  "(e.g. \"shrike-py\") when pyproject lives in a subdir (#731).",
        ),
        "_builder": attr.label(
            default = "//tools/bazel:sdist_builder",
            executable = True,
            cfg = "exec",
        ),
    },
    doc = "Builds a stamped Python sdist via hatchling, hermetically.",
)
