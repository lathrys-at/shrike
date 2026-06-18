# `bin/` — shipped/runnable product entry points

`bin/` holds the **runnable entry points** for the product: thin launchers over
the `//shrike-py/src/shrike:shrike` library, exposed as Bazel `py_binary` targets.

It is **load-bearing, not cruft.** The binaries live in their own package —
outside the `shrike` Python package — so a binary target's output path never
collides with a package subdir (the `rules_python` `py_binary` idiom). Do not
fold these back into `src/shrike/` and do not delete them.

```
./bazel run //shrike-py/bin:shrike -- info
./bazel run //shrike-py/bin:server -- --collection /path/to/collection.anki2
```

| File | Target | What it is |
|------|--------|------------|
| `shrike.py` | `//shrike-py/bin:shrike` | The CLI entry point (`shrike.cli:cli`). |
| `server.py` | `//shrike-py/bin:server` | The MCP server entry point (`python -m shrike.server`). |
| `server.py` | `//shrike-py/bin:server_embedding` | The same server with the in-process embedding backends bundled — a `testonly` variant the embedding integration tests data-dep. |

The published `shrike` console script (`pyproject.toml` /
`//shrike-py:wheel` `entry_points`) points at `shrike.cli:cli` directly; these `bin/`
launchers are the Bazel-runnable equivalents for dev and for tests that spawn a
real server.

See the [`scripts/` vs `tools/` vs `bin/` boundary](../tools/README.md#the-boundary)
for how this package relates to `scripts/` and `tools/`.
