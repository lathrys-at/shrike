"""Interpreter-teardown pin (#374 design 7).

The kernel's runtime threads outlive any kernel (the process-global
runtime is never shut down); the hazard is a Python-touching task being
mid-GIL-attach while the interpreter finalizes. The clean path is
`kernel.close()` (which drains the actor), but the pin here is the UNCLEAN
path: open a kernel, attach a Python backend, do real work, and exit
WITHOUT closing — the process must still exit cleanly (no segfault, no
hang), because by exit time every spawned op has completed and parked
runtime threads hold no Python state.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

shrike_native = pytest.importorskip("shrike_native")


def test_exit_without_close_is_clean(tmp_path) -> None:
    script = textwrap.dedent(
        f"""
        import asyncio
        import shrike_native

        class StubBackend:
            modalities = frozenset({{"text"}})
            def embed_texts(self, texts):
                return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
            def model_fingerprint(self):
                return "stub:v1"
            def embedding_dim(self):
                return 4

        async def main():
            kernel = await shrike_native.async_kernel_open(
                {str(tmp_path / "collection.anki2")!r}, {str(tmp_path / "cache")!r}
            )
            kernel.attach_embedder(shrike_native.PyEmbedder.capture(StubBackend()))
            await kernel.reindex_if_needed()
            core = kernel.core_handle()
            basic = core.notetype_id("Basic")
            results = await kernel.upsert_notes(
                [(basic, 1, ["teardown pin", "no close"], [])], "error"
            )
            assert all(r[0] == "created" for r in results)
            hits = await kernel.search("teardown", 3)
            assert hits
            # Deliberately NO kernel.close(): the unclean exit path.

        asyncio.run(main())
        print("REACHED-EXIT")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"unclean teardown: rc={proc.returncode}\n{proc.stderr}"
    assert "REACHED-EXIT" in proc.stdout
