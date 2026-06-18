"""Native-wheel import spike.

The whole Bazel approach rests on one assumption: the native-dependency wheels
import and *run* on Bazel's hermetic CPython (python-build-standalone), on every
target platform — `anki` (bundled Rust backend), `usearch` (C++ HNSW),
`onnxruntime` (C++). If a wheel won't load against the hermetic interpreter, we
want to find out here, cheaply, before building any of the real graph.

Each check does a little real work, not just `import`, so the native libraries
are actually dlopen'd and exercised. Run as a `py_test` (`./bazel test
//tools:import_spike`); a non-zero exit fails the test.
"""

from __future__ import annotations

import sys


def main() -> int:
    # Each import loads a native library; the work below actually exercises it.
    import anki.collection  # noqa: F401  — loads the Rust `_rsbridge` backend
    import numpy as np
    import onnxruntime as ort
    from usearch.index import Index

    # usearch: build a tiny index and add a vector (C++ HNSW path).
    idx = Index(ndim=4, metric="cos")
    idx.add(0, np.array([1, 0, 0, 0], dtype=np.float32))
    assert len(idx) == 1, "usearch index add failed"

    # onnxruntime: listing providers forces the native runtime to load.
    providers = ort.get_available_providers()
    assert providers, "onnxruntime exposed no execution providers"

    print(
        f"import_spike OK — python={sys.version.split()[0]} "
        f"anki+usearch+onnxruntime loaded; ort providers={providers}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
