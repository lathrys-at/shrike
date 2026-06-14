"""#650 angle-D oversubscription runner.

Spawn M concurrent worker processes (each = one full dedup-flow replica with
its own process-global tokio runtime), WAVES times, and tally OCR-vector
failures. Cranks the process oversubscription factor far past the issue's
72-run process-parallel experiment to test whether load ALONE (no bazel
sandbox, no shared cache) can make the second text vector go absent.

Usage: python oss_runner.py M WAVES
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

WORKER = str(Path(__file__).with_name("oss_worker.py"))


def main() -> int:
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    waves = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    total = 0
    fails = 0
    excs = 0
    fail_lines: list[str] = []
    t0 = time.time()
    for w in range(waves):
        procs = [
            subprocess.Popen(
                [sys.executable, WORKER],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(m)
        ]
        for p in procs:
            out, err = p.communicate()
            total += 1
            if p.returncode == 0:
                continue
            if p.returncode == 1:
                fails += 1
                fail_lines.append(out.strip() or err.strip())
            else:
                excs += 1
                fail_lines.append(f"rc={p.returncode} {err.strip()}")
        print(
            f"wave {w + 1}/{waves}: ran {total}, fails {fails}, excs {excs} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
    print(f"\nTOTAL runs={total} fails={fails} excs={excs} M={m} waves={waves}")
    for line in fail_lines[:40]:
        print("  FAIL:", line)
    return 0 if (fails == 0 and excs == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
