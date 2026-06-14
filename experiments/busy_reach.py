"""#650 angle-D: is a cross-engine SQLITE_BUSY reachable on the
`texts_for_source_for_notes` read SQL, the way the kernel-engine vs host-engine
share the DELETE-journal `shrike.db`?

Replicates the derived store's connection setup (journal_mode=DELETE,
synchronous=NORMAL, busy_timeout) with TWO connections to one file, holds a
write transaction on conn A across an fsync-y commit, and reads on conn B —
measuring whether B's read surfaces BUSY (the unretried error the kernel
swallows) when A's write window outlasts B's busy_timeout.

This isolates the MECHANISM's reachability without a native rebuild. It does
NOT use the high-jobs bazel env — it asks only "can this read BUSY at all,
in-process, with the real pragmas?" If yes, the swallow at
compose_embed_inputs:1275 is a real product-bug path; the bazel env's role is
then to lengthen the write window (slow fsync under N concurrent processes).
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path


def setup(path: str, busy_ms: int) -> sqlite3.Connection:
    c = sqlite3.connect(path, isolation_level=None, check_same_thread=False)  # autocommit; we drive BEGIN
    c.execute("PRAGMA journal_mode=DELETE")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute(f"PRAGMA busy_timeout={busy_ms}")
    return c


def main() -> int:
    busy_ms = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    hold_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    tmp = Path(tempfile.mkdtemp())
    db = str(tmp / "shrike.db")

    a = setup(db, busy_ms)
    a.execute("CREATE TABLE rows(note_id INTEGER, source TEXT, txt TEXT)")
    a.execute("INSERT INTO rows VALUES (1, 'ocr', 'oxaloacetate')")
    b = setup(db, busy_ms)

    busies = {"read": 0, "ok": 0, "errs": []}

    def writer():
        # Hold a write transaction (RESERVED→EXCLUSIVE on commit) for hold_s,
        # mimicking the kernel's ingest commit window under slow fsync.
        a.execute("BEGIN IMMEDIATE")
        a.execute("INSERT INTO rows VALUES (2, 'ocr', 'condenses')")
        time.sleep(hold_s)
        a.execute("COMMIT")

    # The reader hammers the SAME SELECT shape texts_for_source_for_notes runs,
    # while the writer holds the lock. busy_timeout < hold_s means the read's
    # wait expires mid-write -> SQLITE_BUSY surfaces (the kernel's unretried
    # path).
    def reader():
        deadline = time.time() + hold_s + 0.1
        while time.time() < deadline:
            try:
                b.execute(
                    "SELECT note_id, source, txt FROM rows WHERE source=? AND note_id IN (1,2)",
                    ("ocr",),
                ).fetchall()
                busies["ok"] += 1
            except sqlite3.OperationalError as e:
                busies["read"] += 1
                busies["errs"].append(str(e))
            time.sleep(0.002)

    tw = threading.Thread(target=writer)
    tr = threading.Thread(target=reader)
    tw.start()
    tr.start()
    tw.join()
    tr.join()

    print(
        f"busy_timeout={busy_ms}ms write_hold={hold_s}s -> "
        f"reads_ok={busies['ok']} reads_BUSY={busies['read']}"
    )
    if busies["errs"]:
        print("  sample BUSY:", busies["errs"][0])
    # A BUSY here proves the read path can surface SQLITE_BUSY in-process when
    # the write window outlasts busy_timeout.
    return 0 if busies["read"] > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
