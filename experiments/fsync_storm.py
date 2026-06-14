"""Saturate the disk with fsync while the dedup-flow workers run, to lengthen
every process's SQLite commit-EXCLUSIVE window (the bazel I/O-contention
amplifier). Spawns a background fsync storm + M dedup workers."""
import os, sys, tempfile, threading, time
from pathlib import Path

def storm(stop, tmp):
    p = tmp / "storm.bin"
    fd = os.open(str(p), os.O_CREAT | os.O_WRONLY, 0o644)
    buf = b"x" * 65536
    while not stop.is_set():
        os.lseek(fd, 0, 0); os.write(fd, buf); os.fsync(fd)
    os.close(fd)

if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tmp = Path(tempfile.mkdtemp())
    stop = threading.Event()
    ts = [threading.Thread(target=storm, args=(stop, tmp), daemon=True) for _ in range(threads)]
    for t in ts: t.start()
    print(f"fsync storm: {threads} threads for {secs}s (pid {os.getpid()})", flush=True)
    time.sleep(secs)
    stop.set()
