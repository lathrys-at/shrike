"""Server-local filesystem path safety — the shared capability-gate mechanism.

Several MCP surfaces touch a server-local filesystem path at the server user's
privilege under the unauthenticated-loopback trust model: ``store_media``'s
``path`` source reads a file in (#164/#170), ``export_package`` writes a
``.apkg``/``.colpkg`` out (#71), and ``import_package`` will read one in (#72).
Each is a *distinct capability with a distinct blast radius* — a single-file
read, a collection-bearing write, a whole-collection overwrite — so each opts in
through its **own** operator-allowed root list (``--media-path-root`` /
``--export-path-root`` / ``--import-path-root``). This module is the **mechanism**
those policies share: validate a root, decide the server is purely-local, and
test whether a candidate path is contained in a root list. Distinct root lists
per capability = distinct policy; one mechanism = no drift.

Two gates compose for every server-local path capability:

1. **Purely-local** (:func:`server_is_purely_local`): the bind is loopback,
   ``--allow-remote`` is off, the DNS-rebinding guard is on, and no extra
   ``--allowed-host``/``--allowed-origin`` is set. Any of those signals possible
   remote/proxied traffic, where the loopback peer is the proxy, not the real
   client — so server-local filesystem access must stay disabled.
2. **Containment** (:func:`path_within_any_root` / :func:`output_path_within_any_root`):
   the path resolves inside one of the operator's allowed roots.

The two compose: purely-local stops a remote/proxied caller from *reaching* the
capability; the roots bound *what* a permitted caller may touch.
"""

from __future__ import annotations

import ipaddress
import os

__all__ = [
    "is_loopback",
    "output_path_within_any_root",
    "path_within_any_root",
    "server_is_purely_local",
    "validate_path_root",
]


def is_loopback(host: str) -> bool:
    """True if *host* names the loopback interface (so binding is browser-safe)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def server_is_purely_local(
    host: str,
    *,
    allow_remote: bool,
    no_dns_rebinding_protection: bool,
    allowed_hosts: list[str] | None,
    allowed_origins: list[str] | None,
) -> bool:
    """Whether the server is in its default, purely-local configuration (#164).

    Gates every server-local filesystem capability (store_media ``path``, export
    ``output_path``, import source). Only when the bind is loopback,
    ``--allow-remote`` is off, the DNS-rebinding guard is on, and no extra
    ``--allowed-host``/``--allowed-origin`` was added. Any of those signals
    possible remote traffic — and behind a same-host reverse proxy / tailnet the
    loopback peer is the proxy, not the real (remote) client — so server-local
    filesystem access must stay disabled there.
    """
    return (
        is_loopback(host)
        and not allow_remote
        and not no_dns_rebinding_protection
        and not allowed_hosts
        and not allowed_origins
    )


def validate_path_root(raw: str) -> str:
    """Canonicalize and validate one operator-allowed path root at startup, or raise.

    Returns the resolved absolute real path (symlinks collapsed) used for the
    containment checks below. Capability-agnostic — the same validation backs
    ``--media-path-root`` (#170), ``--export-path-root`` (#71), and
    ``--import-path-root`` (#72). Rejects the filesystem root (``dirname(p) == p``
    is true for ``/``, a Windows drive root — confining to ``/`` is no
    confinement) and a root that isn't an existing directory (so it can't 'refuse
    everything' or spring into existence later).
    """
    resolved = os.path.realpath(os.path.expanduser(raw))
    if os.path.dirname(resolved) == resolved:
        raise ValueError(f"refusing the filesystem root '{resolved}' (confines nothing)")
    if not os.path.isdir(resolved):
        raise ValueError(f"'{raw}' is not an existing directory")
    return resolved


def path_within_any_root(path: str, roots: list[str]) -> bool:
    """Whether an **existing** ``path`` resolves inside one of ``roots`` (a read gate).

    For a path that must already exist (a file to read in — store_media's
    server-local source, import's source package). ``commonpath`` on realpath'd
    sides (not ``startswith``) so ``..``, a symlink escape, and the
    ``/srv/media-evil`` prefix-bug are all closed. A non-existent path or an
    unresolvable root yields False (fail-closed). ``roots`` is the capability's
    own allow-list; an empty list denies everything.
    """
    if not roots:
        return False
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    if not os.path.exists(real):
        return False
    return _contained(real, roots)


def output_path_within_any_root(path: str, roots: list[str]) -> bool:
    """Whether a **to-be-created** ``path`` would land inside one of ``roots`` (a write gate).

    The export/write counterpart of :func:`path_within_any_root`: the target file
    does NOT exist yet, so its own ``realpath`` can't be taken. Instead the
    **parent directory** (which must exist) is realpath-resolved — collapsing
    ``..`` and any symlink in the path leading to it — and the resolved
    ``<parent>/<basename>`` is checked for containment. So a symlinked parent
    that escapes the root, or a ``..`` segment, is caught before the write. An
    empty ``roots`` denies everything; a missing parent dir yields False
    (fail-closed — the operator must create the destination dir).
    """
    if not roots:
        return False
    expanded = os.path.expanduser(path)
    parent = os.path.dirname(os.path.abspath(expanded))
    base = os.path.basename(expanded)
    if not base:
        return False
    try:
        real_parent = os.path.realpath(parent)
    except OSError:
        return False
    if not os.path.isdir(real_parent):
        return False
    return _contained(os.path.join(real_parent, base), roots)


def _contained(real_path: str, roots: list[str]) -> bool:
    """True when ``real_path`` (already realpath'd) is inside any realpath'd root.

    ``commonpath`` not ``startswith``: ``/srv/media`` must not match
    ``/srv/media-evil``. A root that can't be resolved is skipped.
    """
    for root in roots:
        try:
            real_root = os.path.realpath(os.path.expanduser(root))
        except OSError:
            continue
        try:
            if os.path.commonpath([real_root, real_path]) == real_root:
                return True
        except ValueError:
            # Different drives (Windows) / mixed absolute-relative — not contained.
            continue
    return False
