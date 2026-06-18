"""The collection/profile registry.

Shrike points at one collection via ``--collection`` / ``SHRIKE_COLLECTION``.
Anki's own answer to "I have several collections" is *profiles*: separate,
isolated collections under Anki's base directory. This module is Shrike's
**superset** of that idea — a name → collection-path mapping where *any*
``.anki2`` path qualifies, not only ones under Anki's base dir — persisted in
``config.yml`` under a ``profiles:`` section.

Two deliberate scope lines:

- **The registry name is a friendly handle only.** The per-collection index
  namespace keys on a stable function of the collection *file path*, never the
  profile name — so this module never touches index cache paths.
- **An "active default" here is a config concept, not a server runtime
  switch.** It records which profile the per-call selector resolves to when no
  selector is passed. This module persists it; it does not route, open
  collections, or apply per-profile settings.

The optional per-profile ``embedding`` / ``cache_dir`` fields round-trip
through config but are **not resolved here** — applying a per-profile embedding
config is routing and resolving a per-profile cache dir is namespacing. They are
modeled and stored, and that is all.

This is distinct from the capability/build :mod:`shrike.harness.profiles` module (which
resolves ``embedders:``/``recognizers:`` against the compiled build). They
share the word "profile" but nothing else — collection profiles are *where the
notes live*, capability profiles are *how vectors are produced*.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


class RegistryError(ValueError):
    """A registry operation was invalid — an unknown name, a duplicate name, a
    missing path, or a default pointing at a profile that isn't registered.

    Always actionable: the message names the offending profile or path. CLI
    callers render it as a clean error; it is never a server bug.
    """


@dataclass(frozen=True)
class CollectionProfile:
    """One registered collection: a friendly ``name`` → a collection ``path``.

    ``embedding`` and ``cache_dir`` are optional per-profile overrides that
    round-trip through config but are deliberately **not resolved** by this
    module (see the module docstring). ``name`` is a handle for humans and the
    per-call selector; index identity never derives from it.
    """

    name: str
    path: str
    embedding: dict[str, Any] | None = None
    cache_dir: str | None = None


@dataclass
class Registry:
    """The in-memory view of the ``profiles:`` config section.

    ``profiles`` preserves insertion order (registration order). ``default`` is
    the name of the active-default profile, or ``None`` when none is set; it
    always names a member of ``profiles`` (enforced on every mutation and on
    load via :meth:`from_config`).
    """

    profiles: list[CollectionProfile] = field(default_factory=list)
    default: str | None = None

    # -- lookups -------------------------------------------------------------

    def names(self) -> list[str]:
        """Registered profile names, in registration order."""
        return [p.name for p in self.profiles]

    def get(self, name: str) -> CollectionProfile | None:
        """The profile registered under ``name``, or None."""
        return next((p for p in self.profiles if p.name == name), None)

    def resolve_default(self) -> CollectionProfile | None:
        """The active-default profile, or None when no default is set.

        This is the config-level fallback the per-call selector resolves to when
        a call passes no explicit selector. It does *not* route.
        """
        if self.default is None:
            return None
        return self.get(self.default)

    # -- mutations -----------------------------------------------------------

    def add(
        self,
        name: str,
        path: str,
        *,
        make_default: bool = False,
        embedding: dict[str, Any] | None = None,
        cache_dir: str | None = None,
    ) -> CollectionProfile:
        """Register a new profile. Raises :class:`RegistryError` on a duplicate
        name, an empty name, or an empty path.

        ``path`` is user-expanded and absolutized so the stored handle is
        stable regardless of the caller's cwd, but it is **not** required to
        exist (a collection may be created later, or live on removable media).
        ``make_default`` records this profile as the active default (also done
        automatically when it is the first profile registered).
        """
        name = name.strip()
        if not name:
            raise RegistryError("profile name must not be empty")
        if not path or not path.strip():
            raise RegistryError(f"profile {name!r}: collection path must not be empty")
        if self.get(name) is not None:
            raise RegistryError(
                f"profile {name!r} already registered (remove it first to re-point it)"
            )

        profile = CollectionProfile(
            name=name,
            path=_normalize_path(path),
            embedding=dict(embedding) if embedding else None,
            cache_dir=_normalize_path(cache_dir) if cache_dir else None,
        )
        self.profiles.append(profile)
        # The first profile registered becomes the default implicitly — a
        # single-profile registry always has a resolvable active default.
        if make_default or len(self.profiles) == 1:
            self.default = name
        return profile

    def rename(self, old: str, new: str) -> CollectionProfile:
        """Rename a registered profile ``old`` to ``new``, in place. Raises
        :class:`RegistryError` if ``old`` isn't registered, ``new`` is empty, or
        ``new`` is already taken by a different profile.

        The entry keeps its list position and every other field — ``path``,
        ``embedding``, ``cache_dir`` — so a rename is purely a relabel. The
        active default follows the rename if it named ``old``. There is **no
        index/cache impact**: index identity keys on the collection *path*, never
        the profile name, so this is a pure config edit.
        """
        new = new.strip()
        if not new:
            raise RegistryError("profile name must not be empty")
        existing = self.get(old)
        if existing is None:
            raise RegistryError(f"profile {old!r} is not registered")
        if new == old:
            # A no-op rename — nothing taken, nothing to change.
            return existing
        if self.get(new) is not None:
            raise RegistryError(
                f"profile {new!r} already registered (remove it first to re-point it)"
            )

        renamed = CollectionProfile(
            name=new,
            path=existing.path,
            embedding=dict(existing.embedding) if existing.embedding else None,
            cache_dir=existing.cache_dir,
        )
        # Replace in place so list position (registration order) is preserved.
        self.profiles[self.profiles.index(existing)] = renamed
        if self.default == old:
            self.default = new
        return renamed

    def remove(self, name: str) -> CollectionProfile:
        """Unregister a profile by name. Raises :class:`RegistryError` if it
        isn't registered.

        Removing the current default clears the default unless exactly one
        profile remains, in which case that one becomes the default — the
        registry never carries a dangling ``default`` that names a missing
        profile.
        """
        profile = self.get(name)
        if profile is None:
            raise RegistryError(f"profile {name!r} is not registered")
        self.profiles.remove(profile)
        if self.default == name:
            self.default = self.profiles[0].name if len(self.profiles) == 1 else None
        return profile

    def set_default(self, name: str) -> CollectionProfile:
        """Make an already-registered profile the active default. Raises
        :class:`RegistryError` if it isn't registered.

        This writes the config-level default — the persistent fallback the
        per-call selector resolves to when no selector is passed. It is not a
        server runtime switch.
        """
        profile = self.get(name)
        if profile is None:
            raise RegistryError(f"profile {name!r} is not registered — add it first")
        self.default = name
        return profile

    # -- (de)serialization ---------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Registry:
        """Build a registry from a loaded config dict's ``profiles:`` section.

        Tolerant of a missing/empty section (an empty registry). A ``default``
        that doesn't name a registered profile is dropped (treated as unset) —
        a hand-edited config can't make ``resolve_default`` point at nothing.
        """
        section = config.get("profiles") or {}
        entries = section.get("entries") or []
        profiles: list[CollectionProfile] = []
        for entry in entries:
            name = str(entry.get("name", "")).strip()
            path = entry.get("path")
            if not name or not path:
                # A malformed entry (no name/path) is skipped rather than
                # crashing every command — the CLI's add/remove rewrites it.
                continue
            emb = entry.get("embedding")
            profiles.append(
                CollectionProfile(
                    name=name,
                    path=str(path),
                    embedding=dict(emb) if isinstance(emb, dict) and emb else None,
                    cache_dir=str(entry["cache_dir"]) if entry.get("cache_dir") else None,
                )
            )
        default = section.get("default")
        registered = {p.name for p in profiles}
        if default not in registered:
            default = None
        return cls(profiles=profiles, default=default)

    def apply_to_config(self, config: dict[str, Any]) -> None:
        """Write this registry back into a loaded config dict's ``profiles:``
        section (mutates ``config``), so a subsequent ``save_config`` persists
        it.

        Always writes the structured ``{entries, default}`` shape (empty
        ``entries`` for an empty registry) so a removed-to-empty registry
        clears the on-disk section rather than leaving stale entries.
        """
        section = self.to_config_section()
        config["profiles"] = section or {"entries": [], "default": None}

    def to_config_section(self) -> dict[str, Any]:
        """The ``profiles:`` config section for this registry, or ``{}`` when
        empty.

        Persists each profile as an ordered ``entries`` list (insertion order
        preserved) plus the ``default`` name. Optional per-profile fields are
        emitted only when set, so a plain registry stays clean.
        """
        if not self.profiles:
            return {}
        entries: list[dict[str, Any]] = []
        for p in self.profiles:
            entry: dict[str, Any] = {"name": p.name, "path": p.path}
            if p.embedding:
                entry["embedding"] = dict(p.embedding)
            if p.cache_dir:
                entry["cache_dir"] = p.cache_dir
            entries.append(entry)
        section: dict[str, Any] = {"entries": entries}
        if self.default is not None:
            section["default"] = self.default
        return section


def _normalize_path(path: str) -> str:
    """User-expand and absolutize a path for stable storage.

    Kept deliberately free of any existence check: a registered collection may
    not exist yet (created later) or may live on removable media. The path is
    the index-identity input, so a stable absolute form matters; whether the
    file is present is a routing-time concern.
    """
    return os.path.abspath(os.path.expanduser(path))
