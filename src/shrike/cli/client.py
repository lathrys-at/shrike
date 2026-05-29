"""Back-compat shim.

``ShrikeClient`` now lives in the standalone, click-free :mod:`shrike.client`
library. This module re-exports it (and the client exception types / launch
spec) so existing imports keep working and the CLI has one import site.
"""

from __future__ import annotations

from shrike.client import (
    ServerError,
    ServerHTTPError,
    ServerSpec,
    ServerStartError,
    ServerUnreachableError,
    ShrikeClient,
    ShrikeError,
)

__all__ = [
    "ServerError",
    "ServerHTTPError",
    "ServerSpec",
    "ServerStartError",
    "ServerUnreachableError",
    "ShrikeClient",
    "ShrikeError",
]
