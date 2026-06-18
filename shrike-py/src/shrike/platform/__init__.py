"""Platform infrastructure: filesystem layout, logging, daemon lifecycle, path safety.

Near-leaf process infrastructure shared across the package. Imports nothing from
the rest of ``shrike`` (the bottom of the layering: ``platform/`` ← contract
← ``harness/`` ← ``api/`` ← ``server/`` ← ``cli/``).
"""
