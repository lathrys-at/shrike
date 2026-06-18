"""The embed-text normalization VERSION pin.

Since the #278 cutover the normalization itself runs in the native core
(shrike-collection's embed_text.rs); the Python reference implementation lives
in tests/oracles/embed_text_oracle.py under a byte-identity contract (the
tests/native corpus compares them exactly). This module keeps only what the
runtime still owns: the version constant folded into the index fingerprint —
bump it (and the Rust EMBED_TEXT_VERSION) whenever the normalized output
changes, including an anki tag bump whose stripper differs.
"""

EMBED_TEXT_VERSION = 1
