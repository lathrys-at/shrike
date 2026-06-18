"""Recognition engines: OCR today, room for ASR/describe facades alongside.

A package (not a bare module) for symmetry with ``embedding/`` and so future ASR
and image-describe facades sit beside the OCR one. The public API is re-exported
from the OCR module so callers import ``shrike.harness.engines.recognition``.
"""

from shrike.harness.engines.recognition.recognition import (
    OCR_BACKENDS,
    RecognizerBackend,
    make_describe_recognizer,
    make_recognizer,
)

__all__ = [
    "OCR_BACKENDS",
    "RecognizerBackend",
    "make_describe_recognizer",
    "make_recognizer",
]
