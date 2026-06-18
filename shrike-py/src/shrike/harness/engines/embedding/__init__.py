"""The embedding runtime and its in-process backends.

``runtime`` owns the backend lifecycle; ``base`` is the minimal backend protocol;
``onnx``/``clip`` are the in-process backends (sharing ``onnx_common``);
``batching`` is the batch-safety probe; ``text`` is the note-text normalization
version pin. Heavy deps (onnxruntime) stay lazily imported inside the backends.
"""
