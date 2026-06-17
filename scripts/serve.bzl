"""Build-time model assembly + per-profile launcher targets for `//scripts:serve` (#699).

The dogfooding launcher (`scripts/serve.py`) boots a real Shrike server from a
checked-in, path-free capability *profile*. Each profile names its onnx models by
bare *dir-name*; the server's onnx backend wants every model assembled into ONE
directory under canonical file names (``model.onnx`` + ``tokenizer.json`` side by
side — `_resolve_files`, the `p.is_dir()` branch). Each model FILE, however, is a
separate `http_file` external scattered across the runfiles tree
(``@model_minilm_int8_onnx//file/model.onnx``, ``@model_minilm_tokenizer//file/...``).

This module assembles those scattered externals into per-model directories AT
BUILD TIME (`copy_file` into ``models/<dir-name>/<file>``), then declares one
`py_binary` PER PROFILE whose ``data`` is scoped to *just that profile's* models —
so running one profile no longer drags every profile's externals. The single
``_MODEL_FILES`` table below is the ONE source of truth for which externals make up
each model dir; the macro derives both the `copy_file` rules and each launcher's
``data`` from it, so the old runtime ``_model_sources()``↔``data`` hand-sync (a
"missing runfile = a forgotten dep" footgun) is gone — adding a model is one row
here, not a row in serve.py + a hand-mirrored ``data`` entry.

The server reads the assembled dirs IN PLACE from runfiles (no runtime copy): the
launcher resolves each model dir-name to its absolute runfiles path and writes that
into the effective config it hands the server. Profiles whose models are
operator-provided ``${ENV}`` paths (jina-omni, jina-text-clip) are ``--config``-
consumed, never ``serve --profile``, so they get NO launcher target here.
"""

load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@rules_python//python:defs.bzl", "py_binary")

# -- The model-assembly table: dir-name -> list of (external file target, canonical
# name within the dir). The downloaded_file_path of each `http_file` (in
# MODULE.bazel) is the canonical name, so the copy is a pure relocation into the
# per-model dir. This is the SINGLE source of truth the macro derives copy_file
# rules AND per-profile `data` from (no serve.py-side mirror).
_MODEL_FILES = {
    # MiniLM int8 text (the text-onnx profile; the same model the integration suite uses).
    "all-MiniLM-L6-v2-onnx-int8": [
        ("@model_minilm_int8_onnx//file", "model.onnx"),
        ("@model_minilm_tokenizer//file", "tokenizer.json"),
    ],
    # embeddinggemma text leg (onnx-multispace): quant graph + its EXTERNAL weight
    # data + tokenizer. The .onnx_data MUST land beside the graph under its exact
    # name (onnxruntime resolves external data relative to the graph dir).
    "embeddinggemma-300m-onnx-int8": [
        ("@model_embeddinggemma_int8_onnx//file", "model_quantized.onnx"),
        ("@model_embeddinggemma_int8_onnx_data//file", "model_quantized.onnx_data"),
        ("@model_embeddinggemma_tokenizer//file", "tokenizer.json"),
    ],
    # MobileCLIP2-S2 image leg (onnx-multispace): text + vision graphs + preprocessor
    # + tokenizer, flat in one dir (the ClipBackend layout spike #568 verified).
    "mobileclip2-s2-onnx": [
        ("@model_mobileclip2_text_onnx//file", "text_model.onnx"),
        ("@model_mobileclip2_vision_onnx//file", "vision_model.onnx"),
        ("@model_mobileclip2_preprocessor//file", "preprocessor_config.json"),
        ("@model_mobileclip2_tokenizer//file", "tokenizer.json"),
    ],
}

def _model_dir_target(dir_name):
    """The label of the assembled per-model dir filegroup for *dir_name*."""
    return ":model_dir_" + dir_name

def assemble_model_dirs():
    """Emit, for every model in `_MODEL_FILES`, the `copy_file` rules that relocate
    its scattered externals into ``models/<dir-name>/<file>`` and a `filegroup`
    (``model_dir_<dir-name>``) collecting them. Call ONCE per BUILD package.
    """
    for dir_name, files in _MODEL_FILES.items():
        outs = []
        for (src, canonical) in files:
            # A copy rule name unique per (model, file). The out path is package-
            # relative, so under runfiles it lands at _main/scripts/models/<dir>/<file>.
            rule = "copy_{}_{}".format(dir_name, canonical).replace("/", "_").replace(".", "_")
            out = "models/{}/{}".format(dir_name, canonical)
            copy_file(
                name = rule,
                src = src,
                out = out,
                allow_symlink = False,
            )
            outs.append(out)
        native.filegroup(
            name = "model_dir_" + dir_name,
            srcs = outs,
            visibility = ["//visibility:private"],
        )

def serve_profile(name, profile, models, deps):
    """Declare a per-profile launcher `py_binary` (``serve_<name>``).

    *profile* is the profile stem (``scripts/profiles/<profile>.yml``); *models* is
    the list of model dir-names the profile needs (each must be assembled by
    `assemble_model_dirs`); *deps* are the shared library deps (serve_lib +
    onnxruntime). ``data`` is scoped to JUST this profile's YAML + its model dirs —
    running this target drags only its own externals. Tagged ``manual`` so the
    per-PR ``./bazel test //...`` lane never force-fetches the model externals; the
    non-manual ``:serve_logic_test`` is what CI exercises.
    """
    py_binary(
        name = "serve_" + name,
        srcs = ["serve.py"],
        main = "serve.py",
        imports = ["."],
        # The profile rides as a default arg, so `bazel run //scripts:serve_<name>`
        # boots that profile with no extra flags (append --seed qa / --daemon etc.).
        args = ["--profile", profile],
        tags = ["manual"],
        visibility = ["//visibility:public"],
        # The whole (tiny, fetch-free) profile filegroup — the launcher resolves a
        # profile by name from runfiles; the scoping that matters is the MODELS.
        data = ["//scripts/profiles"] + [_model_dir_target(m) for m in models],
        deps = deps,
    )
