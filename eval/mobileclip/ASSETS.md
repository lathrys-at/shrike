# MobileCLIP2 spike assets (#568)

Neither the **model bytes** nor the **image bytes** used by this spike are
redistributed in this repository. `eval/mobileclip/verify.py` fetches each on
demand into the gitignored `eval/mobileclip/cache/` — only the pinned source
URLs (in `verify.py`), this attribution table, and `RESULTS.md` are committed.

## Model — MobileCLIP2 ONNX

| Item | Value |
| --- | --- |
| Export repo | [`plhery/mobileclip2-onnx`](https://huggingface.co/plhery/mobileclip2-onnx) |
| Pinned revision | `ba95759a5bdbaca53e9111e2550a76ec09c8fd9e` |
| Base models | Apple [MobileCLIP2](https://github.com/apple/ml-mobileclip) (S0/S2/B/L-14) |
| License | **Apple Sample Code License** (`apple-amlr`) — see the [base model card](https://huggingface.co/apple/MobileCLIP2-S2) and the [`apple/ml-mobileclip` LICENSE](https://github.com/apple/ml-mobileclip/blob/main/LICENSE) |
| Conversion | OpenCLIP → ONNX (`text_model` + `vision_model` dual graphs), by `plhery` for the BestPick project |

The Apple Sample Code License governs the model weights; the ONNX export inherits
it. Model bytes are **never committed** — fetched at runtime, like every other
Shrike model fixture. Whether `apple-amlr` is acceptable for a checked-in
capability profile is a call for the profile-(c) image-leg wave (flagged in
`RESULTS.md`).

## Cross-modal sanity images — Wikimedia Commons

The two images used for the shared-space cosine check are reused from the existing
eval corpora (same Commons files, already attributed in
`eval/search_quality/ASSETS.md` and `eval/multimodal/`). Bytes are fetched on
demand (with a `User-Agent`, per Commons policy) into the gitignored cache.

| Handle | License | Author | Commons page |
| --- | --- | --- | --- |
| `cat` | CC BY-SA 3.0 | Von.grzanka | [File:Felis_catus-cat_on_snow.jpg](https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg) |
| `guitar` | CC0 | Wilfredor | [File:Man_playing_an_acoustic_brazilian_guitar_(Viol%C3%A3o)_on_Marco_Zero_Square,_Refice,_Pernambuco,_Brazil.jpg](https://commons.wikimedia.org/wiki/File:Man_playing_an_acoustic_brazilian_guitar_(Viol%C3%A3o)_on_Marco_Zero_Square,_Refice,_Pernambuco,_Brazil.jpg) |
