"""Unit coverage for the cross-modal coverage matrix (#235).

The honest query-modality × target-modality matrix the harness derives for
``/status``: each cell is ``native`` (one live space embeds both modalities),
``via_derived_text`` (a ready recognizer derives text from the target into the
text space), or ``unavailable``. These tests pin the cell logic across the
deployments that differ — text-only, text+CLIP, +recognizers — and the
degenerate 0-space case, so the surface stays honest (a text+CLIP deployment
must show text↔audio as via-derived-text, never native).
"""

from __future__ import annotations

from shrike.harness import _coverage_matrix
from shrike.schemas import CoverageCell, CoverageMatrix

NATIVE = CoverageCell.NATIVE.value
DERIVED = CoverageCell.VIA_DERIVED_TEXT.value
NONE = CoverageCell.UNAVAILABLE.value


def _all_unavailable() -> dict[str, dict[str, str]]:
    return {q: {t: NONE for t in ("text", "image", "audio")} for q in ("text", "image", "audio")}


def test_embedding_down_all_unavailable() -> None:
    # No live space → every (query, target) cell unavailable, even with
    # recognizers attached (derived text needs a text space to query into).
    assert _coverage_matrix(frozenset(), frozenset({"ocr", "asr"})) == _all_unavailable()


def test_text_only_only_text_to_text_native() -> None:
    # A text-only space makes text→text native; with no recognizers every media
    # target is unreachable.
    matrix = _coverage_matrix(frozenset({"text"}), frozenset())
    expected = _all_unavailable()
    expected["text"]["text"] = NATIVE
    assert matrix == expected


def test_text_plus_clip_image_native_audio_unavailable() -> None:
    # A CLIP/omni space serving {text, image} makes every text↔image pair
    # native; audio stays unavailable (no ASR) — NOT implied native.
    matrix = _coverage_matrix(frozenset({"text", "image"}), frozenset())
    assert matrix["text"] == {"text": NATIVE, "image": NATIVE, "audio": NONE}
    assert matrix["image"] == {"text": NATIVE, "image": NATIVE, "audio": NONE}
    assert matrix["audio"] == {"text": NONE, "image": NONE, "audio": NONE}


def test_text_plus_ocr_image_via_derived_text() -> None:
    # Text-only space + OCR: text→image is reachable only through OCR-derived
    # text, so it reads via_derived_text (not native — no image space).
    matrix = _coverage_matrix(frozenset({"text"}), frozenset({"ocr"}))
    assert matrix["text"] == {"text": NATIVE, "image": DERIVED, "audio": NONE}
    # An image query itself isn't embeddable (no image space) → its row is dead.
    assert matrix["image"] == {"text": NONE, "image": NONE, "audio": NONE}


def test_describe_lights_image_via_derived_text() -> None:
    # The describe (VLM) engine lands under the kernel source ``vlm`` and its
    # prose is in the text space, so it also makes images text-reachable.
    matrix = _coverage_matrix(frozenset({"text"}), frozenset({"vlm"}))
    assert matrix["text"]["image"] == DERIVED


def test_text_plus_asr_audio_via_derived_text_not_native() -> None:
    # Text-only space + ASR: text→audio is reachable only via ASR-derived text,
    # the honesty case — it must NOT read native.
    matrix = _coverage_matrix(frozenset({"text"}), frozenset({"asr"}))
    assert matrix["text"] == {"text": NATIVE, "image": NONE, "audio": DERIVED}


def test_native_wins_over_derived_text() -> None:
    # When a space embeds both modalities natively, the recognizer's derived
    # path doesn't downgrade the cell: text→image stays native even with OCR.
    matrix = _coverage_matrix(frozenset({"text", "image"}), frozenset({"ocr"}))
    assert matrix["text"]["image"] == NATIVE


def test_errored_recognizer_does_not_light_a_cell() -> None:
    # The harness passes only READY recognizer sources; an absent/errored one is
    # simply not in the set, so its target stays unavailable.
    matrix = _coverage_matrix(frozenset({"text"}), frozenset())
    assert matrix["text"]["image"] == NONE
    assert matrix["text"]["audio"] == NONE


def test_image_query_reaches_audio_via_text_space() -> None:
    # text+CLIP+ASR: an image query is embeddable (image space) and the text
    # space is up, so it can reach audio through ASR-derived text.
    matrix = _coverage_matrix(frozenset({"text", "image"}), frozenset({"asr"}))
    assert matrix["image"]["audio"] == DERIVED
    assert matrix["text"]["audio"] == DERIVED


def test_matrix_validates_against_schema() -> None:
    # Every cell value the harness emits is a legal CoverageCell, so the wire
    # model accepts the produced dict unchanged.
    matrix = _coverage_matrix(frozenset({"text", "image"}), frozenset({"ocr", "asr"}))
    model = CoverageMatrix.model_validate(matrix)
    assert model.text.image == CoverageCell.NATIVE
    assert model.text.audio == CoverageCell.VIA_DERIVED_TEXT


# ── Per-space `native` cell (#229/#235 multi-space) ──────────────────────────


def test_dedicated_text_plus_separate_clip_text_image_native_via_clip() -> None:
    # The no-omni deployment: a dedicated TEXT space + a separate CLIP space
    # (text+image). text↔image is native — via the CLIP space, which embeds
    # BOTH — and text↔text is native via either. The union {text,image} alone
    # would (correctly here) say native, but the PER-SPACE check is what makes
    # it honest: it's native because ONE space (the CLIP) embeds both.
    spaces = [frozenset({"text"}), frozenset({"text", "image"})]
    served = frozenset({"text", "image"})
    matrix = _coverage_matrix(served, frozenset(), spaces)
    assert matrix["text"] == {"text": NATIVE, "image": NATIVE, "audio": NONE}
    assert matrix["image"] == {"text": NATIVE, "image": NATIVE, "audio": NONE}


def test_two_disjoint_single_modality_spaces_are_not_cross_native() -> None:
    # THE bug the per-space cell fixes: a TEXT-only space + an IMAGE-only space
    # (two disjoint single-modality spaces). The union is {text,image}, but NO
    # single space embeds both — so text↔image is NOT native (it's unavailable
    # without a recognizer). Each is native only on its own diagonal.
    spaces = [frozenset({"text"}), frozenset({"image"})]
    served = frozenset({"text", "image"})
    matrix = _coverage_matrix(served, frozenset(), spaces)
    assert matrix["text"]["text"] == NATIVE
    assert matrix["image"]["image"] == NATIVE
    # The cross cells are NOT native (no single space spans both).
    assert matrix["text"]["image"] == NONE
    assert matrix["image"]["text"] == NONE


def test_two_disjoint_spaces_cross_via_derived_text_with_ocr() -> None:
    # The same two disjoint spaces + OCR: text↔image is now reachable
    # via_derived_text (OCR derives image text into the text space), NOT native.
    spaces = [frozenset({"text"}), frozenset({"image"})]
    served = frozenset({"text", "image"})
    matrix = _coverage_matrix(served, frozenset({"ocr"}), spaces)
    assert matrix["text"]["image"] == DERIVED
    # image→text stays native-free but the image query reaches text natively?
    # No: image and text live in different spaces, so image→text is not native.
    assert matrix["image"]["text"] == NONE


def test_per_space_reduces_to_union_for_a_single_space() -> None:
    # N=1 byte-identical: passing one space is identical to the union fallback
    # (spaces=None) the pre-#235 tests use.
    one = frozenset({"text", "image"})
    assert _coverage_matrix(one, frozenset({"ocr"}), [one]) == _coverage_matrix(
        one, frozenset({"ocr"})
    )


def test_empty_spaces_list_all_unavailable() -> None:
    # Embedding down (no live space) → every cell unavailable, recognizers or not.
    assert _coverage_matrix(frozenset(), frozenset({"ocr", "asr"}), []) == _all_unavailable()
