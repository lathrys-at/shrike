# Adversarial team review — `fa54f8c`

This directory carries the artifacts of an adversarial team code review of the whole
repository at commit `fa54f8cddd7e97be4b60ab043955e2d038ffe77a`.

- **`SIGNOFF.md`** — the consolidated, user-approved report (severities, the index-consistency cluster, killed/cleared appendix, blind spots).
- **`FINDINGS.md`** — the full per-surface findings board (every vote, refutation, and cross-check).
- **`repros/`** — the **verified reproducing tests** the reviewers ran. Each was confirmed RED (or, where noted, characterizing) against the pinned commit during the review.

Tracking epic: **#584**. Each issue's "definition of done" is: promote its repro into the proper test location as a strict-`xfail` (pytest) / `#[ignore]` (Rust) test, fix until it passes, then remove the mark.

## Issue → repro map

| Issue | Sev | Repro file | Lang | State |
|-------|-----|-----------|------|-------|
| #585 A watermark over-cert | High | `repros/s5_watermark_race.rs` | Rust (kernel) | RED, assert-correct (derived twin shares the `:1773` statement) |
| #586 D reconcile≠rebuild model swap | High | `repros/s7_reconcile_model_swap.rs` | Rust (kernel) | RED, assert-correct (+ mixed variant, commented) |
| #587 E probe NaN swallow | High | _reconstruct_ | Rust (engine-api) | mechanism in FINDINGS/SIGNOFF; RED in review (`0.0f64.max(NaN)==0`) |
| #588 F engine save-lock | High | `repros/s7_save_blocks_search.rs` | Rust (index) | RED, assert-correct (ratio build-dependent) |
| #589 B half-write bad deck ref | Med | `repros/s8b_update_half_write.rs` | Rust (collection) | RED, assert-correct (in-crate `mod tests`) |
| #590 C committed-but-errored write | Med | `repros/s6_repro.rs` | Rust (kernel) | characterizing (PASS) — invert to assert Ok-with-results for xfail; needs dev-deps `futures`,`shrike-ffi` |
| #591 G SSRF 6to4/3fff | Med | `repros/s2_ssrf_6to4_probe.rs` | Rust (kernel) | RED, assert-correct |
| #592 H remote-embed SSRF hardening | Med | _evidence-only_ | — | grep-verified (no classifier / default redirects / permissive sink) |
| #593 I markup injection | Med | `repros/s14a_markup_injection.py` | Python | RED (crash test assert-correct; the "interpreted" test is characterizing) |
| #594 J reap wrong-kill | Med | `repros/s12_reap_wrongkill.rs` | Rust (llama-server, unix) | RED, assert-correct |
| #595 L loopback self-brick | Med | `repros/s1.py` | Python | characterizing — invert (`_validate_host(bind_host)` should be True) for xfail |
| #596 M reload recalibration | Med | `repros/s13_reload_recalibrate.py` | Python | RED (`test_reload_recalibrates_after_reindex`; control passes) |
| #597 O post-shutdown hang | Med | _reconstruct_ | Rust (mobile) | mechanism in record; RED in review ("SILENT HANG confirmed") |
| #598 AA busy-retry contract | Med | _reconstruct_ | Python | RED in review (real FastMCP lowlevel handler → client raises ServerError not CollectionBusyError) |
| #599 K bad-input traceback | Low-Med | `repros/s3_modified_since.py`, `repros/s3_bad_regex.py` | Python | RED, assert-correct |
| #600 N tag-centroid recompute | Low-Med | _reconstruct_ | Python/Rust | GREEN proof-of-mechanism in review |
| #601 U recognition sweep perf | Low-Med | _evidence-only_ | — | structural (per-call full enum + per-probe clone) |
| #602 V alias not normalized | Low-Med | `repros/s11a_alias.py` | Python | RED, assert-correct (+ 2 NaN-safe control tests that PASS) |
| #603 W modalities⊇{TEXT} | Low-Med | _reconstruct_ | Python | image-only entry currently accepted; assert ProfileError |
| #604 Q delete bypasses maintained op | Low | _evidence-only_ | — | architectural divergence |
| #605 P /mcp dns guard | Low | _reconstruct_ | Python | SDK re-enables; RED in review |
| #606 R schema-bounds parity | Low | `repros/s4_review_findings.py` | Python | 25 characterizing tests (PASS = drift confirmed); hygiene, latent |
| #607 S --json/--pretty order | Low | _in S14a record_ | Python | RED in review |
| #608 T VectorIndex::remove doc | Low | _evidence-only_ | — | doc/contract divergence (latent) |
| #609 X mmproj on text-only llama | Low | _reconstruct_ | Python | characterizing (config accepted) |
| #610 Y legacy config degrade vs refuse | Low | _evidence-only_ | — | narrow; may close with #523 |
| #611 Z rrf_fuse NaN-weight parity | Low/dormant | `repros/s9_nan_parity.py` | Python | RED, assert-correct; zero reach today |
| #612 S8a WS_RE C0-separator | Low/latent | _reconstruct (s8a fuzz + anki strip)_ | Rust (collection) | RED vs oracle in review; no live path |

**Legend:** _assert-correct_ = the test asserts the fixed behavior, so it fails today (ready to mark strict-xfail / `#[ignore]`). _characterizing_ = asserts current behavior (passes today); invert the assertion before marking xfail. _reconstruct_ = the reviewer described the mechanism + observed RED but did not save the full test file — rebuild it from FINDINGS.md when fixing. _evidence-only_ = no automatable red test; the issue carries the structural evidence.

Notes:
- Several Rust repros reference an in-crate `mod tests` (private helpers) or extra dev-deps — see each file's header for placement.
- This review is a **floor**, not a replacement for a dedicated pre-release security / performance audit or a fuzzing / 100k-scale load campaign (see SIGNOFF.md "Coverage limits").
</content>
