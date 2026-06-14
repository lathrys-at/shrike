# Adversarial Team Review — Consolidated Signoff Report
**Repo:** shrike · **Pinned commit:** `fa54f8cddd7e97be4b60ab043955e2d038ffe77a` (main)
**Baseline:** green (`cargo build --workspace` clean; `pytest tests/unit` 1346 passed/6 skipped)
**Scope:** whole repo (Python harness + native Rust workspace), 17 surfaces, all 3 lenses each.
**Method:** 17 adversarial reviewers (worktree-isolated, Opus; one Sonnet-then-discarded) → two-step join-validation (broad affirm/refute by all authors, then panel re-validation of refutations; majority-to-kill, repro as tiebreaker, no auto-kill). 14 repros saved under `.team-review/repros/`.

> **STATUS: FINAL — all items resolved (NEW-S14a-1 CONFIRMED as ISSUE-AA). Awaiting your signoff. Nothing is filed/pushed until you approve.**

---

## Headline
- **4 High, 10 Medium, 14 Low/latent** validated defects (28 filed under epic #584); **1 claim killed** by the gate (`run_job`-GC). (Earlier drafts miscounted as "12 Medium" — two Low-Med items were double-counted; corrected.)
- The dominant theme is the **index-consistency invariant** (`reconcile == rebuild` / "the collection never lags the index, and a mismatch self-heals"), violated by **four independent root causes** (ISSUE-A, B→A, D, E) plus a perf hazard on the same engine (F).
- All 4 cross-seed claims (C-1…C-4) **held** under adversarial pressure; one (C-3) spawned a real new hardening finding (ISSUE-H).
- Surfaces that came back **clean / hardened-verified**: S10 (pyo3/async bridge — cancel-detach, finalize-gate, zero unsafe), FFI memory safety (S14b), FTS5 injection (S9), the strip-skip byte-identity invariant (S8a), store_media path containment + SSRF redirect/pinning (S2), note-type soundness machinery (S8b).

---

## HIGH

### ISSUE-A — Index/derived watermark over-certifies what was actually indexed → permanent silent loss (semantic AND lexical search)
**= S5-1 (keystone) + S6-2 (derived/FTS5 twin) + S8b-1's index-desync consequence + a store-contract doc gap (NEW-S14bval-1).**
`advance_watermarks` (kernel/lib.rs:1771-1783) reads the **live** `col.mod` in a separate actor job from the op's write, then stamps it onto both the index (`set_col_mod_all`) and derived (`set_col_mod`, **unconditional** — fires even with no embedder) watermarks. Three triggers reach this — a concurrent op's interleaved write, an intra-batch half-write (ISSUE-B), or a crash between write and index — leave a note in the collection but absent from the index/FTS5 while the watermark already equals `col.mod`. Both heal gates (`check_drift`, `rebuild_derived`) then go quiet → the note is **permanently** unsearchable by semantic AND substring/fuzzy/OCR. Defeats the index's only correctness guarantee.
- **Affirmed by:** S6, S7, S5, S8b, S9 (5 independent); repro `repros/s5_watermark_race.rs` RED; S8b built the derived twin (passed).
- **Fix:** capture `col.mod` in the **same actor job** as the write; a failed/partial index-or-ingest tail leaves the watermark behind (so boot drift heals). Harden `DerivedStore::set_col_mod` doc to state the "set only after rows durably committed" invariant. **ISSUE-C's fix must land here too** (a naive best-effort tail that still advances the watermark would make this strictly worse).

### ISSUE-D — `reconcile` ≠ full `rebuild` on a model swap → stale/mixed-model vectors, drift goes quiet
**= S7-1.** The per-note blake2b fingerprint never folds `model_id`; `reindex_if_needed` calls `reconcile` unconditionally after `check_drift` with no model-change→rebuild gate. A model swap (bumps no `col.mod`, changes no text) yields an empty diff → pure swap leaves `model_id`+vectors stale forever (drift re-fires as a no-op); swap+1-edit stamps the new `model_id` while unchanged notes keep old-model vectors → **silent mixed-model index**. Contradicts the documented "model_id differs → full rebuild." Embedder fingerprint side is *correct* (S11a) — fix is purely the kernel handler.
- **Affirmed by:** S6, S5, S11b, S11a; repros `repros/s7_reconcile_model_swap.rs` (+mixed) RED.
- **Fix:** gate `reconcile`→`rebuild` on a `model_id` mismatch.

### ISSUE-E — Batch-safety probe swallows NaN → `reconcile` ≠ `rebuild`
**= S11b-1 (Rust-lane only; C-4 confirmed the Python lane is safe).** `probe::drift` uses `f64::max`, which discards NaN operands, so a batch-variant model whose drift manifests as NaN is declared batch-**safe** (`safe_batch=64`) → a note's vector becomes batch-dependent → reconcile (small chunks) ≠ rebuild (64-chunks). The probe is the *only* guard for this invariant.
- **Affirmed by:** S7, S8a, S11b; repro RED (reconstruct full test at proposals).
- **Fix:** one line — treat any NaN element as `f64::INFINITY` (mirrors the existing shape-mismatch handling).

### ISSUE-F — Engine `State` mutex held across the on-disk save → concurrent search/write stalls for ≈ the full save window
**= S7-2.** `MultiModalIndex::save` (engine.rs:228) holds the single `Mutex<State>` across the multi-file usearch write; the same lock backs `search_by_modality`/`add`/`remove`. The #445 fix relieved the *orchestrator* lock but not this engine lock one layer down. Recurs on every debounced flush (60s / 100 changes) + on `close()`.
- **Affirmed by:** S5, S10, S14b (ran the timing repro RED). **Correction:** the "303×" headline is release-build-specific — state as "a concurrent search blocks for ≈ the full save window."
- **Fix:** serialize the state under the lock, write bytes outside it (or use usearch's own save concurrency); add a `VectorIndex` trait-doc clause "`save` must not block `search`." Repro `repros/s7_save_blocks_search.rs`.

---

## MEDIUM

- **ISSUE-B — `update_note_named` half-writes fields on a bad deck ref** (write-before-validate; commits fields then resolves deck, errors after — mirror the create path's validate-first). The atomicity bug; its index-desync consequence is folded into ISSUE-A but the half-write is independently fixable and must not be folded away. Affirmed S6/S5/S3; repro `repros/s8b_update_half_write.rs` RED.
- **ISSUE-C — Maintained write tail fails the whole call after the collection write committed.** = S6-1. Caller told a committed write failed → retry hits spurious Duplicate / double-write (client-corroborated by S14a). Fix = best-effort tail (warn+return successes) **bound to ISSUE-A's watermark fix**. Repro `repros/s6_repro.rs`.
- **ISSUE-G — SSRF allowlist permits 6to4 (`2002::/16`) & `3fff::/20` → fail-open to internal IPv4.** = S2-1. Live on the attacker-supplied `store_media` url path. Affirmed S1/S11b/S12 (repro RED). Fix: add both ranges to `ipv6_is_global` + boundary addrs to `IP_CORPUS` (3fff addrs must stay inside /20). Repro `repros/s2_ssrf_6to4_probe.rs`.
- **ISSUE-H — Remote-embed/describe-remote SSRF defense-in-depth gap.** = NEW-VAL-1 (born from C-3). Not attacker-reachable today (endpoints operator-config-only — verified) but: no IP classifier on the remote crates, un-revalidated redirects (vs media_fetch's `.redirects(0)`), and a permissive `start(**overrides)` sink guarded only by a hand-maintained route allowlist — "one careless `**body` from unauth SSRF." Affirmed S2/S11b(crate owner)/S12. Fix: deny `endpoint`/`api_key_env` at the sink for HTTP-sourced overrides + redirect re-vet on both remote crates + document the posture.
- **ISSUE-I — Rich-markup injection from untrusted note content → terminal spoof + content-driven CLI crash (DoS).** = S14a-1 (Medium-High). `output.py` never escapes note content into `console.print`; a malformed `[/tag]` in any field/tag/snippet crashes `note list/show/search` with an uncaught `MarkupError` (fires even on benign synced content). **Broadened:** S2 reproduced the crash via a bracket-bearing media filename → fix must also cover `media_cmd.py`/`export_cmd.py`. Affirmed S1/S2/S3; repro `repros/s14a_markup_injection.py` RED.
- **ISSUE-J — Orphan reap kills an unrelated recycled-PID process.** = S12-1. `pid_alive && port_held` checked independently — never "is the recorded PID the port holder." Recycled PID + any loopback-port holder → SIGKILLs a bystander (Windows worse: `pid_alive` hardcoded true). Affirmed S5/S13; repro `repros/s12_reap_wrongkill.rs` RED. Fix: verify the PID owns the port before terminate.
- **ISSUE-L — Non-127.0.0.1 loopback bind self-bricks (every request 421).** = S1-1 (fail-closed availability). `is_loopback` accepts all 127/8 but the guard allowlist is the fixed loopback trio. Affirmed S2/S12/S13; repro `repros/s1.py`. Fix: fold the actual bind host into the allowlist.
- **ISSUE-M — `Harness.reload()` omits the secondary-floor recalibration every other reindex path does** (stale cross-space image floor after `/reload`). = S13-1. Scope: N≥2 multi-space only (no-op at N=1). Affirmed S7/S9/S11a; repro `repros/s13_reload_recalibrate.py` RED. Fix: one-liner.
- **ISSUE-O — Post-shutdown `shrike_op`/`shrike_close` hang forever (mobile `current_thread` lane).** = S14b-1. A task spawned onto a shut-down runtime is never polled → callback never fires; finalize-gate doesn't cover (no Python interpreter in mobile). Server multi-thread lane unaffected. Affirmed S5/S10/S13. Fix: shut-down flag → fast-fail `Unavailable` completion.
- **ISSUE-AA — Cooperative-lock busy-retry contract broken on the MCP/JSON-RPC client path.** = NEW-S14a-1 (raised S14a@0.5, **CONFIRMED by S1 with a 1.0 empirical repro through the real FastMCP lowlevel handler**). `_safe_tool` re-raises `CollectionBusyError`; FastMCP `Tool.run` prefixes `"Error executing tool <name>: "` → wire text `"Error executing tool …: collection_busy: …"` → `ShrikeClient._call`'s `text.startswith("collection_busy:")` is **False** → raises generic `ServerError`, not the client-side `CollectionBusyError`. Every programmatic `except CollectionBusyError: retry` (and the CLI-against-a-daemon path) silently breaks under `--cooperative-lock`. Both existing busy tests don't bite (one hand-crafts the unprefixed wire text; the other stops at `ToolError`). The `/actions` HTTP edge is NOT affected (inspects `__cause__`). **Medium** (opt-in cooperative mode, but 100% defeated within it). Fix: sentinel-*position* search in `_call` (`text.find("collection_busy:")` then slice — note the current `split(":",1)` extraction is also wrong on the wrapped form); test must drive the real FastMCP lowlevel handler, not a hand-crafted body.

---

## LOW / LATENT / HYGIENE
- **ISSUE-K — Bad input → server-bug traceback instead of `ToolInputError`** [Low-Med]. = **S3-1 + S3-2 (ONE issue)**. Malformed `modified_since` (plain `ValueError`) + invalid regex/backref (`re.error`) hit `_safe_tool`'s catch-all → ERROR+traceback. Info-leak sub-claim **dropped** (S1+S14a). Fix: catch `ValueError`/`re.error` → `ToolInputError`. Repros `repros/s3_modified_since.py`, `repros/s3_bad_regex.py` RED.
- **ISSUE-N — `metadata_changed` fires a full tag-centroid recompute, no relevance probe, on a runtime worker** [Low-Med/perf]. = S9-2 (coalescing caps to ~1 recompute/batch). Fix: membership probe + `spawn_blocking`.
- **ISSUE-U — Recognition sweep re-enumerates the whole collection per batch** [Low-Med/perf]. = S12-2. Cost on cold-start/fingerprint-change backlog drain; steady-state cheap. Fix: enumerate once + `HashMap<i64,HashSet>` done-set + cursor.
- **ISSUE-V — Backend-alias not normalized on `start(backend=)` override → 400s + poisons the runtime kind** [Low-Med]. = S11a-1 (live `/embedding/start` restart path). Fix: apply the alias map at the override site.
- **ISSUE-W — `modalities ⊇ {TEXT}` not enforced in profiles.py** [Low-Med]. = S13-4 (promoted). Image-only `remote` entry → `RemoteBackend(modalities={image})` violates the protocol, no downstream guard. Pairs with C-1 (C-1 = ceiling ≤1; this = missing floor ≥{TEXT}). Fix: require `text` in every entry.
- **ISSUE-Q — `delete_notes`/`find_replace_notes` bypass the maintained single-op kernel write** [Low/arch]. = S3-3. Remediation (reconciled w/ S3+S6): extend `kernel.delete_notes` to return `{deleted, not_found}` in its single write job, then route the action through it.
- **ISSUE-P — `--no-dns-rebinding-protection` not honored on `/mcp`** [Low, fail-closed]. = S1-2.
- **ISSUE-R — Schema-bounds parity drift (latent hygiene)** [Low]. = S4-1…S4-5 consolidated (the Rust schemars output is never served → zero client impact; one issue, not five).
- **ISSUE-S — `--json --pretty` mutual-exclusion is order-dependent** [Low]. = S14a-2.
- **ISSUE-T — `VectorIndex::remove` trait doc says "left" vs the real "removed"** [Low, latent doc-trap]. = S14b-2 (kernel discards the count today).
- **ISSUE-X — mmprojs admitted onto a text-only-consumer managed llama** [Low]. = S13-2 (+ forces a needless TEXT-index rebuild via the fingerprint fold).
- **ISSUE-Y — Legacy config → degrade (CLI) vs refuse-boot (daemon)** [Low, narrow]. = S13-3.
- **ISSUE-Z — `rrf_fuse` NaN-weight parity divergence** [Low, DORMANT]. = S9-1. Zero reach today (no host supplies weights); gated on a future `--search-*` knob. File as must-fix-before-exposing-weights.
- **(latent) S8a-1 — `WS_RE` C0-separator byte-identity divergence** [Low]. Field path masked by anki; OCR path bypasses the normalizer (C-2). No live break. Fold into ISSUE-R or a one-line doc/fix.

---

## KILLED / CLEARED (appendix — NOT filed; evidence the gate worked)
- **run_job-dropped-future GC — KILLED.** Refuted by S5 (runtime) + S10 (bridge owner, 200/200 gc-hammered repro) and **conceded by the raiser (S13)**: `call_soon` parks a strong-ref in the loop `_ready` deque + the actor mpsc send is eager (pre-`.await`), so the release lands independent of cyclic GC.
- **C-1 (≤1 image-space) — ENFORCED** (S13+S7 pressure-tested across mixed/legacy/3-space). S7's original concern resolved.
- **C-2 (OCR bypasses normalize) — CONFIRMED** (S12+S7+S9; S8a author withdrew the severity-rise). S8a-1 stays latent.
- **C-3 (remote endpoints operator-config-only) — CONFIRMED** (S2+S11b+S12). Spawned ISSUE-H.
- **C-4 (Python probe NaN-safe) — CONFIRMED** (S11a+S7+S11b). ISSUE-E is Rust-only.
- **Cleared by adversarial sweep:** FTS5 injection (quoting holds), strip-skip byte-identity, store_media containment + SSRF redirect/pinning/size-cap, FFI guard/unsafe/Send-Sync, finalize-gate SeqCst, cancel-detaches-not-aborts, panicking-actor fail-closed, note-type soundness machinery, IPv4 SSRF classifier.

---

## COVERAGE LIMITS / BLIND SPOTS (this audit is a FLOOR, not a ceiling)
- **Not done:** a fuzzing campaign, a load test at the real 100k-note scale (perf findings F/N/U/Z are complexity-reasoned + small/derived benchmarks, not production-scale measured), cryptographic review (n/a — no crypto in scope).
- **By-design residuals (not filed):** the `--allow-remote` trust model (e.g. `/embedding/start` can spawn an operator-unintended binary path) is documented-by-design; export temp-dir umask is within the single-user/local model.
- **Repros to reconstruct at the proposals phase** (authors described the mechanism + observed result but didn't all paste full test code): ISSUE-E (probe NaN), ISSUE-N (tag recompute, GREEN proof), ISSUE-O (post-shutdown), S6-2 derived-twin assertion.
- **Does NOT replace** the user-triggered ultra/cloud review or the pre-release security/performance audits.

---

## PROPOSED HANDOFF (on your signoff)
- **One tracking epic** "Adversarial team-review (fa54f8c): index-consistency + security + hygiene" homing all children.
- **xfail branches** (one per issue with an automatable repro; strict xfail / `#[ignore]` for Rust): `xfail/issue-A-watermark`, `xfail/issue-D-reconcile-modelswap`, `xfail/issue-E-probe-nan`, `xfail/issue-F-save-lock`, `xfail/issue-B-halfwrite`, `xfail/issue-G-ssrf-6to4`, `xfail/issue-I-markup`, `xfail/issue-J-reap`, `xfail/issue-K-toolinputerror`, `xfail/issue-L-loopback`, `xfail/issue-M-reload`, `xfail/issue-O-postshutdown`, `xfail/issue-AA-busy-retry` (+ the rest as the proposals phase produces repros).
- **Severities** are panel-calibrated; the index-consistency cluster (A,D,E,F + B,C) cross-linked in the epic as one failure-surface, distinct root causes.
- **All step-1/step-2 items resolved; no open items.**

> **Awaiting your signoff.** On your go I run the proposals phase (one verified failing test per validated issue, sink-owner writes / raiser reviews), commit the `xfail/…` branches, open the epic + child issues, and push. Nothing public until then.
</content>
