# Findings board (candidate → validated → filed)

Status legend: CANDIDATE (reported) · TRIAGED (lead sanity-checked) · VALIDATED (survived 2-step gate) · KILLED · FILED
Repro text preserved under `.team-review/repros/<surface>.py`.

## ⚠️ CROSS-SEED CLAIMS — IN PLAY FOR VALIDATION (the seed verdict is ONE VOTE, NOT adjudicated)
Correction (user): cross-seeding produced early evidence but a single seeded author's verdict must NOT
pre-settle a finding. These four enter the 2-step joint validation as OPEN CLAIMS; the seed verdict is
their initial affirm/refute data point only. Do NOT remove from play.
- **C-1 (kills S7's "image-space enforcement missing" concern):** seed verdict = ENFORCED (S13, profiles.py:606-614 + crafted 2-image-space config → ProfileError, test passed). Panel must affirm/refute that the gate truly rejects ALL 2-image configs (incl. via legacy/migrated caps, mixed onnx+remote). If a path slips past, S7's concern revives.
- **C-2 (downgrades S8a-1 from possibly-live to latent):** seed verdict = OCR/ASR text embedded RAW, bypasses normalize_for_embedding/WS_RE (S12). Panel must affirm/refute the routing (trace recognize_pending → derived ingest → text-embed). If S12 is wrong, S8a-1 is a LIVE C0-separator break on recognized text (severity rises).
- **C-3 (scopes S2-1's SSRF reach):** seed verdict = remote-embed/describe-remote endpoints operator-config-only, not attacker-influenceable (S11b, harness.py:803-845 + ctor lib.rs:435). Panel must affirm/refute that NO path (incl /embedding/start overrides, future config) lets an attacker influence base_url; +confirm the 2 residuals (un-revalidated redirects; the asymmetry). If reachable, a 2nd SSRF finding (no classifier at all on remote crates) is born.
- **C-4 (scopes S11b-1 to Rust lane):** seed verdict = Python embed_batching.py NaN-SAFE (S11a, np.max propagates NaN, 2 tests passed). Panel must affirm/refute (re-read the exact comparison; confirm no np.nanmax/sorted/NaN-dropping reduction). If wrong, a Python-lane twin of S11b-1 exists.
Also-open seeded items: **modalities⊇{TEXT} not enforced** (S11a appendix; NOT verified by S13 — open finding-candidate for profiles.py) and **run_job-dropped-future GC** (S13 appendix → S10 binding; open claim).

## NEW FINDINGS RAISED DURING VALIDATION
- **NEW-VAL-1 [Medium, security/hardening] (raised by S2 in step-1, from pressure-testing C-3):** remote-embed/describe-remote SSRF defense is structurally fragile: (a) the ONLY thing stopping unauth SSRF to an attacker-chosen endpoint is the /embedding/start route's HAND-MAINTAINED key-allowlist (server.py:401-413, excludes endpoint/api_key_env) sitting over a PERMISSIVE sink — EmbeddingRuntime.start() accepts endpoint=/api_key_env= kwargs (embedding.py:729-730,762-765) and harness.start_embedding does runtime.start(**overrides) (harness.py:623, direct splat). One careless **body / one added key = instant unauth SSRF. (b) NO IP classifier at all on the remote crates (unlike store_media's media_fetch). (c) Both remote crates follow redirects UN-revalidated (ureq default 5, no pinning; embed-remote:137, describe-remote:234) vs media_fetch.rs .redirects(0)+re-vet → a compromised/redirecting/MITM'd (plaintext http) endpoint pivots the operator-privileged request to internal/metadata. Operator-trusted TODAY (not attacker-reachable, per C-3) so defense-in-depth, but the posture is one change from a hole. Fix: deny endpoint/api_key_env at the sink for any HTTP-sourced override + add redirect re-vet (or .redirects(0)) on the remote crates + document the posture. Needs step-2 + a 2nd vote (S11b owns the crates).
- **S6-2 [Medium, correctness] (raised by S6 in step-1):** advance_watermarks falsely advances the DERIVED watermark too (lib.rs:1773 self.derived.set_col_mod(col_mod) reads the same separately-fetched stale col_mod), not just the index. Same root as S5-1 → the lexical/FTS5 twin of the silent-loss bug: a concurrent op (or intra-batch half-write) whose derived ingest_many hasn't run/failed is certified by op A's derived watermark advance → rebuild_derived (gated on derived.get_col_mod()!=col_mod) never fires → note invisible to substring/fuzzy too. FOLDS INTO S5-1's remediation (capture col.mod in the write job) but WIDENS the blast radius. Needs a step-1 vote from another author + step-2.

---

## S1 — Server, transport & trust boundary  (reported, worktree reaped)
Surface health: GOOD. All 10 custom routes correctly `@_guard`-wrapped; path-safety/purely-local
composition sound; CSRF closed (Origin check). Risk only in 2 subtle guard-composition edge cases,
both FAIL-CLOSED (over-restrictive, NOT exploitable holes).

### S1-1 — Non-127.0.0.1 loopback bind self-bricks (every request → 421)  [Medium] [TRIAGED]
- Lens: general/security (fail-closed availability). Confidence 0.9.
- Location: server.py:106-107,153-154 (allowlist = fixed loopback trio, never folds in bind host);
  server.py:851 (`_is_loopback(args.host)` accepts the bind) vs pathsafety.py:43-50 (`is_loopback` accepts all 127/8).
- Class: parser/consumer disagreement — main() accepts `--host 127.0.0.2` (no --allow-remote needed),
  but the guard allowlist only trusts 127.0.0.1/localhost/[::1] → client's `Host: 127.0.0.2:PORT` → 421 for every request.
- Predicted-correct: a loopback bind the server accepts must stay reachable — guard allowlist must include the actual bind host.
- Repro: characterizing test (5 passed) — `mw._validate_host("127.0.0.2:8372") is False` while bind accepted as loopback.
- Cross-surface seam: is_loopback lives in pathsafety.py (S2's file); bind-acceptance gate in server.py/main (S1) → route validation to **S2** (pathsafety owner) + sanity from S13 (host assembly). Validator task: try to falsify — is there any path that DOES fold the bind host into the allowlist? Is `--host 127.0.0.2` truly accepted without --allow-remote?

### S1-2 — `--no-dns-rebinding-protection` doesn't disable the guard on /mcp (only custom routes obey)  [Low] [TRIAGED]
- Lens: general/architecture (control split across two layers that disagree). Confidence 0.85. FAIL-CLOSED.
- Location: server.py:1061-1066,1082 (transport_security=None to both) vs MCP SDK fastmcp/server.py:178-183 (auto-re-enables guard when None + loopback host).
- Class: fail-open/closed inconsistency — operator turns guard off; custom routes honor it, /mcp silently re-guarded by FastMCP.
- Predicted-correct: consistent guard state across /mcp and custom routes (pass explicit settings with protection disabled, not None), or document /mcp is always guarded on loopback.
- Repro: 2 asserts pass — custom-route mw protection False while FastMCP(None, host=127.0.0.1) has protection True.
- Cross-surface: none. Validator: S13 or self-contained (S1 + SDK behavior).

### S1 appendix (noted, NOT findings)
- `/embedding/start` lets a `--allow-remote` client spawn an arbitrary binary via `llama_server`/`extra_args` (harness.py:623). WITHIN the documented --allow-remote trust model (by-design) but a sharp edge → worth a doc callout in the final report. **Flag for report's "blind spots / by-design risks" section.**
- `_guard` never checks Content-Type on POSTs (is_post=False) — confirmed SAFE (Origin check closes CSRF; no-Origin POST = non-browser, out of model).
- `/export/{token}` guarded (server.py:352) but absent from test_security.py `_CUSTOM_ROUTES` list — test-coverage gap, not a defect.

---

## S3 — Actions registry & MCP tool surface  (reported, worktree reaped)
Surface health: GOOD. Per-item batch error handling, on_duplicate/dry_run, note-type soundness,
col_mod-bump routing correctly delegated to kernel ops; path-safety gates compose soundly.
Two correctness defects (same root class), one arch/perf divergence.

### S3-1 — Malformed `modified_since` (list_notes) → unhandled server-bug traceback, not ToolInputError  [Low-Med] [TRIAGED]
- Lens: general correctness (error handling) + minor info-leak. Confidence 0.95. Repro RED (asserts correct behavior).
- Location: actions.py:658 (`datetime.fromisoformat(modified_since)`); dup at collection.py:432.
- Class: caller-input → unguarded sink → `ValueError` → `_safe_tool` (mcp_adapter.py:105-107) logs `exception("Unhandled error")` w/ traceback at ERROR + leaks "Invalid isoformat string".
- Predicted-correct: like sibling collection_query (actions.py:729-732), raise ToolInputError → WARNING, no traceback.
- Repro: scratch/test_modified_since.py (needs kharness fixture) → observed ERROR "Unhandled error in list_notes" + traceback.
- Cross-surface seam: input is S3 (actions), sink `_safe_tool` is **S1's file (mcp_adapter.py)** → route validation to S1 owner (confirm logging-policy claim + neighbor pattern). Also S4 (ToolInputError).

### S3-2 — Invalid regex/backref (find_replace_notes) → unhandled server-bug traceback; fires even on real apply  [Low-Med] [TRIAGED]
- Lens: general correctness (error handling). Confidence 0.95. Repro RED.
- Location: collection.py:643→apply_replacement→re.sub at collection.py:107/112, from action actions.py:1649 (no re.error guard). Preview loop runs on EVERY call incl dry_run=False.
- Class: same as S3-1 — re.error (ValueError subclass) uncaught → "Unhandled error in find_replace_notes" + traceback. Also bad backref `\1` no group.
- Predicted-correct: bad regex/backref → ToolInputError, WARNING, no traceback (CLAUDE.md house rule).
- Repro: scratch/test_bad_regex.py → observed ERROR "Unhandled error" + `re.error: missing )`.
- Cross-surface: route validation to **S1** (same sink). DEDUPE NOTE: S3-1 + S3-2 share one root cause → may file as ONE issue "expected-bad-input must raise ToolInputError, not surface as a server bug" with both repros.

### S3-3 — delete_notes/find_replace_notes bypass maintained single-op kernel write path (2-3 actor round trips + redundant existence pre-check)  [Low] [TRIAGED, evidence-only]
- Lens: architecture + perf (per-call, not per-item; NOT a 100k N+1). Confidence 0.8 divergence / 0.5 worth-fixing.
- Location: actions.py:1564,1572 (delete_notes), 1649,1669 (find_replace_notes). Unused maintained op: AsyncKernel::delete_notes (lib.rs:1900, does delete+drop_note_sidecars in ONE op). wrapper.delete_notes (collection.py:560-569) adds its own find_notes("nid:..") existence pre-check (3rd round trip).
- Class: divergence from stated invariant "write actions route through maintained kernel ops". Action uses raw wrapper.delete_notes + separate kernel.forget_notes instead. Consistency preserved (failed sidecar → next-boot reconcile), so NOT a correctness defect.
- Evidence: kernel.delete_notes exercised only by Rust unit tests, never the action. The #476-class "policy ported without single-op batching" smell.
- Cross-surface: **S6** owns whether kernel.delete_notes should be the action entry point → route there.

### S3 appendix (noted, NOT findings)
- store_media lacks the belt-and-suspenders `server_purely_local` re-check that export_package(actions.py:528)/import_package(actions.py:2230) have — store_media (actions.py:2037-2043) relies solely on roots being emptied upstream (server.py:1142-1149). Sound today, asymmetric/fragile. Confidence 0.45. **Cross-surface w/ S2** → mention to S2 validator. **Flag for report hardening suggestions.**
- Duplicated modified_since parse model (actions.py:656-661 vs collection.py:430-435) — both carry S3-1 bug.
- find_replace literal dry-run "exactness" claim vs Anki Rust matcher on Unicode case-folding — unverified, confidence 0.2.

---

## S2 — Media, SSRF & path-safety  (reported, worktree reaped)
Surface health: STRONG (production SSRF guard + store_media containment now entirely Rust; Python
pathsafety.py only for import/export). Path-safety gates sound (purely-local composition, prefix-collision,
symlink-parent, empty-root fail-closed, export temp+rename ALL verified clear). ONE real security defect.
NOTE: S2 CORROBORATED S3's appendix concern — confirmed store_media cannot half-enable roots (non-purely-local
server gets [] roots; export/import re-check at call time) → S3 store_media appendix item is sound-today, low prio.

### S2-1 — SSRF allowlist permits 6to4 (2002::/16) & 3fff::/20 IPv6 → fail-open to internal IPv4  [Medium/High-as-control] [TRIAGED — strongest finding]
- Lens: SECURITY (SSRF / weakened fail-closed control). Confidence 0.9 (defect) / 0.5 (real-world exploit needs 6to4 route). Repro RED (Rust, pure-function).
- Location: native/shrike-kernel/src/media_fetch.rs:83-114 (ipv6_is_global) via ip_is_allowed(:117)←resolve_public_ip(:127)←fetch_media_url(:161). LIVE in production store_media url path (actions.py:2038 → media_fetch::prepare_media_item → fetch_media_url).
- Class: weakened control — allowlist is a strict subset of "globally routable" EXCEPT 2002::/16 & 3fff::/20 where it fails open. 2002:7f00:1:: = 6to4 of 127.0.0.1; 2002:a00:1:: = 10.0.0.1; 2002:c0a8:1:: = 192.168.0.1.
- Predicted-correct: ip_is_allowed must return false for 2002::/16 and 3fff::/20 (Python is_global = False for all, verified CPython 3.12).
- Repro: native/shrike-kernel/tests/s2_ssrf_6to4_probe.rs → panics "2002::1 is NON-global per Python … but the Rust SSRF allowlist permitted it". Saved: .team-review/repros/s2_ssrf_6to4_probe.rs.
- Why undetected: parity test tests/native/test_media_url_fetch.py::test_classifier_parity_corpus maps to ip_is_allowed (anki_core.rs:443-447) but IP_CORPUS (:34-97) has no 2002::*/3fff::* → test-that-doesn't-bite. Adding either addr turns it red.
- Fix: add 2002::/16 + 3fff::/20 to ipv6_is_global private set; add boundary addrs to IP_CORPUS.
- Cross-surface: **S11b** owns remote-embed (shrike-embed-remote, ureq) — does it reuse ip_is_allowed (inherits gap) or have its own classifier? → route validation to S11b owner (independent confirm + check remote-embed parallel).

### S2 appendix (noted, NOT findings)
- _safe_media_name Py/Rust whitespace divergence: Rust safe_media_name (media.rs:22-32, #382-hardened) trims trailing ws / treats " .. " as empty; Python _safe_media_name (collection.py:67-75) was NOT given #382 fix → returns "  ", " .. " literally. Used in /media route (server.py:343) + image resolver (server.py:82). NO traversal (OS doesn't collapse ".. "+trailing-space to parent — verified). Parity defect 0.8, vuln <0.1. **Flag for report defense-in-depth.**
- Export temp-dir default-umask world-readable on multi-user host until reaped — within single-user/local trust model (cache dir per-user). Not a finding.
- Export /export/{token} concurrent-GET double-stream — benign (same caller, secrets-random token).
- CLEARED (verified sound): store_media Rust containment (Path::starts_with on canonicalize, component-aware, /srv/media vs /srv/media-evil closed); export write-gate ..-basename (abspath normalizes before dirname/basename); purely-local composition (cannot half-enable); SSRF redirect per-hop re-vet+pin+cap; size-cap placement (encoded before decode; streaming take(MAX+1)); IPv4 classifier (no divergence).

---

## S10 — PyO3 binding & async bridge  (reported, worktree reaped) — CLEAN, NO FINDINGS
Surface health: STRONG / well-engineered. Reviewer read whole shrike-py crate, traced every
pyfunction/pymethods op entry→spawn_op→kernel→one-wake bridge, ran adversarial repros against the built ext.
VERIFIED SOUND adversarially: cancel-detaches-not-aborts (cancelled upsert still creates note — the
"never abort a write" invariant); run_job-after-close errors cleanly (NativeInternalError, no hang);
GIL released across every blocking/await hop; ZERO unsafe/raw-ptr/transmute in prod; Send/Sync derived
(no hand-asserted impls); finalize-gate SeqCst handshake total-order proof walked; set_result defers
done-callbacks so __call__ poll-under-Mutex can't re-enter (no self-deadlock/double-resolve). 34 bridge/teardown tests pass.

### S10 residuals (NOT findings)
- run_job re-entrancy is an unenforced documented contract (async_kernel.rs:986) — hard to trigger from typed surface (sync callable can't await a kernel op); misuse-prone-interface note only.
- DRAIN_DEADLINE=5s (finalize_gate.rs:94) deliberate documented residual (wedged backend >5s reverts to pre-gate SIGABRT class) — accepted tradeoff.
- **Cross-surface → S5:** AsyncKernel::close (async_kernel.rs:1027) calls kernel.index().save() directly inside the spawned op (not spawn_blocking) — whether that fs write lands on a runtime worker is orchestrator's concern (S5/S7). AND the S5↔S10 sync-op invariant is satisfied ONLY because client sync #33/#362 isn't wired (anki runtime-spinning services never dispatched, pinned by runtime_singularity). When #362 lands, the actor's inline-job execution of ensure_open/reopen/close is the place to re-verify. **Route to S5 validator as a seam to confirm (not a current defect).**

---

## S6 — Kernel actions & write ops  (reported, worktree reaped)
Surface health: GOOD on perf — fused search/cross-space/neighbor-dedup all hoist per-candidate reads
into ONE batched note_dicts/note_texts/derived_field_rows; index_written is single-txn (ingest_many) +
one batched embed (honors #445). ONE proven correctness defect.

### S6-1 — Maintained upsert/delete fail the WHOLE call when the index/embed tail fails — for an already-committed write (committed-but-errored window)  [Medium] [TRIAGED]
- Lens: general/correctness (partial-failure + inconsistency-with-siblings). Confidence 0.85. Repro PASSES (characterizing the window).
- Location: lib.rs:1759 (upsert_notes_wire `self.index_written(&written).await?`), lib.rs:1853 (upsert_notes NoteSpec `?`), lib.rs:1906 (delete_notes `drop_note_sidecars(...).await?`). Tail bodies: index_written lib.rs:1659 (embed 1690-1693, ingest_many 1719, advance_watermarks 1720). Prod reach: async_kernel.rs:399 propagates to Python.
- Class: partial-failure / committed-but-errored + a control siblings DON'T skip. Note committed to collection, THEN embed/index/derived Err → `?` propagates → MCP isError. A transient embed/network hiccup fails the call; caller retry → spurious Duplicate error (on_duplicate=error) or double-write (allow).
- Predicted-correct: per CLAUDE.md "Vector index and consistency" §2 ("index update failure logs a warning but doesn't fail the tool call") and EVERY sibling maintained op (collection_prune lib.rs:2078, migrate_note_type lib.rs:2321, find_replace_note_types lib.rs:2269, field-metadata lib.rs:2289, metadata_tail lib.rs:2119 — all best-effort warn-and-return). The write tail should log+warn, advance/leave watermarks for next-boot reconcile, and RETURN the successful per-item results.
- Repro: native/shrike-kernel/tests/s6_repro.rs (FailingEmbedder) → PASSED: upsert_notes_wire returns Err yet a follow-up dry_run upsert reports Duplicate (note committed). Saved: .team-review/repros/s6_repro.rs. (needs dev-deps futures + shrike-ffi.)
- Cross-surface: **S3** owns actions.py upsert_notes (actions.py:1115 → upsert_notes_json) — S6 confirmed it does NOT catch this; fix belongs in the kernel tail. CONTRACT DECISION (best-effort tail vs fail-the-call) → route to S3 validator + lead. RELATED to S3-3 (both write-path/maintained-op divergences).

### S6 appendix (noted, NOT findings)
- metadata_changed (lib.rs:1888) unconditionally tag_refresh.request() (line 1896); metadata_tail (lib.rs:2119) routes deck-rename/find_replace_note_types/field-metadata through it. TagRefresher::run_once (tag_centroids.rs:309) has NO relevance probe beyond "embedder attached" → every fired refresh = full note_tag_rows()+note_count()+whole-collection recompute() (O(collection) at 100k). The per-op tail only REQUESTS (O(1), good — documented coalesced over-trigger, NOT a tail violation) BUT the "cheap relevance probe" the #445 rule names is absent from the BACKGROUND TASK — a deck-rename burst on 100k still drives full centroid recomputes. **Cross-surface S7/S9 (tag-centroid perf); flag for report.**
- Kernel::search (lib.rs:2416) maps score Option<f64>→f64 via unwrap_or(0.0), collapsing lexical-only (unscored) into cosine-0.0 for search()'s direct callers (KernelIndexView/sync); wire path keeps Option. Low.
- note-type structural ops (lib.rs:2205-2336) carry NO tail — verified correct by omission (col.mod mismatch drives next-boot drift reconcile+rebuild).

---

## S4 — Wire schemas (binding + canonical)  (reported, worktree reaped)
Surface health: GOOD structurally — discriminator wiring, tag values, field optionality correct + well-tested.
All findings are NUMERIC-BOUNDS PARITY DRIFT (the existing contract test test_schema_contract.py explicitly
skips numeric bounds). ALL confidence 1.0 (code-confirmed) BUT ALL LATENT (no active production bug — values
always in-range at runtime). LEAD TRIAGE: likely DEDUPE into ONE "schema bounds parity" issue; severities
probably calibrate DOWN at validation (S4-1/2 "Medium" are latent-defensive). Repro: .team-review/repros/s4_review_findings.py (25 tests, all pass = divergence confirmed).

### S4-1 — ExportPackageResult.note_count: Python unbounded int vs Rust u32  [Medium→likely Low at validation] [TRIAGED]
- schemas.py:1353 (int) vs shrike-schemas/lib.rs:1169 (u32). Python admits -1/>u32; Rust round-trip raises NativeInputError. LATENT: actions.py:541 always int(u32-from-Rust). Predicted-correct: Python ge=0 + doc u32 ceiling. Cross-surface S3 (produces it).
### S4-2 — ExportPackage{Path,Url}.bytes (+note_count): Python int vs Rust u64/u32  [Medium→likely Low] [TRIAGED]
- schemas.py:1370-1381 vs lib.rs:1182-1195. LATENT (bytes=getsize always ≥0). Predicted-correct: ge=0.
### S4-3 — ServerStatus.wire_protocol_version: int vs u32  [Low] [TRIAGED]
- schemas.py:1132 vs lib.rs:1048. LATENT (const=1). Predicted-correct: ge=0.
### S4-4 — FieldMetadataInput.size: Python ge=1 vs Rust NO bound  [Low] [TRIAGED]
- schemas.py:475 vs lib.rs:355 (Option<i64> no bound). DIRECTION MATTERS: Python enforces ge=1 (rejects 0); Rust SCHEMA (what MCP tools/list advertises) lacks minimum → Rust-schema consumers get inaccurate type info. Predicted-correct: Rust schemars minimum:1.
### S4-5 — FieldOp/TemplateOp position: Python ge=0 vs Rust NO bound (4 fields)  [Low] [TRIAGED]
- schemas.py:439,461,507,529 vs lib.rs:330-347. Same class as S4-4 (Python enforces, Rust schema doesn't advertise).

### S4 appendix (noted, NOT findings)
- UpsertNoteOk/NoteTypeOk/UpsertDeckOk: Python ONE model w/ Literal['created','updated'] vs Rust TWO variants (Created/Updated) — passes contract test (identical field shapes), house-style inconsistency only, no runtime bug.
- SearchMatch.provenance default [] but docstring "always non-empty" — docstring-vs-schema gap, neither side enforces.
- NoteInput.fields/Stats.decks_summary: Python dict (insertion order) vs Rust BTreeMap (sorted) — JSON keys unordered, not a bug; caller relying on order through Rust round-trip surprised.

NOTE: S4-4/5 (Rust schema missing bounds it should advertise) are the more meaningful direction — the MCP-advertised schema is inaccurate. S4-1/2/3 (Python looser than Rust) are defensive-only since Python receives these FROM Rust.

---

## S11b — Rust engine crates  (reported, worktree reaped)
Surface health: GOOD on TLS (rustls+webpki-roots) + secrets (header-injection guard at ctor, no secret logging,
non-Debug structs). ONE High correctness defect; remote-SSRF seam refuted-but-with-residuals.

### S11b-1 — Batch-safety probe declares a NaN-under-batching model "safe" → reconcile ≠ rebuild (silent index corruption)  [HIGH] [TRIAGED]
- Lens: general/correctness (defeats the index core invariant — the probe's whole reason to exist). Confidence 0.85. Repro RED (deterministic).
- Location: native/shrike-engine-api/src/probe.rs:453-467 (`drift`), via probe_chunks(:427), probe_max_safe_batch/probe_image_max_safe_batch, max_probe_drift/max_probe_image_drift.
- Class: float comparison swallowing NaN. `drift` does `max = max.max((finite - NaN).abs())` = `max.max(NaN)`; Rust f64::max RETURNS THE NON-NaN OPERAND → NaN drift discarded → total ≤ BATCH_DRIFT_TOL → probe_chunks returns items.len()=64 ("safe") → WithPolicy safe_batch=64 → Blocking batches up to 64 → note vector depends on batch-mates → reconcile (small batches) ≠ rebuild (batches of 64).
- Predicted-correct: a batch-variant model (batched ≠ serial, NaN or not) must probe to safe_batch==1 and max_probe_drift > tol.
- Repro: scratch #[test] in probe.rs test module — `probe_max_safe_batch` returned 64 (asserted ==1); max_probe_drift = 0 for a NaN-variant engine. Standalone confirm: (1.0-NaN).abs()=NaN; max.max(NaN)=0; 0<=1e-3=true. **NOT pasted inline by agent — RECONSTRUCT at proposals (NaN-producing mock Embedder + assert probe→1).**
- Fix (one-liner): `drift` should treat any NaN element as INFINITE drift (`if x.is_nan()||y.is_nan() { return f64::INFINITY }`, mirroring the existing mismatched-shape→INFINITY handling). Existing tests still pass.
- Why undetected: existing variant_engine_probes_to_serial test only exercises a FINITE shift; the NaN path is untested + unguarded. Spiked probe set is built to drive int8 magnitude extremes (exactly what triggers NaN/inf).
- Cross-surface: **S11a** (Python sibling shrike/embed_batching.py) — does it share the NaN-blind comparison? (numpy max PROPAGATES NaN, so Python lane MAY be safe — S11a must confirm; seed into S11a Wave-3 prompt). Downstream corruption lands in S7 (index orchestrator).

### S11b cross-surface verdict + residuals
- **S2-1 remote-embed SSRF seam: REFUTED as attacker-reachable.** Neither shrike-embed-remote nor describe-remote applies ANY SSRF/IP classifier — base_url handed straight to ureq (embed-remote:134,141-149; describe-remote:226,238-247); ureq follow_redirects would chase internal redirects unguarded. BUT endpoint is OPERATOR-CONFIG-ONLY (verified harness.py:803-845, pyo3 ctor lib.rs:435 — NOT overridable via /embedding/start body), so reaching internal is the operator's own config choice (unlike store_media url = attacker-supplied → guarded). Confidence not-a-hole: 0.75.
  - RESIDUAL 1: redirects followed UN-revalidated → a compromised/redirecting trusted endpoint pivots to internal with no hop check (media path defends this; here absent). **Flag for report.**
  - RESIDUAL 2: asymmetry — if a future change ever lets /embedding/start or any unauth route influence base_url → live SSRF, zero defense. **Flag for report + recommend documenting the "endpoint trusted, no SSRF guard, redirects unchecked" posture.**

### S11b appendix (lower-confidence, NOT findings)
- Zero-width text vector accepted where image path rejects it (embed_chunk :308-319 vs embed_one_media :409-413 whose comment claims a text-path ndim==0 guard that DOESN'T EXIST). 0.45 — needs misbehaving endpoint; downstream guard may exist (S11a/S7).
- Remote text response `index` permutation not validated (embed_chunk :306-307 sorts by index but doesn't check 0..n perm) → duplicate/constant indices misassign vectors. 0.3 (operator-trusted endpoint).
- CLIP center-crop when crop_size>size leaves zero-padded edges (clip.rs:216-247) — quality-only, unusual config. 0.2.
- Probe serial_reference uses v.pop() (last vector, no arity check) (probe.rs:441). 0.2 latent.

---

## S7 — Index orchestration & multi-space  (reported, worktree reaped)
Surface health: TWO High defects (both RED repros). ≤1-image-space write path holds structurally; floor direction correct.

### S7-1 — reconcile != full rebuild on a model swap → stale/mixed-model vectors, drift goes quiet  [HIGH] [TRIAGED]
- Lens: general/architecture (violates the load-bearing reconcile==rebuild invariant). Confidence 0.9. TWO repros RED.
- Location: index_orchestrator.rs:1174-1235 (reconcile_with_mode; empty-diff branch 1192-1199; model_id update 1219); via lib.rs:1078-1121 (reindex_if_needed calls reconcile unconditionally after check_drift); harness.py:615-637,892-894 (start_embedding→_drive_boot_reindex).
- Class: drift-handling gap. note_hash (1174.. lines 49-79) NEVER folds model_id; a model swap (/embedding/start new model) bumps no col.mod + changes no text → reconcile_diff empty → empty-diff branch advances col_mod + save_meta_only + LEAVES model_id+vectors untouched. reconcile only falls back to rebuild when reconcile_diff returns None (no prior hashes) — NO model-change gate. detach_embedder (lib.rs:767-775) saves but doesn't clear hashes → stop→start preserves the triggering hashes.
- Two variants: (a) PURE swap (same col_mod): model_id left stale "model-A" → check_drift reports drift FOREVER, every reconcile a no-op, wrong vector space permanently. (b) swap + 1 edit (col_mod bumps): non-empty-diff branch STAMPS model_id="model-B" → drift reports CLEAN but 2/3 unchanged notes keep OLD-model vectors → MIXED-MODEL index, undetectable (the insidious one).
- Predicted-correct: a model swap must re-embed every note into the new space (== full rebuild) + stamp model_id.
- Repro: s7_reconcile_model_swap.rs (3/3 diverge, model_id stale) + ..._mixed.rs (2/3 stale, drift clean). Saved: .team-review/repros/s7_reconcile_model_swap.rs.
- Why undetected: invariant test reconcile_matches_rebuild_end_state (index_orchestrator.rs:1465) only tests same-model col_mod bump — model-change case is the precise gap (test-that-doesn't-bite).
- Cross-surface: same root cause hits image-primary route (reconcile_image_route lib.rs:1127-1153) on CLIP model swap. **PAIRS with S11b-1** (both reconcile≠rebuild, different root cause). Validator: S11a/harness owner (model-swap operational flow) or S5 (reindex path).

### S7-2 — engine State mutex held across the on-disk save → every concurrent search/write stalls full save window (303×)  [HIGH/perf] [TRIAGED]
- Lens: performance (lock across I/O on latency path; violates "never hold a lock across file writes"). Confidence 0.85. Repro RED (timing).
- Location: shrike-index/src/engine.rs:228-257 (MultiModalIndex::save takes self.state.lock() at 232, holds across every sub.index.save(tmp)+rename); same self.state mutex taken by search_by_modality(:328), add(:273), remove(:294). Driven by index_orchestrator.rs:459-514 save→engine.save on debounced/burst flush (DebouncedSaver::flush_background 641-649) + close() sync flush.
- Class: lock-across-I/O / hot-path contention. #445 fixed IndexOrchestrator::save's OWN lock (snapshot-then-write-outside) but engine.save serializes the ENTIRE State behind one Mutex for the full multi-file usearch write. Background flush every save_delay(60s)/save_threshold(100) → recurring search freeze.
- Predicted-correct: serialize vectors under lock (or use usearch concurrency), write bytes OUTSIDE the State lock.
- Repro: s7_save_blocks_search.rs (60k×256d): uncontended 158.9µs, save alone 48.1ms, contended search 48.2ms = 303×. Saved: .team-review/repros/s7_save_blocks_search.rs.
- Cross-surface: this is the engine UNDER the S10 close()/save() flag. spawn_blocking moves the write off the runtime worker but NOT the engine-lock contention vs in-flight searches. Validator: S5 (runtime/close path) or self (S7 owns engine.rs).

### S7 verified the S10 seam: REFUTED for orchestrator's OWN lock (IndexOrchestrator::save correct, #445 holds — save_guard + snapshot shared under lock, drop, then engine.save+write_atomic outside) — real hazard is one layer down = S7-2. "Real in effect, wrong in location."
### S7 appendix: ≤1-image-space write path holds (image_primary_keyed returns FIRST image-capable space; only one holds image vectors). BUT S7 did NOT find profiles.py enforcement of ">1 image space = config error" (#580) within surface → **CROSS-SURFACE to S13 (profiles): if two image-capable spaces slip past validation, build_cross_space fuses two image rankings + retired relative gate no longer guards. SEED into S13 Wave-3 prompt.** Also: apply_image_floor direction correct; calibrate_secondary_floors O(collection) on build path (permitted); check_drift text-only-v1 correct.

---

## S8b — Collection write, note-types & adapter  (reported, worktree reaped)
Surface health: note-type soundness machinery SOUND (constructed several defeating sequences — rename cascades, mid-list rename, reposition edges, cloze/empty maps — all behaved correctly, matched anki by-ord migration). ONE High defect in the note UPDATE path.

### S8b-1 — update_note_named half-writes fields on a bad deck ref, returns error, silently desyncs the index  [HIGH] [TRIAGED]
- Lens: general/architecture (non-atomic write) + cross-surface index consistency. Confidence 0.95 (half-write) / 0.8 (batch index-staleness). Repro RED.
- Location: write.rs:322-336 (update_note_named): adapter.update_note(&note)? at :322 COMMITS fields/tags FIRST, THEN resolves deck at :324-329; a numeric/#id ref resolving to no deck → resolve_deck_ref Ok(None) → Err. Fields already persisted, col.mod bumped. Contrast create_note_named resolves deck BEFORE write (write.rs:167) — the asymmetry IS the bug.
- Class: partial write on failure branch / write-before-validate.
- Cross-surface (raises severity → route S3 + S6): upsert_notes_wire only puts Created/Updated ids in `written`; the half-written item returns Error so NO new vector. BUT if the batch has ANY success, index_written runs its tail → advance_watermarks→index_set.set_col_mod_all(col_mod) (lib.rs:1779) advancing the watermark PAST the col_mod the half-write produced → next boot drift sees watermark==col_mod → NEVER reconciles the stale vector → PERMANENT silent index/collection divergence (the "collection never lags index + mismatch self-heals" invariant broken). Single-item case: index_written returns early (written empty) → drift WOULD catch it. Batch-dependent but real.
- Predicted-correct: an item returning error must not have mutated the note — resolve deck + all failable preconditions BEFORE update_note (mirror create path).
- Repro: s8b_update_half_write.rs → fields=["NEW front","NEW back"] despite error. Saved: .team-review/repros/s8b_update_half_write.rs.
- Validator: S6 (kernel owner, confirm the batch advance_watermarks index-staleness end-to-end with embedder) + S3 (actions upsert). RELATED to S6-1 (write-path error handling) + the consistency invariant.

### S8b appendix (noted, NOT findings)
- Non-atomic multi-RPC writes are a PATTERN (S8b-1 worst): rename_tag note-scoped (write.rs:409-410 remove-then-add), update_note_tags add/remove (write.rs:379-384), migrate_note_type scm-bump+readback+change (note_types.rs:789-808). Each anki RPC = own txn; no cross-RPC transact. Lower sev (failure points rarer). Class: validate/resolve all failable preconditions before first mutating RPC. **Consolidate as one architectural finding for report.**
- migrate_note_type manual scm pre-bump redundant (anki's change_notetype_of_notes_inner already does require!+set_schema_modified) + bypasses op/undo via raw db_execute (note_types.rs:789-808, adapter.rs:516-521). Internally consistent (actor-serialized, no TOCTOU). Low conf, no corruption proven.
- `where id in ()` latent hazard (write.rs:548-549 find_replace, :587 delete_note_types) — guarded by callers today, unreachable; latent footgun.
- set_note_tags_bulk one get_note RPC per note (adapter.rs:728-733, 1000 reads at cap) — acceptable per #445 (removed per-note journal commit; reads ≪ fsyncs). Do NOT re-flag.

---

## S5 — Kernel core, actor & runtime  (reported, worktree reaped)
Surface health: ONE High concurrency defect (the keystone). Panicking-actor non-defect verified (fails closed). Sync-op invariant confirmed safe-by-reachability-only.

### S5-1 — Concurrent ops falsely advance the index/derived watermark → permanent silent note loss the drift mechanism can't heal  [HIGH] [TRIAGED — KEYSTONE, shares root with S8b-1]
- Lens: concurrency / general correctness (silent data loss; defeats the self-healing drift = index's only correctness guarantee). Confidence 0.9. Repro RED (deterministic, gated embedder).
- Location: advance_watermarks lib.rs:1771-1783 (reads LIVE col.mod via self.collection.run(|core| core.col_mod()) at :1772, not the col.mod captured with THIS op's write), reached from index_written(:1720), drop_note_sidecars(:1874), metadata_changed(:1889), recognition sweep(:1624). Suppression site: index_orchestrator.rs:389 (col_mod != current → drift; ==→ no reconcile).
- Class: race — actor serializes individual JOBS, not multi-job op TRANSACTIONS. Each maintained op = write job, then .await embed/index, then col.mod read in a SEPARATE job. Concurrent op B's write job interleaves (ops spawned independently via spawn_op on multi-thread runtime — real concurrency, exercised by slow_recognition_concurrency_is_bounded). Op A's advance_watermarks reads a col.mod already reflecting B's write. If B's index maintenance fails (or crash before B indexes) → B's note in collection but absent from index, watermark ALREADY == col.mod → check_drift false → NO reconcile EVER → note permanently unsearchable (semantic + lexical/OCR via identical derived watermark).
- Predicted-correct: watermark must never advance past col.mod values whose writes this op did not index — capture col.mod in the SAME actor job as the write (or serialize write+index+watermark). A failed/partial index op must leave the watermark behind so boot drift heals it.
- Repro: s5_watermark_race.rs (gated embedder; alpha indexed, bravo not; primary.col_mod()==live col_mod confirms false-advance; reindex_if_needed()==false = no heal). Saved: .team-review/repros/s5_watermark_race.rs.
- Why undetected: detach_degrades_and_reattach_recovers only covers watermark-stays-put-on-detach; nothing covers the false-advance.
- Cross-surface: advance_watermarks body at lib.rs:1771 is just inside S6's [1736-2455] → route fix to **S6**. SHARES ROOT CAUSE with S8b-1 (half-write trigger) → consolidate as "watermark over-certifies what was actually indexed" with multiple triggers (concurrent interleave / half-write / crash window).

### S5 verified the S10 seam: sync-op-never-on-worker is SAFE-BY-REACHABILITY-ONLY, not by construction (confirmed kernel-side). Actor runs ensure_open→reopen→open_collection INLINE on a runtime worker (lib.rs:179); reopen reachable TODAY (kernel_ops_reopen_after_cooperative_release). Safe only because open_collection/close/import/media don't spin anki's runtime (block_on); runtime_singularity (adapter.rs:846) pins sync=41/ankiweb=45/ankihub=47 NOT dispatched. NO structural barrier — when #362 routes a sync service through an actor job it will block_on on a worker → panic (#503 hazard). **Architectural note for report.**
### S5 appendix: panicking actor job does NOT wedge (REFUTED — channel closes → clean "actor is gone" Err, fail-closed); Kernel::close can be raced into a reopen (low conf 0.4, benign); TagRefresher::request store-after-clear TOCTOU benign.

---

## S9 — Derived store, fusion & ranking  (reported, worktree reaped)
Surface health: TWO Medium defects. FTS5 injection REFUTED (quoting sound). substring-annotation gap REFUTED (post-pass fills it).

### S9-1 — rrf_fuse native diverges from search_fusion.py reference on a NaN weight (the parity contract)  [Medium] [TRIAGED]
- Lens: correctness (parity contract). Confidence 0.8. Repro RED.
- Location: fusion.rs:80-89 (sort_by b.1.total_cmp(&a.1)) vs search_fusion.py:89 (sort key -h.score). args.weights used verbatim, NO NaN/inf validation at actions.rs:1597.
- Class: two impls of one model drift; test-that-doesn't-bite. Both explicitly defend NaN weight (#382 "must not panic") so NaN is anticipated — and they must AGREE. Python Timsort over -h.score (NaN compares False) → input-order-dependent [3,2]; Rust total_cmp total-orders NaN → [2,3]. inf/-inf AGREE; only NaN diverges. Property suite (test_search_pipeline.py:39) samples weights {0.25,0.5,1.0,2.0} → can't catch.
- Predicted-correct: byte-identical order. Fix: validate/sanitize non-finite weights before rrf_fuse (preferred), or match reference NaN order to total_cmp.
- Repro: s9_nan_parity.py → ref=[3,2,1] native=[2,3,1]. Saved: .team-review/repros/s9_nan_parity.py.
- Cross-surface: weight-supply path + future --search-* knob = S3/host → weight validation belongs there. Goes live only when host/knob supplies weights (current caller always finite).

### S9-2 — metadata_changed fires full whole-collection tag-centroid recompute on every deck-rename/template-find-replace/field-metadata edit, no relevance probe, on a runtime worker  [Medium/perf] [TRIAGED — SUPERSEDES S6 appendix tag-centroid note]
- Lens: performance (violates 2 #445 rules). Confidence 0.85. Repro GREEN (proof).
- Location: lib.rs:1888-1898 (metadata_changed → unconditional tag_refresh.request()); tag_centroids.rs:309-326 (run_once: only guard embed.is_empty()); recompute reads read.rs:252 (full scan) + per-distinct-tagged-note vector fetch (tag_centroids.rs:361-377).
- Class: per-op tail O(collection) behind no relevance probe + synchronous recompute on a runtime worker (not spawn_blocking). Production reach: upsert_decks/delete_decks/find_replace_note_types/update_note_type_field_metadata (lib.rs:2176/2189/2269/2289) → metadata_changed → unconditional request. None of these edits inputs any centroid → pure waste.
- CORRECTS S6's flag: the relevance probe IS present on upsert/delete tails (lib.rs:1725 tagged||any_member_of(written); :1879 any_member_of(note_ids)). ABSENT ONLY on metadata_changed. So narrower than "every refresh." an any_tagged SQL probe exists (read.rs:277) but isn't used here.
- Predicted-correct: metadata-only bump O(1)-ish; refresh only when tag MEMBERSHIP could change. Fix: metadata_changed gets a membership_may_have_changed bool (or separate tag-only entry) so only tag ops request; run recompute via spawn_blocking.
- Repro: s9_metadata_changed_runs_full_recompute_with_no_relevance_probe (GREEN = proof; out-of-band tag mutation appears in centroid keys only if metadata_changed ran a fresh full scan+recompute). [in kernel test module; reconstruct at proposals]
- Cross-surface: fix in metadata_changed → **S6** owns lib.rs[1736-2455]. DEDUPE: S6 appendix tag-centroid note → folds into S9-2.

### S9 appendix: FTS5 injection REFUTED (fts_quote shrike-derived/lib.rs:1352 — threw column filters/NEAR/AND/OR/NOT/prefix/negation/quote-breakout, all held, no error oracle/DoS). substring field-source annotation REFUTED (filled by post-pass actions.rs:1327-1336; the comment at :1077-1082 is misleading but behavior correct). LOW: tag_ranking select_nth_unstable before sort_by → theoretical tie wobble (f32 dot ties vanishingly rare); recompute non-atomic tag-space window (self-healing, graceful).

---

## S8a — Collection read & embed-text  (reported, worktree reaped)
Surface health: strip-skip byte-identity predicate SOUND (4000-case fuzz + real anki stripper — DO NOT RE-CHASE). extract_image_refs/cloze/field_is_blank byte-identical. ONE Low divergence.

### S8a-1 — WS_RE=\s+ diverges from Python byte-identity contract on C0 separators U+001C–U+001F  [Low] [TRIAGED]
- Lens: general/correctness (byte-identity invariant). Confidence 1.0 (divergence) / 0.5 (worth fixing).
- Location: embed_text.rs:47 (WS_RE = Regex::new(r"\s+")) used at :107. Rust regex \s = Unicode White_Space (EXCLUDES U+001C–001F); Python re \s INCLUDES them. "a\u{1c}b" → oracle "a b", port "a\u{1c}b".
- Class: parser/consumer disagreement (two regex engines, different \s). NO field-path impact TODAY: anki invalid_char_for_field strips U+001C–001F from stored fields via normalize_field (rslib/src/notes/mod.rs:350) → embedding read path never sees them.
- Repro: s8a_finding1_c0_separators_break_byte_identity (real anki strip injected) + 4000-case fuzz (631/3541 divergences, 100% explained by U+001C/1D/1E). [needs reconstruction — anki strip injection]
- Predicted-correct: byte-identity with oracle — fold C0 separators explicitly, OR document the contract holds only over anki-sanitized field text.
- Cross-surface (LIVE PATH): anki's masking does NOT cover OCR/ASR derived text (bypasses normalize_field). If source='ocr' recognized text (arbitrary control bytes) routes through normalize_for_embedding → LIVE spec break. **SEED into S12 (recognition): does OCR/ASR text pass through normalize_for_embedding?** If yes, S8a-1 severity rises.
### S8a appendix: list_notes modified_since (read.rs:447-456) materializes ALL recent ids into a HashSet to intersect — push cutoff into scoped query / per-row filter. Low perf (read tool, not per-item loop). note_texts/note_embed_inputs repeated-id footgun (read.rs:123-167, dup id → silent "" for 2nd+; doc-comment contract not type-enforced) — leaky interface, low.

---

## S11a — Python embedding stack  (reported, worktree reaped)
Surface health: probe core / fingerprint namespacing / pooling-mmproj folding / _effective_batch all SOUND.
KEY: **S11b-1 is RUST-LANE-ONLY** — Python probe drift = float(np.max(np.abs(ref-batched))); np.max PROPAGATES
NaN, nan<=tol is False → NaN-under-batching declared VARIANT→serial (safe). Verified w/ 2 passing tests. So
S11b-1 has NO Python sibling. ONE Low-Med lifecycle defect.

### S11a-1 — Backend-alias not normalized on start(backend=) override → documented alias 400s + permanently poisons runtime  [Low-Med] [TRIAGED]
- Lens: general/correctness (lifecycle state corruption under lock). Confidence 0.9. Repro RED (2 failed).
- Location: embedding.py:738-739 (override self._backend_kind=backend, MISSING the BACKEND_ALIASES.get(...) that __init__ applies at :632); sink _make_backend :840-843. Source: POST /embedding/start body `backend` → server.py:401-415 → harness.start_embedding → runtime.start(**overrides).
- Class: divergent handling of same value in two places (ctor normalizes, override doesn't) + mutate-before-validate (kind overwritten BEFORE the try → a failed start leaves _backend_kind = un-normalized alias → subsequent NO-override start() ALSO raises). `{"backend":"onnx-rs"}` (documented alias) → ValueError "Unknown embedding backend" → HTTP 400, runtime bricked for daemon life.
- Predicted-correct: start(backend="onnx-rs") behaves like ctor — normalize to "onnx", fail later (if at all) on the real reason. One-line fix: apply alias map at the override site.
- Repro: s11a_alias.py (2 tests RED). Saved: .team-review/repros/s11a_alias.py.
- Tests-that-dont-bite: alias normalization tested only at BOOT/ctor (test_fused_pipeline.py:38-43 --embedding-backend onnx-rs flows through ctor); NO test exercises start(backend=) override with an alias. Boot path uses ctor → production boot unaffected; only live /embedding/start restart with an alias.
- Cross-surface: none (within S11a).

### S11a appendix (NOT findings)
- Probe unguarded against backend returning wrong vector COUNT under batching: OnnxBackend._finish_start (embedding_onnx.py:229) catches only ProbeError → a ragged-count batched result propagates uncaught ValueError out of start() (llama/clip/remote caught by broad except → serial). 0.4, needs buggy native engine.
- **Modality invariant modalities⊇{TEXT} NOT enforced** (profiles.py:240-255 parse_capabilities validates non-empty subset but doesn't require TEXT; plan_to_runtime_params_one :756-766 passes [image] through → runtime:remote builds RemoteBackend(modalities={image}) violating protocol embedding_base.py:64-65). Probe still embeds TEXT probe-texts vs image-only endpoint (ProbeError swallowed→silent serial). **Cross-surface → S13 (profiles) — PAIRS with the ≤1-image-space enforcement gap (S7 seam). Both are profiles.py validation gaps.** 0.85 gap / 0.4 harmful.
- native_embedder() 2-3 /v1/models round-trips per attach (not hot path).

---

## S14a — CLI & client  (reported, worktree reaped)
Surface health: ONE High security defect + one Low contract bug. CollectionBusyError sentinel-spoof + deck-id ambiguity investigated → NOT defects.

### S14a-1 — Rich-markup injection from untrusted note content → terminal spoofing + content-driven CLI crash (DoS)  [HIGH] [TRIAGED]
- Lens: security (source→sink) + general. Confidence 0.97. Repros RED (unit + full CliRunner e2e).
- Location: cli/output.py — note_detail (L190-221: field name L209, value L211, tags L202, note_type/deck L199-200), note_summary_row (L159-171), note_type_detail (L243/249-250/254-255), result_status (L276 r.error); also note_cmd.py:623 search snippet. Sink: console.print (markup=True default) / output.table. NO rich.markup.escape() anywhere in output.py (grep-confirmed).
- Class: output/markup injection (CWE-150) + unhandled-exception DoS. Note field/tag/deck/snippet — authored by anyone who can write the collection (Anki sync, shared/imported .apkg, MCP upsert_notes) → server returns verbatim → Note.content (free-form dict[str,str]) → output.* → console.print parses [..] as markup. (A) well-formed tags → ANSI restyle/conceal/hyperlink (terminal spoof); (B) malformed tag (e.g. "see [/cyan] here") → rich.errors.MarkupError UNCAUGHT (ShrikeGroup.invoke only catches ShrikeError) → whole command dies with traceback. EVERY display mode affected (detail, --brief table, search snippet). --json safe; --no-pretty avoids spoof (no_color) but STILL crashes (markup parsing still on).
- Predicted-correct: escape untrusted content (rich.markup.escape) or pass as Text/markup=False → displayed literally, can't restyle or crash.
- Repro: s14a_markup_injection.py (unit: interpret + crash + brief-table) + e2e CliRunner (note show 42 → exit 1, MarkupError, empty output). Saved: .team-review/repros/s14a_markup_injection.py.
- Cross-surface: same content flows to S2's media/export renders IF they use output.* → sweep S2's renders too.

### S14a-2 — --json --pretty mutual-exclusion is order-dependent  [Low] [TRIAGED]
- Lens: general (boundary/contract; guard fails to reject an illegal state). Confidence 0.9. Repro RED.
- Location: cli/output.py _merge_pretty (L78-83) + _merge_json (L71-75), eager Click callbacks fire in CLI order. _merge_pretty raises only when --pretty seen AFTER json set → `info --json --pretty` errors (good) but `info --pretty --json` silently sets json (no error). Predicted-correct: error regardless of order. Repro: s14a_markup_injection.py note / test_json_pretty_order.py.

### S14a appendix (NOT findings): CollectionBusyError sentinel-spoof investigated (client.py:265 startswith "collection_busy:") — no ToolInputError places attacker input at position 0 with that prefix; ~0.1 exploitable, low impact. Deck-id ambiguity (_match_deck bare int = ID then name) documented = intended. **client.download_export/read_media blind GET on server-supplied URL (client.py:643-652) — trusts server URL unconditionally; write-path/traversal angle → CROSS-SURFACE to S2 (export_cmd/media_cmd write the bytes+filename).** server_status swallows non-200 (minor).

---

## S13 — Config, profiles & host assembly  (reported, worktree reaped)
Surface health: **SEEDED IMAGE-SPACE CHECK = ENFORCED** (resolve_profile profiles.py:606-614 rejects 2 image-capable spaces w/ ProfileError "at most ONE image-embedding space" — verified w/ crafted config → S7's worry ADDRESSED, NOT a finding). purely-local→media-path-root composition SOUND. ONE Medium + 2 Low.

### S13-1 — Harness.reload() omits secondary cross-space floor recalibration that every other reindex path performs → stale image floor after /reload  [Medium] [TRIAGED]
- Lens: general/correctness (architecture: divergent sibling verbs). Confidence 0.85. Repro RED (control passes, reload fails).
- Location: harness.py:936-944 (reload) vs siblings _drive_reindex(440-443)/_rebuild_then_calibrate(597-601)/_drive_boot_reindex(892-895) which all call _recalibrate_secondary_floors after reindex_if_needed. POST /reload → reload() → reindex_if_needed True (reconciles secondary image vectors, lib.rs:1078-1153) → NO recalibration → secondary cross-space image floor (mean+margin·std, #580/#576) stays computed vs pre-reload vectors → floor-admission mis-gates until next boot/rebuild/cooperative-reacquire.
- Predicted-correct: after reload() whose reindex returns True, recalibrate (one await self._recalibrate_secondary_floors() after :943).
- Repro: s13_reload_recalibrate.py. Saved. Cross-surface: S7's floor mechanism; fix in harness.py.

### S13-2 — mmprojs admitted onto a managed llama-server whose consumer is text-only  [Low] [TRIAGED]
- Lens: general/correctness (bound on the wrong thing). Confidence 0.6. Repro PASSES (documents).
- Location: profiles.py:712-724 — check is `managed_llama.mmprojs and not any("image" in e.modalities for e in caps.embedders)` (tests "SOME space has image" not "the entry CONSUMING the managed llama has image"). Config: managed llama backs text-only no-endpoint primary, image lives on a separate remote endpoint → projectors load onto a server that never embeds images (silent no-op). Predicted-correct: ProfileError (managed server's consumer must declare image) or don't attach.

### S13-3 — same legacy config → opposite outcomes via two launch paths (degrade vs refuse-boot)  [Low] [TRIAGED]
- Lens: general/correctness (duplicated/divergent resolution models). Confidence 0.55. Repro PASSES.
- Location: cli/config.py:559 (resolve_embedding_profile short-circuits caps.legacy → resolve_embedding, no build validation → DEGRADE) vs server.py:976-979 (daemon --config path calls resolve_profile(caps, build_features()) unconditionally incl. migrated legacy → ProfileError REFUSE-BOOT). Narrow reachability (direct --config of legacy config on mismatched build; normal CLI never sets --config for legacy).

### S13 appendix (NOT findings): malformed scalar config → bare AttributeError (_merge only recurses dicts) — hardening. _deep_copy_defaults shallow-copies server lists (shared w/ module default) — latent, no live in-place mutation. **run_job-dropped-future GC concern: _fire_release (collection.py:341-344) drops the run_job awaitable; first poll IS driven (bridge correct for happy path) BUT once py_future dropped + Pending, only the cyclic py_future↔poll_cb ref keeps it alive → theoretical drop-the-release if cyclic GC collects mid-flight → CROSS-SURFACE to S10 (asyncio_bridge.rs); low conf, no GC-pressure repro.** _migrate_legacy embedder construction vestigial (dead code).

### UNRESOLVED SEED: S11a's modalities⊇{TEXT} enforcement gap was NOT explicitly verified by S13 (S11a finished after S13 dispatched). Carry to validation step 1 — an Opus peer checks profiles.py parse_capabilities for TEXT-requirement.

---

## S12 — Recognition & managed lifecycle  (reported, worktree reaped)
Surface health: **S8a-1-OCR verdict: NOT a live break** — recognized text embedded RAW, never through normalize_for_embedding/WS_RE → S8a-1 stays field-path-only/latent. reconcile==rebuild holds on OCR path. Gated-marker dedup / fingerprint re-derive / OCR-hash byte-identity / vector_worthy all CLEARED. ONE Medium + 1 Low-Med.

### S12-1 — orphan reap kills an unrelated recycled-PID process (dual signal necessary-but-not-sufficient)  [Medium] [TRIAGED]
- Lens: security/general (wrong-target privileged kill; TOCTOU). Confidence 0.9. Repro RED.
- Location: shrike-llama-server/src/lib.rs:358-377 (reap_orphan), claim :354-357 + CLAUDE.md. Guard pid_alive(recorded) && port_held(host,port) checks the two conditions INDEPENDENTLY — never "is recorded PID THE process bound to port?" → recycled PID + ANY unrelated loopback-port holder → terminate_pid(recorded) SIGTERM→SIGKILL the bystander. Compound cost: after wrong-kill, wait_port_bindable times out (real holder still there) → fresh spawn EADDRINUSE → embedding start errors (kills bystander AND leaves embedding down).
- Predicted-correct: reap recorded PID only if THAT PID genuinely holds our port; a recycled PID not holding the port survives even when some OTHER process holds it.
- Repro: s12_reap_wrongkill.rs (binds port H, records live sleep PID U, reap → U SIGTERMed; asserts U alive). Saved.
- Why undetected: existing alive_pid_without_the_port_is_not_reaped (:721-742) only covers "recycled PID + port FREE", never "recycled PID + UNRELATED port holder".
- Cross-surface: none. **Windows WORSE: pid_alive hardcoded true (:97-102) → guard collapses to port_held alone → ANY port-holder triggers a kill of the recorded (possibly recycled) PID.**

### S12-2 — recognition sweep re-enumerates whole collection + reloads full done-set per bounded batch → O(image-notes × backlog/batch) drain + per-probe clone  [Low-Med/perf] [TRIAGED, evidence-only]
- Lens: performance (cost scales w/ whole collection not work; #445 no-clones-in-hot-loop). Confidence 0.7.
- Location: lib.rs:1355-1361 (per-call full note_image_refs), :1476-1486 (per-call done reload + per-pair name.clone() in HashSet probe). Driven by harness.py:743-744 (while True: recognize_pending(8)). Draining N pending at batch=8 = ~N/8 calls, each O(image_notes + judged) + exists syscall/Python-call per pending pair per sweep → ~O(image_notes²/8) for initial backlog. name.clone() per probe (set keyed (i64,String)).
- Predicted-correct: don't pay O(collection) enumeration + full done-reload per batch — enumerate once across the drain / key done as HashMap<i64,HashSet<String>> (drop clone) / cursor past scanned notes.
- Repro: evidence-only (100k benchmark is real demonstrator; cost structural in call graph). Cross-surface: shrike-derived (refs_for_source) + shrike-collection (note_image_refs).

---

## S14b — Mobile / store-api / ffi  (reported, worktree reaped)
Surface health: FFI machinery SOUND (guard fire-once/no-unwind verified; no panic=abort → catch_unwind valid; all 7 C-ABI symbols exported; UserData unsafe impl Send sound; Arc-clone-before-spawn closes close/op use-after-free; ImportUpdateCondition cast matches anki proto; bad-input probes all return clean error envelopes not panics). ONE Medium + 1 Low-Med.

### S14b-1 — shrike_op/shrike_close after shrike_runtime_shutdown hang forever (silent hang, never an error envelope)  [Medium] [TRIAGED]
- Lens: general/correctness + concurrency (FFI robustness). Confidence 0.75. Repro RED (silent hang, 5s timeout).
- Location: shrike-mobile/src/lib.rs:309-320 (shrike_runtime_shutdown stops+joins the only driver thread), :456-481 (shrike_op), :393-424 (shrike_close); root cause undriven current_thread runtime (shrike-kernel/runtime.rs:103-114 spawn_op). Post-shutdown op → spawn onto undriven runtime → task queued never polled → callback NEVER fires; shrike_close blocks forever on rx.recv().
- Class: leaky-interface/unbounded-wait. Contradicts crate doc "a synchronous misuse … reported through the callback, never a panic" — a post-shutdown op is exactly such a misuse but silently hangs. Documented-contract misuse (doc says shutdown only at teardown after all ops complete) BUT realistic on a racy iOS suspend/teardown (#393 motivation).
- Predicted-correct: post-shutdown op fires a Unavailable/Internal error completion (guard on a shut-down flag), or shutdown is asserted terminal w/ ops fast-failing.
- Repro: scratch_postshutdown.rs test binary → "POST-shutdown op HUNG ... SILENT HANG confirmed". [reconstruct at proposals]
- Cross-surface: kernel runtime spawn_op (S5) — route the "spawn onto a possibly-undriven runtime" concern there; defense belongs in shrike-mobile.

### S14b-2 — VectorIndex::remove trait doc says "how many vectors left" but contract/all callers = "count removed"  [Low-Med] [TRIAGED, evidence-only]
- Lens: general/architecture (leaky/misuse-prone interface). Confidence 0.7.
- Location: shrike-store-api/src/lib.rs:75-77 (trait doc "returns how many vectors LEFT") vs canonical impl shrike-index/engine.rs:291-316 ("count removed from text sub-index", test :1461 asserts removed) + consumer index_set.rs:250-263 ("removal count"). The store-api crate EXISTS so 3rd-party implementers don't read the impl → "left" (natural reading = remaining) is opposite the real contract → a fresh impl returns remaining → kernel reports wrong removal count, no type/test in store-api catches it.
- Predicted-correct: doc says "returns the number of vectors REMOVED (text-modality note count)".
- Cross-surface: none (trait doc is the fix site).

### S14b appendix (NOT findings): VectorIndex::add keys/vectors length-pairing unenforced at trait level (parallel-arrays misuse-prone; canonical impl checks engine.rs:262). Collection release/ensure_open/reopen temporal coupling documented-unenforced (collection.rs:224-232). Error-taxonomy cause flattening (no source(), folds into message) — BY DESIGN, acceptable. bad-cache-dir classified `internal` for caller path (kernel lib.rs:419-420) → arguably invalid_input, minor, kernel surface. top_k unbounded passthrough — trusted in-process caller, no finding. async callback on Rust thread (no JNI attach) — contract sharp edge. shrike_close re-entered from callback → deadlock current_thread — caller misuse.

---

## S12 appendix (NOT findings): NaN/Inf confidence bypasses gate floor on PyRecognizer path (recognize.rs:152 confidence<min; NaN→false→passes; Apple safe via serde; PyRecognizer py_recognizer.rs:25-37 extracts f64 unvalidated). Operator-trusted not external; is_finite() check would close. 0.6. host settable on EmbeddingRuntime (pin at call site server.py:1013 not in manager; passthrough strips --host). No reachable non-loopback today; defense-in-depth (reject non-loopback in LlamaServerManager::new). 0.8 safe. Reap can miss a still-starting orphan (port not yet bound → skip → race). 0.5. CLEARED: gated-marker dedup, fingerprint re-derive convergence, OCR hash byte-identity (Cow::Borrowed empty), vector_worthy consistency, sweep bounded (truncate), reserved-flag stripping, SwiftString allocator.
</content>
