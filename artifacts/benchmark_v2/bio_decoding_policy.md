# BIO decoding / evaluation policy — benchmark_v2 (R0/R1/R2)

Generated while investigating the smoke-test finding "148 BIO violations after
2 optimizer steps". This document defines exactly what "BIO violation" means
in this codebase, what decode policy is used, and what changed to close the
gap the smoke test exposed.

## A. Definition used (before this round's fix)

Source: `src/bioes_utils.py:find_bio_violations()`, called from
`src/evaluator_extended.py` on tag sequences already filtered for special
tokens (see below).

| question | answer |
|---|---|
| Which transitions are invalid? | `I-X` immediately preceded by anything other than `B-X`/`I-X` of the **same type** `X`. This single rule captures `O -> I-X`, `B-X -> I-Y` (X≠Y), `I-X -> I-Y` (X≠Y), and `START -> I-X` (start = `prev_type=None`), because all of them reduce to "an `I-X` with no matching open entity of type X immediately before it". |
| Start constraint checked? | Yes, implicitly: `prev_type` starts as `None`; a sequence starting with `I-X` is flagged with `prev_type=None` ≠ `X`. |
| End constraint checked? | No, and this is correct for plain BIO (no closing marker exists — `bio_legal_end()` in `bioes_utils.py` always returns `True`). A sequence ending on `B-X` is **not** a violation. |
| `O -> I-X` counted? | Yes (see above). |
| `B-X -> I-Y`, X≠Y counted? | Yes (see above). |
| Sequence ending on `B-X` invalid? | No — not a violation under BIO. |
| `[CLS]`/`[SEP]`/pad excluded before counting? | Yes. `evaluator_extended._run_inference()` (and `Evaluator.evaluate()` independently, same logic) filters `if mask==0 or gold_label==-100: continue` **before** building the tag list that gets checked/scored. `find_bio_violations()` never sees a CLS/SEP/pad/subword-continuation position. |
| `-100` labels fed to the violation checker? | No — same filter as above excludes them. |
| Violation count unit | **Number of invalid tag occurrences** (`len(violations)`), i.e. one count per illegal `I-X` position, summed per sentence into `total_violations`. Not "number of invalid sequences" (a sentence with 3 illegal `I-` tags counts as 3, not 1) and not "number of repairs performed" as a separate concept — under the repair policy below, occurrence count and repair count happen to coincide (1 repair per violation). |
| Repair policy | `src/audit_utils.py:bio_to_entities()` — an orphan `I-X` is **reinterpreted as if it were `B-X`** (opens a new entity there), flagged `violation=True`. It never raises, never drops a token, never rewrites the tag string. |
| Repair timing vs. strict entity-level eval | **Before this fix**, repair was applied *only* inside the enrichment path (`evaluator_extended.py`'s entity extraction for seen/unseen, GR-11 examples, span categories) — it never touched the primary `test_f1`. |
| What did seqeval score? | **Raw, unconstrained, un-repaired decoded tags** (filtered for special tokens only). `Evaluator.evaluate()` took `model(...)` output directly — the model's `torchcrf.CRF.decode()` call has no hard transition constraints for the `guwenbert_crf` architecture (unlike the M2/M3 boundary-heads variant, which already hard-constrains via `apply_hard_transition_constraints`). |

## B. Root cause of the 148 violations (smoke test)

Confirmed via `test_transition_constraints.py`'s own established pattern
(adversarial emissions) applied to this case: `torchcrf.CRF` for
`GuwenBertCRF` is initialized with **unconstrained, freely-learned**
`transitions`/`start_transitions`/`end_transitions` — nothing in
`src/models.py:GuwenBertCRF` masks illegal entries to `-inf`. After only 2
optimizer steps (smoke test), the learned transition scores have not yet
suppressed illegal paths, so Viterbi decode legitimately picks some. This is
**category 1: raw CRF illegal transition** — not an evaluator bug, not a
mask artifact, not a gold-label issue. Classification of the 148 (see
`artifacts/benchmark_v2/bio_violation_examples.json` for the dumped
examples, generated in this round): all fall under category 1 (raw CRF
illegal transition, expected instability from a 2-step "trained" model); 0
were caused by special-token/padding artifacts (excluded correctly before
counting, see above); 0 were gold-label invalidity (validated separately,
see section C below).

A secondary, unrelated finding surfaced during this same investigation:
`SIKU-BERT/sikubert`'s tokenizer silently drops PUA (Private Use Area)
characters common in this corpus (documented separately in
`src/evaluator_extended.py`'s `subword_coverage_gap_note` field) — this
does **not** contribute to BIO violations (it's a token-count mismatch
issue, already fixed, unrelated to CRF decode legality).

## C. Fix implemented: decode-time constrained Viterbi (opt-in)

New module `src/constrained_decode.py`:
- Reuses **existing, already-tested** machinery
  (`src/bioes_utils.py:build_transition_masks(scheme="bio")` +
  `apply_hard_transition_constraints`) — the same mechanism already
  production-proven for the M2/M3 boundary-heads architecture
  (`tests/test_transition_constraints.py`).
- `build_constrained_crf(crf, label2id, scheme="bio")` returns a **deep
  copy** of the model's CRF with illegal transitions set to `-inf`. The
  original `model.crf` (and therefore training loss / `optimizer.step()`)
  is **never touched**.
- `constrained_decode_batch(...)` replays the pre-CRF forward pass
  (backbone → dropout → linear) to get emissions, then decodes with the
  constrained copy via `torchcrf.CRF.decode()` (the real package, same
  transition orientation as everywhere else in this codebase —
  `transitions[i, j]` = score of going **from** tag `i` **to** tag `j`,
  confirmed by `test_orientation_transitions_i_j_is_from_i_to_j`).

This is a **decode-time-only** constraint (training loss/partition function
unaffected), matching the requirement "do not silently change training loss
unless proven the CRF implementation supports constrained partition
training correctly." Applying it during training too (as M2/M3 do) would be
a defensible future enhancement but is out of scope for this benchmark's
current, minimal-blast-radius change.

## D. Wiring (opt-in, zero effect on any other experiment)

- `src/evaluator.py`: `Evaluator.__init__` gained
  `use_constrained_decode: bool = False, label2id: dict = None,
  constrained_scheme: str = "bio"`. **Default False** — every existing
  caller (M0/M1/M2/DAPT/other experiments) that doesn't pass these new
  kwargs gets byte-for-byte the same behavior as before. All three decode
  call sites (`evaluate`, `error_analysis`, `confusion_matrix`) route
  through one `self._decode(...)` method so the whole `Evaluator` (dev-F1
  checkpoint selection, headline test F1, confusion matrix, error analysis)
  is internally consistent.
- `configs/benchmark_v2/{R0_v1,R1_v2_full,R2_v2_clean}.yaml`: added
  identical `evaluation: {constrained_decode: true, constrained_scheme:
  bio}` block to all three (verified via diff — only `data.*`/`output_dir`/
  `project.name` differ between them).
- `scripts/train.py`: reads `cfg.evaluation.constrained_decode` (default
  `False` via `OmegaConf.select`), passes it into `Evaluator`, and passes
  the same `constrained_crf` into `extended_test_report`, so **checkpoint
  selection, headline test F1, and the enrichment report all use the same
  decode policy** — no inconsistency between "model selected under policy
  A, reported under policy B."
- `src/evaluator_extended.py`: now decodes **both** raw and constrained
  tags per sentence (one backbone forward, two CRF decodes) and reports
  `bio_violations.raw_total_violations` vs
  `bio_violations.constrained_total_violations` side by side. Entity
  extraction (tp/fp/fn, span categories, ORG↔TITLE confusion, seen/unseen,
  GR-11 examples) uses **constrained** tags when a `constrained_crf` is
  supplied — no repair heuristic needed in that path, since constrained
  decode guarantees 0 violations by construction (proven by unit tests,
  not just claimed).

## E. Production benchmark F1 (what R0/R1/R2 will actually use)

With `evaluation.constrained_decode: true` (set in all three configs):
- **Dev F1 used for checkpoint selection**: constrained decode tags.
- **Headline test F1 / per-label P/R/F1**: constrained decode tags (same
  `Evaluator.evaluate()` call, reused by `full_report()`).
- **Confusion matrix / error analysis plots**: constrained decode tags.
- **Enrichment report** (seen/unseen, span categories, GR-11 examples,
  ORG↔TITLE): constrained decode tags.
- **Repair-by-reinterpretation is not on the primary path** at all in this
  configuration — it remains available only as a fallback for the
  `constrained_crf=None` case (kept for backward compatibility with any
  future non-benchmark caller of `evaluator_extended.py`).
- Diagnostic-only: raw (unconstrained) violation counts are still recorded
  every run for comparison/audit (`bio_violations.raw_total_violations`),
  never used to select or score anything.

## F. Unit test coverage (`tests/test_constrained_decode.py`, 7 tests)

All passing:
1. `test_orientation_transitions_i_j_is_from_i_to_j` — confirms matrix
   orientation against the real `torchcrf` package.
2. `test_illegal_O_to_I_cannot_be_output_when_constrained` — adversarial
   emissions prove raw decode *would* pick `O -> I-X`; constrained decode
   never does.
3. `test_start_with_I_cannot_be_output_when_constrained`.
4. `test_mismatched_B_X_to_I_Y_cannot_be_output_when_constrained`.
5. `test_legal_B_X_to_I_X_path_remains_outputtable` — constraint doesn't
   over-block legal continuations.
6. `test_special_and_pad_token_positions_excluded_before_violation_check`.
7. `test_constrained_decode_zero_violations_on_random_adversarial_batch` —
   5-seed fuzz test, 0 violations every time.

Plus `tests/test_evaluator_extended.py::test_extended_report_uses_constrained_tags_when_provided`
proving the report layer actually switches to constrained tags when
available, and reports 0 constrained violations.
