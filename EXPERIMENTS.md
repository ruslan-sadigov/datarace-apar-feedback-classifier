# Experiments: what worked, what didn't

The final solution is the product of many experiments. This page lists the notable ones with
their outcomes. Every number comes from the per-model documents in [`docs/`](docs/); the
notebooks and scripts for the rejected ideas are not part of this repository.

## Decision discipline

- **Out-of-fold (OOF) evaluation** with identical `StratifiedKFold(5, shuffle=True,
  random_state=42)` folds for every model, so predictions can be blended without leakage.
- **Noise floor of ~0.003–0.006 Macro F1.** Identical configs re-run on different GPU sessions
  varied by about that much, so differences inside it were treated as "worth checking", not
  "confirmed".
- **Pre-registered adoption gates.** A change had to beat its baseline by more than the noise
  floor, measured in a paired comparison (same folds, seed and hardware).
- **Leaderboard as tie-breaker.** The primary submission was promoted on its public score, not
  on an OOF delta that sat inside the noise floor.

## What was adopted

| Idea | Result | Decision |
|---|---|---|
| Task-adaptive pretraining (TAPT): continue XLM-R-large's MLM objective on the training texts (labels unused) before fine-tuning | Mean CV 0.9272 vs 0.9207 for plain XLM-R-large; better on all 5 folds (+0.005 to +0.008) | **Adopted** |
| Fine-tuning learning rate 2e-5 instead of the cautious 1e-5 | Solo mean CV 0.9284 vs 0.9272 for lr=1e-5; blend OOF 0.9298 (W=0.35). Gain was inside the noise floor, so it was checked on the leaderboard: public 0.93446 vs 0.93120 | **Adopted** (primary submission) |
| Soft-voting blend with a TF-IDF + LinearSVC model | Blend CV 0.9284 vs 0.9272 for the TAPT model alone (lr=1e-5); TAPT and the SVM disagree on 8.4% of rows | **Adopted** |

## What was tried and rejected

| Idea | Result | Why it was dropped |
|---|---|---|
| Spelling correction of the Azerbaijani text (LLM-reviewed word list) | Real, leaderboard-confirmed win for the SVM alone (+0.00418). Flat to negative at the blend level; a TAPT retrain on corrected text gave blend OOF +0.0012 vs the anchor | Below the 0.006 gate; not adopted |
| LLM-generated synthetic data for `customer_support` | Fold-honest check (synthetic rows in training folds only, real rows for validation) showed no transferable signal at the linear-model level, with a small consistent negative lean on the targeted class | No gain |
| Temperature scaling of the blend | Nested CV 0.9294 / 0.9293 vs 0.9284 plain. Widening the temperature grid did not raise the held-out score | Grid-search overfitting on a flat surface; the ~+0.001 effect is below the noise floor |
| Finance/refund keyword override (23-term lexicon, 90.2% standalone precision toward `customer_support`) | Mean delta −0.0036, 0 of 5 folds improved | The models already learn this correlation; the override mostly fires on exception rows |
| Layer-wise learning-rate decay, weight souping, distillation | Tested as XLM-R-large training-recipe variants | None cleared its adoption gate |
| Stacking instead of a fixed blend weight | Tested | Did not beat the fixed-weight blend |
| MLM epoch sweep {2, 5, 10} | No candidate beat 10 epochs by more than the noise floor | Kept 10 |
| 3-way blend of smaller models (SVM + XLM-R-base + mDeBERTa) | CV 0.9254, public 0.92012 | Scored below the large-model blends (0.93120 and 0.93446 public) |
| Second transformer in the blend | Not possible: any pairing with XLM-R-large exceeds 838M parameters | 600M cap |
| SVM variants: word trigrams, cross-lingual keyword collapsing, decision-threshold offsets | Tested while building the linear model | No improvement |

## The remaining error

The remaining errors are mostly at the `technical_support` ↔ `customer_support` boundary. The
organizer's labeling rule is that mixed reviews (a technical complaint plus an actionable
finance, refund or payment issue) are always `customer_support`. The models have already learned
this rule. On the 72 genuinely mixed rows, baseline Macro F1 is 0.9117 (97.2% accuracy), and the
only two remaining errors come from ambiguous word senses. Per-class OOF F1 for the secondary
blend: `customer_support` 0.8946, `other` 0.9278, `technical_support` 0.9627.
