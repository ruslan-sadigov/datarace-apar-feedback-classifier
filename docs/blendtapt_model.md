# 2-way Blend: TAPT XLM-R-large (lr=1e-5) + Linear SVM (soft voting)

Soft-voting blend of the TAPT-adapted XLM-R-large fine-tuned at lr=1e-5
(`xlmrlarge_tapt_model.md`) with the linear SVM (`linear_model.md`) — the only ensemble the
large model's size permits under the 600M parameter cap.

- **CV Macro F1 (out-of-fold):** 0.9284 (vs 0.9272 for the TAPT-large model alone; historical
  comparisons from earlier in the project, not part of this bundle: 0.9238 for the same pairing
  with a plain non-TAPT large model, 0.9254 for a 3-way blend of smaller models with temperature
  scaling)
- **Public leaderboard score:** 0.93120 (`submissions/submission_blendtapt_1.csv`)
- **Status:** one of this project's **two locked final submissions**. The other,
  `blendtapt_lr2e5_model.md` (public 0.93446), is the higher-scoring one — see that document for
  the primary recommendation. This one is kept as the second locked choice.

## How it works

```
blend = W_TAPT * P_tapt + (1 - W_TAPT) * P_svm,   W_TAPT = 0.4
prediction = argmax(blend)
```

- `P_tapt`: softmax probabilities from the TAPT-adapted XLM-R-large.
- `P_svm`: LinearSVC `decision_function` scores converted to probabilities via softmax.
- Each model receives its own training-matched preprocessing (lowercased text for the SVM;
  case-preserved `"tag | feedback"` for the transformer).
- As with all blends in this project, the below-half nominal weight on the (much stronger)
  transformer is a probability-scale correction — softmaxed SVM scores are far softer than the
  transformer's sharp probabilities — not a statement of model trust.
- No temperature scaling: see below.

## Why it helps, and the temperature scaling question

Measured in `notebooks/blend_tapt_tuning.ipynb` on out-of-fold predictions (identical
`StratifiedKFold(5, shuffle=True, random_state=42)` folds across all models — leakage-free):

- TAPT-large and the SVM disagree on 8.4% of rows; TAPT-large is right in 66.4% of those
  disagreements vs the SVM's 27.9% (a plain, non-TAPT large model in the same pairing, tested
  earlier in the project and not part of this bundle, had the SVM winning 32.0% of
  disagreements — TAPT-large wins more of the arguments it used to lose, on top of already
  being the stronger solo model).
- The weight sweep peaks at w=0.4 (Macro F1 0.9284) on a healthy plateau (w=0.35-0.45 all
  >= 0.9282).

**Temperature scaling was re-evaluated for this pairing** (previously rejected for the plain
large+SVM blend — with a single transformer, the blend weight already absorbs the calibration
correction — but the TAPT model's calibration could plausibly differ). Nested fold-wise
protocol (temperature and weight tuned on 4 folds, scored on the held-out 5th):

| Grid extent | Nested +T mean | Selected T (most folds) | Selected w (most folds) |
|---|---|---|---|
| T in [1, 2, 3, 5] | 0.9294 | 5.0 | 0.50-0.55 |
| T in [1, 2, 3, 5, 8, 12, 20] | 0.9293 | 20.0 | 0.80-0.85 |

Widening the grid moved the selected T from 5 to 20 and the selected w from ~0.5 to ~0.85 —
but the **held-out mean did not move** (0.9294 -> 0.9293, i.e. flat/marginally worse). If
higher temperature were a real effect, the held-out mean would keep rising as the grid widened;
it plateaued instead. This is the classic signature of grid-search overfitting on a flat
surface: with more (T, w) combinations available, the inner 4-fold search finds a point that
fits those folds slightly better by chance, without any real generalization gain. The genuine
effect (~+0.001, present in 4/5 folds) already saturates by T~=5, and +0.001 is below the
project's established ~±0.008 CV-to-public noise band — it would not reliably show up anywhere
that matters. The nested procedure also never hands over one deployable (T, w) pair (it
selects independently per fold); the plain sweep does. **Decision: deploy the plain blend,
w=0.4, no temperature scaling.**

Per-class OOF F1 at the deployed configuration (w=0.4, no temperature): customer_support
0.8946, other 0.9278, technical_support 0.9627, macro avg 0.9284 — the project's best result on
every class.

## Relative to the other locked submission

`blendtapt_lr2e5_model.md` differs from this one only in the TAPT model's fine-tuning learning
rate (2e-5 vs this one's 1e-5) and blend weight (0.35 vs this one's 0.4) — same linear SVM
component, same architecture, same recipe otherwise. It scores higher (public 0.93446 vs
0.93120) and is the primary recommendation; this document is kept because it's the second of
the two locked final submissions.

## Relevant files (all paths relative to this bundle's root)

```
src/predict_blend_tapt.py                inference script (CPU-only): runs both models on
                                         test.csv, blends probabilities, writes + validates
                                         the submission
notebooks/blend_tapt_tuning.ipynb        weight tuning + disagreement + temperature analysis
models/linear_model.joblib               fitted SVM pipeline (linear_model.md)
models/xlmrlarge_tapt_model/             TAPT-adapted XLM-R-large weights (xlmrlarge_tapt_model.md)
artifacts/oof_predictions/linear_oof_predictions.csv   SVM OOF predictions + decision scores
artifacts/oof_predictions/xlmrlarge_tapt_oof_predictions.csv   TAPT XLM-R-large OOF probabilities
submissions/submission_blendtapt_1.csv   generated submission (public 0.93120)
```

## Competition constraint compliance

- **Parameter count**: TAPT XLM-R-large = 559,893,507 (TAPT continues training, doesn't change
  architecture, so the count is identical to the plain model) + LinearSVC sparse weights ~1-2M
  -> **~562M total, under the 600M cap** (interpreted as total per submission, including
  ensembles). No other transformer partner is possible at this size.
- **CPU-only inference**: both components run on CPU.
- **Memory**: models loaded and run sequentially; peak ~2.5 GB (fp32 large-model weights +
  activations), well under the 8 GB limit.
- Ensembles are explicitly permitted by the competition rules.

## Environment and reproduction

No new dependencies beyond the component models' environments (`requirements.txt` +
`requirements-xlmr.txt` with CPU-only torch). To reproduce the submission with both trained
models present locally:

```powershell
python src/predict_blend_tapt.py
```

To reproduce the weight tuning: regenerate the OOF artifacts (`python src/train_linear.py`
locally; `notebooks/xlmrlarge_tapt_colab.ipynb` on Colab for the transformer), then run
`notebooks/blend_tapt_tuning.ipynb`.

## Estimated runtime

- `predict_blend_tapt.py` (4,797 test rows, CPU-only): **~13.1 minutes** (dominated by the
  560M transformer's forward passes; the SVM contributes seconds).

## Reproducibility

- Deterministic given the two trained models: fixed weight (0.4), no temperature scaling,
  argmax over averaged probabilities.
- Both component models trained with fixed seed 42 (see their respective docs).

## Known limitations / project status

- Residual error structure: the technical_support ↔ customer_support intent boundary, with a
  (shrinking) label-ambiguity floor — the project's persistent bottleneck across every model
  tried.
- Temperature scaling was investigated and deliberately rejected for this pairing (see above) —
  closed by the nested fold-wise evidence, not an open question.
- **The technical/customer boundary's underlying business rule was disclosed by the competition
  organizer and investigated (in the project's fuller analysis, not part of this bundle) —
  confirmed already learned by the models, not a modeling gap.** Apar's internal labeling rule:
  mixed reviews (both a technical complaint and a finance/refund/payment issue) are always
  labeled `customer_support` (priority `customer_support > technical_support > other`) — note
  this only covers *actionable* finance issues (refunds, billing errors, payment failures); a
  complaint that the service is simply too expensive, with nothing to resolve, stays `other`. A
  finance/refund keyword lexicon (23 terms, 90.2% standalone precision toward `customer_support`)
  was tested as a directional post-processing override on this blend's OOF predictions —
  **rejected**: net negative on every fold (mean delta −0.0036, 0/5 folds improved), because the
  models already learn the correlation from training exposure to these same words, so the
  override predominantly mis-fires on the rare *exception* rows (money mentioned but not the
  true issue) rather than recovering genuine misses. Isolating the 72 rows that are genuinely
  "mixed" (both technical and finance signal present) showed baseline Macro F1 already at
  **0.9117 (97.2% accuracy)** there, with the only 2 remaining errors traceable to polysemous
  non-financial senses of two ambiguous words rather than failure to apply the priority rule.
  Closed: no further gain available from this angle.
- This is a closed, final project snapshot; the recipe/weight here won't change further.
