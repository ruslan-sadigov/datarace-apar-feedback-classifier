# 2-way Blend: TAPT XLM-R-large (lr=2e-5) + Linear SVM (soft voting) — PRIMARY

Soft-voting blend of the TAPT-adapted XLM-R-large fine-tuned at lr=2e-5
(`xlmrlarge_tapt_lr2e5_model.md`) with the linear SVM (`linear_model.md`) — the only ensemble
the large model's size permits under the 600M parameter cap. Differs from
`blendtapt_model.md` (the other locked submission) only in the TAPT model's fine-tuning
learning rate and the blend weight; everything else — architecture, preprocessing, the linear
SVM component — is identical.

- **CV Macro F1 (out-of-fold):** 0.9298 at `W_TAPT=0.35` (vs 0.9284 for the TAPT-large-lr2e5
  model alone)
- **Public leaderboard score:** **0.93446** — this project's best-scoring submission
- **Status:** this project's **primary locked final submission**.

## How it works

```
blend = W_TAPT * P_tapt + (1 - W_TAPT) * P_svm,   W_TAPT = 0.35
prediction = argmax(blend)
```

- `P_tapt`: softmax probabilities from the TAPT-adapted, lr=2e-5-fine-tuned XLM-R-large.
- `P_svm`: LinearSVC `decision_function` scores converted to probabilities via softmax.
- Each model receives its own training-matched preprocessing (lowercased text for the SVM;
  case-preserved `"tag | feedback"` for the transformer).
- As with the other blend in this bundle, the below-half nominal weight on the (much stronger)
  transformer is a probability-scale correction — softmaxed SVM scores are far softer than the
  transformer's sharp probabilities — not a statement of model trust.

## Why `W_TAPT=0.35` here, not `0.4` (the other blend's weight)

A nested per-fold weight retune picked `0.35` in 4/5 folds for this specific TAPT variant, and
it also wins the apples-to-apples fixed-weight full-OOF comparison (0.9298 vs 0.9294 at `0.4`).
Both checks agree, so `0.35` is used here rather than reusing the lr=1e-5 model's tuned weight
of `0.4` from `blendtapt_model.md`.

## Why this variant was worth testing on the leaderboard

The lr=2e-5 fine-tune beat the lr=1e-5 anchor solo in its own sweep (mean CV 0.9284 vs 0.9260,
4/5 folds — see `xlmrlarge_tapt_lr2e5_model.md`). At the blend level, though, that gain shrank
to +0.0010–0.0014 (both weight candidates), inside this project's ~0.006 documented CV noise
floor — not a confirmed improvement by the same discipline this project otherwise holds
submissions to. It was still judged worth one leaderboard check because the direction was
consistent across three independent measurements (solo CV, solo full-OOF, blend full-OOF) and
the nested per-fold variance was lower (0.0038 vs 0.0061) than the lr=1e-5 model's. The actual
result confirmed a real gain: public 0.93446 vs the other blend's 0.93120 — a +0.00326
leaderboard improvement, larger than the OOF-predicted delta but in the same direction,
consistent with ordinary public-sample variance on top of a genuine (if modest) underlying
effect.

**This became the primary submission on the strength of that leaderboard result, not the OOF
number alone** — this project's working discipline treats CV/OOF deltas inside the noise floor
as "worth checking," not "confirmed," and reserves conclusions for the actual submission score.

## What was tried afterward and did not improve on this blend

For context, everything else attempted in this project after this became the primary submission
was measured against it and did not beat it — all closed, none part of this bundle:
- Spelling-correction preprocessing on the linear SVM branch: a real, LB-confirmed win *solo*
  (+0.00418), but flat-to-negative at the blend level in every variant tested, including a
  from-scratch TAPT retrain on spelling-corrected data (blend-OOF gate: +0.0012 vs anchor,
  below the 0.006 noise floor — not adopted).
- LLM-generated synthetic training data targeting the customer_support minority class: a
  fold-honest CV check (synthetic rows in training folds only, scored against real,
  unmodified validation rows) showed no transferable signal at the linear-model level, with a
  small, consistent negative lean on the targeted class — not pursued further.
- Several other transformer training-recipe variants (layer-wise learning-rate decay, weight
  souping, distillation, stacking instead of a fixed blend weight, MLM epoch-count sweeps) — all
  tested, none cleared their respective adoption gates.

## Relevant files (all paths relative to this bundle's root)

```
src/predict_blend_tapt_lr2e5.py           inference script (CPU-only): runs both models on
                                          test.csv, blends probabilities, writes + validates
                                          the submission
notebooks/xlmrlarge_tapt_lr_sweep_colab.ipynb   produces the TAPT-lr2e5 checkpoint
notebooks/linear_models.ipynb             produces the linear SVM component
models/linear_model.joblib                fitted SVM pipeline (linear_model.md)
models/xlmrlarge_tapt_lr2e5_model/        TAPT-adapted XLM-R-large weights (xlmrlarge_tapt_lr2e5_model.md)
artifacts/oof_predictions/linear_oof_predictions.csv   SVM OOF predictions + decision scores
artifacts/oof_predictions/xlmrlarge_tapt_lr2e-05_oof_predictions.csv   TAPT XLM-R-large OOF probabilities
submissions/submission_blendtapt_lr2e5_1.csv   generated submission (public 0.93446)
```

## Competition constraint compliance

- **Parameter count**: TAPT XLM-R-large = 559,893,507 (TAPT continues training, doesn't change
  architecture) + LinearSVC sparse weights ~1-2M -> **~562M total, under the 600M cap**
  (interpreted as total per submission, including ensembles). No other transformer partner is
  possible at this size.
- **CPU-only inference**: both components run on CPU.
- **Memory**: models loaded and run sequentially; peak ~2.5 GB (fp32 large-model weights +
  activations), well under the 8 GB limit.
- Ensembles are explicitly permitted by the competition rules.
- `test.csv` is used only for inference (predicting on it), never for training either component.

## Environment and reproduction

No new dependencies beyond the component models' environments (`requirements.txt` +
`requirements-xlmr.txt` with CPU-only torch). To reproduce the submission with both trained
models present locally, from this bundle's root:

```powershell
python src/predict_blend_tapt_lr2e5.py
```

To reproduce from scratch: train the linear SVM (`python src/train_linear.py`), train the
TAPT-lr2e5 checkpoint (`notebooks/xlmrlarge_tapt_lr_sweep_colab.ipynb` on Colab — requires the
`mlm_epoch_10` checkpoint described in `xlmrlarge_tapt_lr2e5_model.md`), then run the command
above.

## Estimated runtime

- `predict_blend_tapt_lr2e5.py` (4,797 test rows, CPU-only): **~13 minutes** (dominated by the
  560M transformer's forward passes; the SVM contributes seconds).

## Reproducibility

- Deterministic given the two trained models: fixed weight (0.35), argmax over averaged
  probabilities.
- Both component models trained with fixed seed 42 (see their respective docs).
- Run-to-run GPU non-determinism of ~±0.003-0.006 Macro F1 was measured during training
  (identical config/fold/seed across sessions); the competition rules explicitly accept such
  hardware-level variance.

## Known limitations / project status

- Residual error structure: the technical_support ↔ customer_support intent boundary, with a
  (shrinking) label-ambiguity floor — the project's persistent bottleneck across every model
  tried, and confirmed (see `blendtapt_model.md`) to reflect an already-learned business rule
  rather than a recoverable modeling gap.
- This is a closed, final project snapshot — the project's two submission slots are locked to
  this document and `blendtapt_model.md`. No further changes are planned.
