# XLM-RoBERTa-large with task-adaptive pretraining (TAPT), fine-tuned at lr=1e-5

Extends `xlm-roberta-large` (559,893,507 params, MIT license, FacebookAI, via Hugging Face) with
a task-adaptive pretraining phase: continue the base model's masked-language-model objective on
the competition's own training texts before any label is seen, then run a standard 5-fold
classification fine-tune from the adapted checkpoint instead of the generic pretrained one.

This is the **first** of two TAPT fine-tuning variants in this bundle — fine-tuned at the
originally-chosen learning rate `1e-5` (picked a priori as a cautious, smaller-than-default
value for large-transformer stability). A later learning-rate sweep found `2e-5` to be a real
improvement; that variant is documented separately in `xlmrlarge_tapt_lr2e5_model.md`, and its
blend (`blendtapt_lr2e5_model.md`) is this project's actual best-scoring locked submission. This
document is kept because `blendtapt_model.md` — the other locked submission — still uses this
lr=1e-5 checkpoint.

- **Mean 5-fold CV Macro F1:** 0.9272 (+/- 0.0060) — beat a plain (non-TAPT) `xlm-roberta-large`
  fine-tune (0.9207 mean CV, same recipe otherwise) on **all 5 folds** (+0.005 to +0.008); that
  plain-model comparison run is historical and not part of this bundle
- **Public leaderboard score:** not submitted solo — used as a member of the 2-way blend
  documented in `blendtapt_model.md` (public 0.93120)
- **Adoption gate (pre-registered):** replace the plain (non-TAPT) model only if mean CV >= 0.9240
  (a gain clearly above the ~±0.003 noise floor) — passed with room to spare (0.9272)

## Comparison against a plain (non-TAPT) XLM-R-large fine-tune

Historical reference only (the plain-model run isn't part of this bundle) — shows the TAPT
effect was consistent, not a lucky split:

| Fold | Plain XLM-R-large | TAPT XLM-R-large (this model) |
|---|---|---|
| 1 | 0.9272 | **0.9340** |
| 2 | 0.9189 | **0.9241** |
| 3 | 0.9252 | **0.9309** |
| 4 | 0.9235 | **0.9299** |
| 5 | 0.9088 | **0.9172** |
| **Mean** | **0.9207** | **0.9272** |

Per-class OOF F1: technical_support 0.9620, customer_support 0.8920, other 0.9276 — a gain over
the plain model on every class, the largest on customer_support (the project's persistent
macro-score bottleneck).

## Approach summary

Training notebook: `notebooks/xlmrlarge_tapt_colab.ipynb` (Google Colab, **L4 GPU** runtime).
Two phases, both using the shared preprocessing convention for every transformer trained in this
project (input `"tag | feedback"`, no lowercasing, `MAX_LENGTH = 96`):

1. **Phase 1 — MLM pretraining on train text only.** Continues `xlm-roberta-large`'s
   masked-language-model objective on the 17,769 training texts (`AutoModelForMaskedLM`,
   `DataCollatorForLanguageModeling` at `mlm_probability=0.15`), 10 epochs at lr 1e-5. Labels are
   never used in this phase — the goal is purely to adapt the encoder's representations to
   informal Azerbaijani/Russian/English micro-mobility language before classification training
   starts. Actual run: `train_runtime` 3206s (~53 min), final training loss 3.725 — longer than
   the notebook's initial 20-30 min estimate.
2. **Phase 2 — standard fine-tuning CV**: 5-fold `StratifiedKFold(random_state=42)`, weighted
   cross-entropy, batch 16 / grad-accum 2 (effective 32) / lr 1e-5 / bf16, 5 epochs — the same
   recipe used for every transformer trained in this project — except `from_pretrained(TAPT_DIR)`
   loads the MLM-adapted checkpoint instead of the generic pretrained one. bf16 (not fp16) is
   used throughout because XLM-R-large has a known fp16 NaN-loss instability; the L4 GPU runtime
   was chosen specifically for its bf16 support.
3. **Final retrain** on 100% of the data at 5 epochs (matching the plain model's chosen epoch
   count), producing the deployed checkpoint.

**Compliance note (deliberate design):** the notebook never loads `test.csv`. The MLM corpus is
`train.csv` text only — using test text in any training step, including unlabeled MLM, would
violate the competition's prohibition on using the test dataset for training.

## Why it helps

Standard TAPT effect: continuing the MLM objective on in-domain text before fine-tuning lets the
encoder adapt to this dataset's specific register (informal, typo-heavy, code-switched
Azerbaijani/Russian/English, domain-specific vocabulary around bike/scooter rentals) without
ever seeing labels — a form of unsupervised domain adaptation that stays fully compliant with
the no-test-in-training rule. The consistent per-fold gain (positive on all 5 folds, not just on
average) indicates a genuine representational improvement rather than a lucky split.

## Parameter budget consequence

At 559,893,507 parameters (identical count to a plain fine-tune — TAPT only continues training,
it doesn't change the architecture), this model fits the 600M cap solo or with the linear SVM
(~562M total) — but **cannot be combined with any other transformer** (any pairing would exceed
838M). This constrains its ensemble role to the 2-way blend documented in `blendtapt_model.md`.

## Relevant files (all paths relative to this bundle's root)

```
notebooks/
  xlmrlarge_tapt_colab.ipynb      TAPT (MLM + fine-tune CV) training notebook (Colab, L4 GPU)
models/xlmrlarge_tapt_model/      fine-tuned model weights (safetensors) + tokenizer files
artifacts/oof_predictions/xlmrlarge_tapt_oof_predictions.csv   out-of-fold probabilities (5-fold CV)
src/predict_blend_tapt.py         inference entry point that uses this model (as a blend member)
```

There is no standalone prediction script; `src/predict_blend_tapt.py`'s `transformer_probs()`
shows how to run the model alone if needed.

## Environment

**Training (Google Colab, L4 GPU runtime):**
- Python 3.12.13, torch 2.11.0+cu128, transformers 5.12.1, CUDA 12.8
- GPU: NVIDIA L4 (24GB, bf16)
- Phase 1 (MLM, 10 epochs): ~53 min (`train_runtime` 3206.45s)
- Phase 2 (5-fold CV): ~1h, comparable to the plain model's per-fold cost since the recipe is
  identical
- Final retrain: 5 epochs on 100% of the data, same cost profile as the plain model's final
  retrain (~12.5 min)

**Inference (local machine):** identical environment to the other transformers
(`requirements-xlmr.txt`, CPU-only torch, transformers 5.12.1). The fine-tuned checkpoint is
saved in fp32 and loads on CPU without any dtype issues.

## Reproduction steps

Upload `notebooks/xlmrlarge_tapt_colab.ipynb`, select **L4 GPU** runtime, data in private Drive,
run top to bottom: setup → version capture → mount Drive + load `train.csv` only → Phase 1 MLM
pretraining → save adapted checkpoint → Phase 2 fine-tuning setup → smoke test (fold 1) → full
5-fold CV → save OOF predictions → final retrain (gated on the adoption rule passing) → param
count + save. Download `final_model` to `models/xlmrlarge_tapt_model/` and the OOF CSV to
`artifacts/oof_predictions/`.

Documented failure-mode handling: OOM → batch 8 / accumulation 4; NaN loss on a T4 → switch to
an L4 runtime (bf16); F1 collapse on a fold → learning rate 5e-6.

## Estimated runtime

- **Training** (Colab, L4): Phase 1 MLM ~53 min + Phase 2 5-fold CV ~1h + final retrain
  ~12.5 min ≈ **~2h10m total**.
- **Inference** (local, CPU-only): ~13 minutes as part of the 2-way blend's ~13.1 total
  (`predict_blend_tapt.py`, 4,797 rows, batch 32).

## Reproducibility

- Same seeding scheme as all other models (`RANDOM_SEED = 42` everywhere, including the MLM
  phase).
- `MLM_EPOCHS = 10`, `MLM_PROBABILITY = 0.15` fixed a priori, not tuned on CV results.
- `FINAL_NUM_EPOCHS = 5`, matching the plain model's chosen epoch count.
- Expected benign warnings: `UNEXPECTED`/`MISSING` key report when loading the MLM-adapted
  checkpoint into the classification head (MLM head dropped, classification head freshly
  initialized) — same pattern as all other transformer models in the project.
- Run-to-run GPU non-determinism of ~±0.003 Macro F1 on a single fold was measured elsewhere in
  this project (identical config/fold/seed across sessions); the competition rules explicitly
  accept such hardware-level variance.

## External libraries and models used

| Library/Model | Version | License |
|---|---|---|
| `xlm-roberta-large` (FacebookAI) | pretrained checkpoint via Hugging Face | MIT |
| transformers | 5.12.1 | Apache-2.0 |
| torch | 2.11.0 (training, CUDA) / 2.12.1 (inference, CPU) | BSD-3-Clause |
| datasets | Colab-preinstalled version | Apache-2.0 |
| accelerate | Colab-preinstalled version | Apache-2.0 |
| pandas, scikit-learn | match `requirements.txt` | BSD-3-Clause |

No third-party datasets or synthetic data — only the competition-provided `train.csv`, and only
its text (never labels) during the MLM phase. All licenses permit commercial use, per
competition rules.

## Data usage

Competition data stays out of version control; `train.csv` uploaded only to a private Google
Drive for training (the MLM phase deliberately never loads `test.csv`), deleted from
Drive/Colab after training.

## Known limitations / project status

- MLM epoch count (10) and masking probability (0.15) were fixed a priori. A follow-up sweep
  (not part of this bundle) tested `{2, 5, 10}` MLM epochs and found no candidate beat 10 by
  more than the noise floor — closed, kept at 10.
- A separate fine-tuning learning-rate sweep (also not part of this bundle) found `2e-5` beats
  this model's `1e-5` — see `xlmrlarge_tapt_lr2e5_model.md` for that variant, which is this
  project's actual best-scoring locked submission.
- Residual error structure: the technical_support ↔ customer_support intent boundary, with a
  (shrinking) label-ambiguity floor — the project's persistent bottleneck across every model
  tried.
- This is a closed, final project snapshot; the recipe/hyperparameters here won't change further.
