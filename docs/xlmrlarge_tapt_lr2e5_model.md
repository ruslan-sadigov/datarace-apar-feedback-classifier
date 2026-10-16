# XLM-RoBERTa-large with task-adaptive pretraining (TAPT), fine-tuned at lr=2e-5

Same TAPT approach as `xlmrlarge_tapt_model.md` (continue `xlm-roberta-large`'s
masked-language-model objective on the competition's training texts before any label is seen,
then fine-tune for classification from the adapted checkpoint) — this document covers the
**Phase 2 fine-tuning learning rate** variant that supersedes it. `xlmrlarge_tapt_model.md`
used `lr=1e-5`, chosen a priori as a cautious, smaller-than-default value out of concern for
large-transformer training instability. By the time this sweep ran, every transformer trained
in this project on bf16/L4 had been fully stable (zero NaN losses), so that caution was
re-examined directly instead of assumed.

- **Mean 5-fold CV Macro F1 (solo):** 0.9284
- **Public leaderboard score:** not submitted solo — used as a member of the 2-way blend
  documented in `blendtapt_lr2e5_model.md` (public **0.93446**, this project's best-scoring
  submission)
- **Status:** the TAPT model used by this project's **primary locked final submission**.

## The learning-rate sweep

Reuses the already-trained MLM-adapted checkpoint from `xlmrlarge_tapt_model.md`'s Phase 1
directly (Phase 1 is not part of this sweep — only the classification fine-tuning stage, Phase
2, varies). Swept `FT_LR_GRID = [5e-6, 1e-5, 2e-5]`:
- `5e-6`: this project's documented fallback for fine-tuning instability, worth checking on its
  own merits, not just as a safety net.
- `1e-5`: the previously deployed anchor (`xlmrlarge_tapt_model.md`).
- `2e-5`: the learning rate already used by the smaller base-size transformers in this project —
  worth re-testing on the large model now that bf16 training had proven stable throughout.

Same 5-fold `StratifiedKFold(random_state=42)`, same weighted cross-entropy, same batch 16 /
grad-accum 2 (effective 32) / bf16 / 5 epochs recipe as every other transformer in this project —
only the fine-tuning learning rate changes across the sweep. Adoption gate (pre-registered):
a candidate must beat the `1e-5` anchor's mean CV by more than the ~0.006 documented noise floor
in this *same sweep* (paired comparison, same folds/seed/hardware — cancels ordinary run-to-run
GPU noise), not against an older cross-session number.

**Result: `lr=2e-5` cleared the gate.** Solo mean CV moved from 0.9272 (`lr=1e-5`) to 0.9284.

## Approach summary

Training notebook: `notebooks/xlmrlarge_tapt_lr_sweep_colab.ipynb` (Google Colab, **L4 GPU**
runtime). Loads the existing `mlm_epoch_10` MLM checkpoint (produced once, reused across all
three learning-rate candidates — Phase 1 is not repeated). For the winning `lr=2e-5`:

- Same preprocessing as every transformer in this project: input `"tag | feedback"`, no
  lowercasing, `MAX_LENGTH = 96`.
- 5-fold CV at `lr=2e-5`, then a final retrain on 100% of the training data at 5 epochs,
  producing the deployed checkpoint (`models/xlmrlarge_tapt_lr2e5_model/`).
- bf16 throughout (not fp16) — XLM-R-large has a known fp16 NaN-loss instability; the L4 GPU
  runtime was chosen for its bf16 support.

**Compliance note (deliberate design):** the MLM phase this checkpoint is built on used
`train.csv` text only; `test.csv` is never loaded anywhere in this sweep, in either phase.

## Why it helps

The a priori caution behind picking a smaller-than-default fine-tuning learning rate for a
large transformer turned out not to be necessary once bf16 training was already proven stable
throughout this project — `2e-5` (the base-size models' learning rate) fine-tunes the
TAPT-adapted large encoder faster and to a better optimum than the originally-chosen `1e-5`,
without any of the instability the smaller rate was meant to guard against.

## Parameter budget consequence

At 559,893,507 parameters (identical count to every other variant — only the fine-tuning
learning rate changes, not the architecture), this model fits the 600M cap solo or with the
linear SVM (~562M total) — but **cannot be combined with any other transformer** (any pairing
would exceed 838M). This constrains its ensemble role to the 2-way blend documented in
`blendtapt_lr2e5_model.md`.

## Relevant files (all paths relative to this bundle's root)

```
notebooks/
  xlmrlarge_tapt_lr_sweep_colab.ipynb   fine-tuning LR sweep notebook (Colab, L4 GPU)
models/xlmrlarge_tapt_lr2e5_model/      fine-tuned model weights (safetensors) + tokenizer files
artifacts/oof_predictions/xlmrlarge_tapt_lr2e-05_oof_predictions.csv   out-of-fold probabilities (5-fold CV)
src/predict_blend_tapt_lr2e5.py         inference entry point that uses this model (as a blend member)
```

There is no standalone prediction script; `src/predict_blend_tapt_lr2e5.py`'s
`transformer_probs()` shows how to run the model alone if needed.

## Environment

**Training (Google Colab, L4 GPU runtime):**
- Python 3.12.13, torch 2.11.0+cu128, transformers 5.12.1, CUDA 12.8
- GPU: NVIDIA L4 (24GB, bf16)
- Phase 2 (5-fold CV) for the winning `lr=2e-5` candidate: ~1h, same per-fold cost as every
  other fine-tuning run in this project since only the learning rate differs.
- Final retrain: 5 epochs on 100% of the data, ~12.5 min.

**Inference (local machine):** identical environment to `xlmrlarge_tapt_model.md`
(`requirements-xlmr.txt`, CPU-only torch, transformers 5.12.1). The fine-tuned checkpoint is
saved in fp32 and loads on CPU without any dtype issues.

## Reproduction steps

Requires the `mlm_epoch_10` MLM checkpoint from `xlmrlarge_tapt_model.md`'s Phase 1 (or
re-running that phase fresh, since the underlying corpus and recipe are unchanged). Upload
`notebooks/xlmrlarge_tapt_lr_sweep_colab.ipynb`, select **L4 GPU** runtime, data in private
Drive, run top to bottom: setup → version capture → load `train.csv` + the MLM checkpoint →
smoke test → LR sweep (5-fold CV per candidate) → adoption gate → final retrain at the winning
learning rate → param count + save. Download `final_model` to
`models/xlmrlarge_tapt_lr2e5_model/` and the OOF CSV to `artifacts/oof_predictions/`.

Documented failure-mode handling: OOM → batch 8 / accumulation 4; NaN loss on a T4 → switch to
an L4 runtime (bf16); F1 collapse on a fold → learning rate 5e-6.

## Estimated runtime

- **Training** (Colab, L4): ~1h for the winning candidate's 5-fold CV + ~12.5 min final retrain
  (the other two LR candidates in the sweep add ~1h each, already spent to find the winner).
- **Inference** (local, CPU-only): ~13 minutes as part of the 2-way blend's total
  (`predict_blend_tapt_lr2e5.py`, 4,797 rows, batch 32).

## Reproducibility

- Same seeding scheme as every model in this project (`RANDOM_SEED = 42` everywhere).
- The sweep's adoption decision was made by comparing candidates against this *same sweep's*
  own `lr=1e-5` point (paired, same folds/seed/hardware), not an older cross-session number —
  this cancels ordinary run-to-run GPU noise out of the comparison.
- Expected benign warnings: `UNEXPECTED`/`MISSING` key report when loading the MLM-adapted
  checkpoint into the classification head (MLM head dropped, classification head freshly
  initialized) — same pattern as every other transformer model in this project.
- Run-to-run GPU non-determinism of ~±0.003-0.006 Macro F1 was measured elsewhere in this
  project (identical config/fold/seed across sessions); the competition rules explicitly accept
  such hardware-level variance.

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

- The MLM phase (epoch count, masking probability) is unchanged from `xlmrlarge_tapt_model.md`
  and was not re-swept here — this document is scoped to the fine-tuning learning-rate question
  only.
- Residual error structure: the technical_support ↔ customer_support intent boundary, with a
  (shrinking) label-ambiguity floor — the project's persistent bottleneck across every model
  tried.
- A later attempt to retrain the MLM phase itself on a spelling-corrected version of the
  training data (separate from this document's learning-rate question) was tested at the blend
  level and did not clear its adoption gate (blend-OOF peak +0.0012 over anchor, below the 0.006
  noise floor) — closed, not part of this bundle.
- This is a closed, final project snapshot; the recipe/hyperparameters here won't change further.
