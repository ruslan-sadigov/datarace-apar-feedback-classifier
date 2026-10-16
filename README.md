# Apar feedback classifier

Multilingual (Azerbaijani / Russian / English) customer-feedback classification for the Apar
bike and scooter rental platform. Each message is routed to one of three departments —
`technical_support`, `customer_support` or `other` — and models are scored on **Macro F1**.

The final solution is a soft-voting blend of a task-adaptively pretrained `xlm-roberta-large`
and a TF-IDF + LinearSVC model. It scored **0.93446** on the public leaderboard.

## Results

| Submission | Public LB | CV (OOF) | Details |
|---|---|---|---|
| **Primary** — TAPT XLM-R-large (lr=2e-5) + LinearSVC, `W_TAPT=0.35` | **0.93446** | 0.9298 | [`docs/blendtapt_lr2e5_model.md`](docs/blendtapt_lr2e5_model.md) |
| Secondary — TAPT XLM-R-large (lr=1e-5) + LinearSVC, `W_TAPT=0.4` | 0.93120 | 0.9284 | [`docs/blendtapt_model.md`](docs/blendtapt_model.md) |

Public leaderboard scores are computed on a subset of the test set. CV is 5-fold out-of-fold
Macro F1 on the training data.

## How it works

```
feedback text + tag
        │
        ├─► TAPT XLM-R-large ──► softmax probabilities ───┐
        │   input: "tag | feedback", case preserved       │  blend = 0.35·P_tapt + 0.65·P_svm
        │                                                 ├─► argmax ──► label
        └─► TF-IDF (word 1–2 + char 3–5) + tag one-hot    │
            LinearSVC ──► softmaxed decision scores ──────┘
            input: lowercased
```

- **TAPT (task-adaptive pretraining).** `xlm-roberta-large` first continues its masked-language-model
  training on the training texts only (labels unused), then is fine-tuned for classification.
  On 5-fold CV this beat plain fine-tuning on every fold (0.9272 vs 0.9207 mean).
- **Linear model.** Word and character TF-IDF are robust to the spelling and diacritic variation
  common in the Azerbaijani text. It is much weaker alone (0.8879 CV) but adds a complementary
  signal in the blend.
- **Blend weight.** The transformer's nominal weight is below one half because the softmaxed SVM
  scores are far softer than the transformer's probabilities. The weight corrects the scales; it
  does not mean the SVM is trusted more.

The remaining errors are mostly at the `technical_support` ↔ `customer_support` boundary, which
reflects label ambiguity rather than a modeling gap. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the
ideas that were tried and rejected, and why.

## Quickstart

Requires Python 3.11. Inference is CPU-only and takes about 13 minutes for the 4,797 test rows.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-xlmr.txt
```

Then add the two things that are not in this repository:

1. **Data** — place `train.csv` and `test.csv` in `data/` (see [`data/README.md`](data/README.md)).
2. **Model weights** — place the transformer checkpoints in `models/` (see [`models/README.md`](models/README.md)).

Run from the repository root (all paths are relative to the working directory):

```powershell
python src/predict_blend_tapt_lr2e5.py    # primary submission
python src/predict_blend_tapt.py          # secondary submission
```

To retrain the linear model (about 9 seconds): `python src/train_linear.py`. The transformers
were trained on Google Colab (L4 GPU); see the notebooks and per-model docs.

## Repository layout

```
README.md                    this file
EXPERIMENTS.md               what was tried, what worked, what was dropped
LICENSE                      MIT
requirements.txt             core dependencies (scikit-learn, pandas, ...)
requirements-xlmr.txt        transformer inference dependencies
src/                         training and inference scripts
notebooks/                   Colab training notebooks and local analysis notebooks
docs/                        one document per model or blend (index in docs/README.md)
models/                      linear_model.joblib (tracked); transformer weights are downloaded
artifacts/oof_predictions/   out-of-fold predictions used to tune the blend weight
submissions/                 the two generated submission CSVs
data/                        empty — add the competition files here
```

| Path | Purpose |
|---|---|
| `src/linear_model.py`, `src/train_linear.py` | Linear pipeline definition and training with 5-fold CV |
| `src/predict_blend_tapt_lr2e5.py` | Primary inference script |
| `src/predict_blend_tapt.py` | Secondary inference script |
| `notebooks/xlmrlarge_tapt_colab.ipynb` | TAPT (MLM + 5-fold fine-tune), lr=1e-5 |
| `notebooks/xlmrlarge_tapt_lr_sweep_colab.ipynb` | Fine-tuning learning-rate sweep, produces the lr=2e-5 model |
| `notebooks/blend_tapt_tuning.ipynb` | Blend-weight tuning and temperature-scaling analysis |
| `notebooks/linear_models.ipynb` | Full experiment log for the linear model |

## Constraints the solution was built under

- 600M parameter cap including ensembles: the blend is ~562M (559.9M transformer plus a ~1–2M
  linear model), which is why no second transformer fits.
- CPU-only inference within 8 GB RAM (about 2.5 GB peak measured).
- No test-set data in any training step, including the unlabeled MLM phase.
- Fixed seeds (`RANDOM_SEED = 42`). Run-to-run GPU variance of ~±0.003–0.006 Macro F1 is expected.

The full check against the rulebook is in [`docs/competition_compliance.md`](docs/competition_compliance.md).

## Data

The competition dataset belongs to Apar and is not redistributed here. The OOF prediction
files contain only row ids, labels and model scores, and the submission files contain only ids
and predicted labels.

## License

Code is released under the [MIT License](LICENSE). The pretrained `xlm-roberta-large` model
(FacebookAI, via Hugging Face) is MIT licensed; other dependencies are Apache-2.0 or BSD-3-Clause.
