# Linear Model — TF-IDF + LinearSVC

Classifies multilingual (Azerbaijani / Russian / English) user feedback from the Apar
bike/scooter rental platform into one of three department labels: `technical_support`,
`customer_support`, `other`. Evaluated on Macro F1.

**Role in this bundle**: this model is not submitted on its own. It's the second (much
smaller) member of both locked 2-way blend submissions documented in
`blendtapt_model.md` and `blendtapt_lr2e5_model.md` — same fitted pipeline, same
`models/linear_model.joblib`, shared by both. The solo numbers below are for context on how
this component behaves in isolation, not a standalone deliverable of this bundle.

- **Mean 5-fold CV Macro F1 (solo):** 0.8879 (+/- 0.0082)
- **Public leaderboard score (solo, historical, not one of the two locked submissions):** 0.89901

## Approach summary

Full exploratory analysis and experiment log lives in `notebooks/linear_models.ipynb` (read
top to bottom — section headers and "Finding"/"Conclusion" notes explain the reasoning).
Short version:

- **Features:** word-level TF-IDF (uni+bigrams) and character-level TF-IDF (3-5 char
  n-grams, `char_wb`, robust to spelling/diacritic variation across AZ/RU/EN), both with
  `sublinear_tf=True`, plus one-hot encoding of the `tag` field (missing values treated as
  their own category).
- **Classifier:** `LinearSVC(C=0.5, class_weight="balanced")`, chosen after comparing
  against LogisticRegression and sweeping the regularization strength `C`.
- **Rejected during experimentation** (tested, did not help — see notebook for details):
  widening word n-grams to trigrams, cross-lingual keyword normalization/dictionary
  collapsing, decision-threshold offset tuning.
- **Known limitation:** feedback mentioning ride activation/unlock language (~6% of the
  data) has a much more balanced true-label split (55% technical / 42% customer) than the
  rest of the dataset, and the model underperforms there. This looks like a genuine
  intent-ambiguity in the labels themselves, not just a modeling gap — motivating a
  planned multilingual transformer follow-up to better capture context/intent rather than
  surface vocabulary.

## Relevant files (all paths relative to this bundle's root)

```
notebooks/
  linear_models.ipynb                        full experiment log for this model
src/
  linear_model.py                            preprocessing + feature pipeline + model definition
  train_linear.py                            5-fold CV, saves out-of-fold predictions, retrains on full data, saves model
models/linear_model.joblib                    fitted pipeline (trained on full train.csv)
artifacts/oof_predictions/linear_oof_predictions.csv   out-of-fold decision scores (5-fold CV)
requirements.txt                              pinned dependency versions
```

There is no standalone `predict_linear.py` in this bundle — the fitted pipeline is loaded
and consumed directly by the two blend inference scripts (`src/predict_blend_tapt.py`,
`src/predict_blend_tapt_lr2e5.py`), documented in `blendtapt_model.md` and
`blendtapt_lr2e5_model.md`.

## Environment

- **OS used for development:** Windows 11
- **Python:** 3.11.9
- **Hardware:** no GPU required or used; runs on a standard laptop CPU
- **Peak memory:** well under the 8 GB inference limit (TF-IDF + LinearSVC on ~18k short
  text rows is lightweight; the fitted pipeline is a few MB)
- **Model size:** LinearSVC is a linear model (sparse weight matrix over the TF-IDF
  vocabulary) — far below the 600M parameter limit

## Reproduction steps

```powershell
# 1. Create and activate a virtual environment (Python 3.11.x)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install pinned dependencies
python -m pip install -r requirements.txt

# 3. Place train.csv (competition data, not included -- see "Data usage" below) under data/

# 4. Train (5-fold CV + retrain on full data), from this bundle's root
python src/train_linear.py
```

`src/train_linear.py` prints per-fold and mean out-of-fold Macro F1, saves out-of-fold
predictions to `artifacts/oof_predictions/linear_oof_predictions.csv`, and saves the final
fitted pipeline (retrained on the full training set) to `models/linear_model.joblib` — the
file the two blend scripts load. See `blendtapt_model.md` / `blendtapt_lr2e5_model.md` for
the actual submission-generating step.

## Estimated runtime

Measured on the development machine (Windows 11, CPU-only, no GPU):

- `train_linear.py` (5-fold CV + final full-data fit): **~9 seconds**
- Inference cost as part of each blend is documented in `blendtapt_model.md` /
  `blendtapt_lr2e5_model.md` (dominated by the transformer branch; this component's own
  contribution is a few seconds).

## Reproducibility

- A single fixed `RANDOM_SEED = 42` (defined in `src/linear_model.py`) is used for the
  `StratifiedKFold` splits and the `LinearSVC` random state. Re-running `train_linear.py`
  from a clean checkout reproduces the same CV scores.
- Dependency versions are pinned in `requirements.txt` to match the versions used to
  produce the scores above.

## External libraries and models used

Only standard open-source Python libraries — no external pretrained models, embeddings,
or third-party datasets are used in this model:

| Library | Version | License |
|---|---|---|
| scikit-learn | 1.9.0 | BSD-3-Clause |
| pandas | 3.0.3 | BSD-3-Clause |
| numpy | 2.4.6 | BSD-3-Clause |
| joblib | 1.5.3 | BSD-3-Clause |

(matplotlib/seaborn/jupyter are used only for the exploratory notebooks, not the
inference pipeline.) All licenses permit commercial use, per competition rules.

## Data usage

`train.csv`, `test.csv`, and `sample.csv` are the property of Apar and are not
included anywhere in this project's version control, per the competition's data usage
rules. Place them under `data/` locally to reproduce the results above.
