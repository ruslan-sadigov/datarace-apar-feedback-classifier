"""Blend submission: linear SVM + TAPT-adapted XLM-R-large fine-tuned at lr=2e-05.

Diagnostic variant of predict_blend_tapt.py (lr=1e-05, deployed). An LR sweep
(notebooks/xlmrlarge_tapt_lr_sweep_colab.ipynb) found lr=2e-05 -- the base model's
default, only dropped to 1e-05 a priori as a large-model stability hedge -- beat the
lr=1e-05 anchor solo (mean CV 0.9284 vs 0.9260, 4/5 folds). At the blend level
(notebooks/lr_sweep_blend_check.ipynb) the gain shrank to +0.0010 (W_TAPT=0.4, full-OOF)
or +0.0014 (W_TAPT=0.35, full-OOF) -- both inside the ~0.003 noise floor, so this is not
a confirmed improvement, but a real enough signal (consistent direction across solo CV,
solo full-OOF, and blend full-OOF; lower nested fold variance 0.0038 vs 0.0061) to be
worth a leaderboard check.

W_TAPT = 0.35 (not 0.4): a nested per-fold weight retune picked 0.35 in 4/5 folds for
this model, and it also wins the apples-to-apples fixed-weight full-OOF comparison
(0.9298 vs 0.9294 at 0.4) -- both checks agree, so 0.35 is used here rather than reusing
the lr=1e-05 model's tuned weight.

Parameter budget: identical to the deployed TAPT blend (562M, XLM-R-large + SVM).
"""

import re

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TEST_PATH = "data/test.csv"
LINEAR_MODEL_PATH = "models/linear_model.joblib"
TAPT_MODEL_DIR = "models/xlmrlarge_tapt_lr2e5_model"
SUBMISSION_PATH = "submissions/submission_blendtapt_lr2e5_1.csv"

LABELS = ["technical_support", "customer_support", "other"]
W_TAPT = 0.35

MAX_LENGTH = 96
BATCH_SIZE = 32
TAG_MISSING_VALUE = "missing"


def clean_text_shared(text: str) -> str:
    """Cleaning steps shared by both models (repeated chars, whitespace)."""
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_for_linear(df: pd.DataFrame) -> pd.DataFrame:
    """Linear model preprocessing: also lowercases (matches src/linear_model.py)."""
    df = df.copy()
    df["feedback"] = df["feedback"].astype(str).str.lower().apply(clean_text_shared)
    df["tag"] = df["tag"].fillna(TAG_MISSING_VALUE)
    return df


def prepare_for_transformer(df: pd.DataFrame) -> pd.DataFrame:
    """Transformer preprocessing: no lowercasing, tag prepended (matches training notebook)."""
    df = df.copy()
    df["feedback"] = df["feedback"].astype(str).apply(clean_text_shared)
    df["tag"] = df["tag"].fillna(TAG_MISSING_VALUE)
    df["input_text"] = df["tag"] + " | " + df["feedback"]
    return df


def linear_probs(test_df: pd.DataFrame) -> np.ndarray:
    model = joblib.load(LINEAR_MODEL_PATH)
    class_order = list(model.named_steps["classifier"].classes_)
    scores = model.decision_function(test_df[["feedback", "tag"]])
    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)  # softmax
    return probs[:, [class_order.index(label) for label in LABELS]]


def transformer_probs(model_dir: str, test_df: pd.DataFrame) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    class_order = [model.config.id2label[i] for i in range(len(LABELS))]
    if class_order == [f"LABEL_{i}" for i in range(len(LABELS))]:
        # This checkpoint's config.json wasn't saved with id2label set (the LR sweep
        # notebook's final-retrain cell initialized the model with num_labels only).
        # Training used LABEL2ID = {l: i for i, l in enumerate(LABELS)} throughout this
        # project (see notebooks/xlmrlarge_tapt_lr_sweep_colab.ipynb), so id order matches
        # LABELS directly -- no remapping needed for this specific checkpoint.
        class_order = LABELS

    texts = test_df["input_text"].tolist()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            inputs = tokenizer(
                batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
            )
            logits = model(**inputs).logits
            all_probs.append(torch.softmax(logits, dim=1).numpy())
    probs = np.vstack(all_probs)
    return probs[:, [class_order.index(label) for label in LABELS]]


def main() -> None:
    test_df = pd.read_csv(TEST_PATH)

    p_linear = linear_probs(prepare_for_linear(test_df))
    print("Linear model probabilities computed.")
    p_tapt = transformer_probs(TAPT_MODEL_DIR, prepare_for_transformer(test_df))
    print("TAPT XLM-R-large (lr=2e-05) probabilities computed.")

    blend = W_TAPT * p_tapt + (1 - W_TAPT) * p_linear
    predictions = [LABELS[i] for i in blend.argmax(axis=1)]

    submission = pd.DataFrame({"id": test_df["id"], "label": predictions})

    assert submission["id"].nunique() == len(submission), "duplicate ids in submission"
    assert set(submission["label"]) <= set(LABELS), "unexpected label value in predictions"

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {len(submission)} predictions to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
