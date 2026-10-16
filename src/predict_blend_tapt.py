"""Blend submission: linear SVM + TAPT-adapted XLM-R-large, soft-voting on probabilities.

The TAPT model (task-adaptive MLM pretraining on train.csv text, then the same
5-epoch classification recipe) scores CV Macro F1 0.9272 solo, beating plain
XLM-R-large (0.9207) on all 5 folds. Weight (W_TAPT = 0.4) was tuned on out-of-fold
predictions in notebooks/blend_tapt_tuning.ipynb (CV Macro F1 0.9284, vs 0.9238 for
the plain-large blend and 0.9254 for the 3-way blend). Temperature scaling was
tested under a nested fold-wise protocol and rejected: the nested held-out mean was
flat (0.9294 at T<=5, 0.9293 at T<=20) while the selected T and w kept running to
the grid's edge, indicating grid-search overfitting on a flat surface rather than a
real effect worth deploying.

Parameter budget: XLM-R-large = 559,893,507 + SVM sparse weights ~2M ≈ 562M, under the
competition's 600M cap. This model's size permits no other transformer partner.
Inference is CPU-only.
"""

import re

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TEST_PATH = "data/test.csv"
LINEAR_MODEL_PATH = "models/linear_model.joblib"
TAPT_MODEL_DIR = "models/xlmrlarge_tapt_model"
SUBMISSION_PATH = "submissions/submission_blendtapt_1.csv"

LABELS = ["technical_support", "customer_support", "other"]
W_TAPT = 0.4

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
    print("TAPT XLM-R-large probabilities computed.")

    blend = W_TAPT * p_tapt + (1 - W_TAPT) * p_linear
    predictions = [LABELS[i] for i in blend.argmax(axis=1)]

    submission = pd.DataFrame({"id": test_df["id"], "label": predictions})

    assert submission["id"].nunique() == len(submission), "duplicate ids in submission"
    assert set(submission["label"]) <= set(LABELS), "unexpected label value in predictions"

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {len(submission)} predictions to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
