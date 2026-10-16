import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from linear_model import RANDOM_SEED, build_model_pipeline, prepare_dataframe

N_SPLITS = 5
TRAIN_PATH = "data/train.csv"
MODEL_PATH = "models/linear_model.joblib"
OOF_PREDICTIONS_PATH = "artifacts/oof_predictions/linear_oof_predictions.csv"


def run_cross_validation(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    X = df[["feedback", "tag"]]
    y = df["label"]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    fold_macro_f1 = []
    oof_pred = pd.Series(index=df.index, dtype=object)
    class_order = None
    oof_scores = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model_pipeline()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if class_order is None:
            class_order = model.named_steps["classifier"].classes_
            oof_scores = np.zeros((len(df), len(class_order)))
        oof_scores[val_idx] = model.decision_function(X_val)

        macro_f1 = f1_score(y_val, y_pred, average="macro")
        fold_macro_f1.append(macro_f1)
        oof_pred.iloc[val_idx] = y_pred

        print(f"Fold {fold_idx}: Macro F1 = {macro_f1:.4f}")

    print(f"\nMean Macro F1: {np.mean(fold_macro_f1):.4f} (+/- {np.std(fold_macro_f1):.4f})")
    print("\nOut-of-fold classification report:")
    print(classification_report(y, oof_pred, digits=4))

    oof_scores_df = pd.DataFrame(
        oof_scores, columns=[f"score_{c}" for c in class_order], index=df.index
    )
    return oof_pred, oof_scores_df


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    train_df = prepare_dataframe(train_df)

    oof_pred, oof_scores_df = run_cross_validation(train_df)

    oof_df = train_df[["id", "label"]].copy()
    oof_df = pd.concat([oof_df, oof_scores_df], axis=1)
    oof_df["oof_pred"] = oof_pred
    os.makedirs(os.path.dirname(OOF_PREDICTIONS_PATH), exist_ok=True)
    oof_df.to_csv(OOF_PREDICTIONS_PATH, index=False)

    final_model = build_model_pipeline()
    final_model.fit(train_df[["feedback", "tag"]], train_df["label"])
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"\nSaved final model (trained on full data) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
