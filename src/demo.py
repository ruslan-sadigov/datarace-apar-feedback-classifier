"""Classify a few feedback messages with the primary blend (no dataset needed).

Reuses the preprocessing and model-loading code of predict_blend_tapt_lr2e5.py, so the
output matches what the submission pipeline would produce for the same text.

Run from the repository root:

    python src/demo.py                          # built-in example messages
    python src/demo.py "pedal is broken" "..."  # your own messages
    python src/demo.py --tag Brakes "brake does not work"

The transformer is read from models/xlmrlarge_tapt_lr2e5_model/ if that folder exists,
otherwise it is downloaded from the Hugging Face Hub on first use (~2.2 GB, then cached).
The linear model (models/linear_model.joblib) is part of this repository.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import predict_blend_tapt_lr2e5 as blend  # noqa: E402

HUB_REPO = "HuGoBass09/apar-xlmr-large-tapt-lr2e5"

# Invented examples (not taken from the competition data).
EXAMPLES = [
    "velosipedin tormozu isləmir",
    "kartdan eyni sürüş üçün iki dəfə pul çıxdı, geri qaytarın",
    "Çox yaxşı xidmətdir, təşəkkür edirəm",
    "Не работает замок на самокате",
    "The app charged me twice for one ride",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("messages", nargs="*", help="feedback messages to classify")
    parser.add_argument("--tag", default=None, help="optional part tag, e.g. Brakes (applied to all)")
    args = parser.parse_args()

    messages = args.messages or EXAMPLES
    df = pd.DataFrame({"feedback": messages, "tag": args.tag})

    model_dir = blend.TAPT_MODEL_DIR if os.path.isdir(blend.TAPT_MODEL_DIR) else HUB_REPO
    print(f"Transformer: {model_dir}")

    p_linear = blend.linear_probs(blend.prepare_for_linear(df))
    p_tapt = blend.transformer_probs(model_dir, blend.prepare_for_transformer(df))
    p_blend = blend.W_TAPT * p_tapt + (1 - blend.W_TAPT) * p_linear

    print()
    for text, probs in zip(messages, p_blend):
        best = probs.argmax()
        scores = "  ".join(f"{label}={p:.2f}" for label, p in zip(blend.LABELS, probs))
        print(f"{blend.LABELS[best]:<18} {text}")
        print(f"{'':<18} {scores}")


if __name__ == "__main__":
    main()
