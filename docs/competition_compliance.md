# Competition rules compliance

This solution was built for a competition with a parameter cap, a CPU-only inference
environment, and a ban on using the test set for training. Each rule was checked against the
competition rulebook (Sections 2–5) and against the code, not just asserted.

- **Parameter cap (600M, incl. ensembles)**: both locked blends measure ~562M (TAPT XLM-R-large,
  559,893,507 params, printed and asserted at training time — plus a ~1–2M-parameter linear
  SVM). This is also why neither blend could add a second transformer (any pairing would exceed
  838M).
- **CPU-only inference pipeline**: verified in the actual inference code
  (`src/predict_blend_tapt.py`, `src/predict_blend_tapt_lr2e5.py`) — no `.to("cuda")`, no device
  placement calls anywhere; models load and run on whatever `torch` gives them by default (CPU),
  regardless of what hardware they happen to run on.
- **8 GB RAM limit**: measured at ~2.5 GB peak (fp32 large-model weights + activations),
  documented per-blend.
- **No test-set usage in training**: `test.csv` is loaded only inside the two inference scripts
  above, for generating predictions — never inside a `.fit()`/training call. Both TAPT training
  notebooks carry an explicit compliance note and load `train.csv` only, including during the
  unlabeled MLM phase (continuing pretraining on `test.csv` text, even without its labels, would
  itself violate this rule — deliberately avoided throughout).
- **Reproducibility (fixed seeds, same procedure → statistically equivalent results)**:
  `RANDOM_SEED = 42` fixed everywhere (`StratifiedKFold`, `LinearSVC`, MLM/fine-tuning training
  args) — see the Reproducibility section in each model doc. Hardware-driven run-to-run
  variance (~±0.003–0.006 Macro F1, measured across repeated runs of the identical config) is
  expected and accepted per the rules.
- **External models/libraries formally declared**: every model doc has an "External libraries
  and models used" table with version and license (all MIT/Apache-2.0/BSD-3-Clause, all permit
  commercial use). Only pretrained model used: `xlm-roberta-large` (FacebookAI, via Hugging Face,
  MIT) — no other external/pretrained models or third-party datasets.
- **No paid/commercial API calls in inference code**: confirmed by reading
  `predict_blend_tapt.py`/`predict_blend_tapt_lr2e5.py` — both load models from local disk
  (`from_pretrained(<local dir>)`) and never make a network call. Google Colab was used for
  *training* only, which the rules explicitly permit regardless of tier.
- **Competition dataset not redistributed**: `data/` in this repository is empty by design
  (train/test are never included). Notebook outputs that printed sample rows of real feedback
  text were cleared; everything else contains only aggregate metrics, row ids, labels and
  model scores.
- **Documentation package (Section 5's code-review requirement)**: dependency files
  (`requirements.txt`, `requirements-xlmr.txt`), environment specs, and estimated runtimes are
  present in every model doc, satisfying "fully documented ... dependency file, specifications
  of the utilized environment, and estimated runtime."

Not independently re-verified in this pass (carried forward from the original project's
documentation): the exact commercial-use terms of the `xlm-roberta-large` MIT license claim, and
the ~2.5 GB peak RAM figure, which was measured during actual runs rather than re-derived
analytically.
