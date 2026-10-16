# Models

The transformer checkpoints are ~2.2 GB each, over GitHub's file-size limit, so they are **not
stored in this repository** (excluded by `.gitignore`). Only the small linear model is tracked.

Expected layout for running inference from the repository root:

```
models/
  linear_model.joblib                 tracked (~3 MB) — TF-IDF + LinearSVC pipeline
  xlmrlarge_tapt_lr2e5_model/         primary blend's transformer   (download)
    config.json  model.safetensors  tokenizer.json  tokenizer_config.json
  xlmrlarge_tapt_model/               secondary blend's transformer (download)
    config.json  model.safetensors  tokenizer.json  tokenizer_config.json
```

| Directory | Used by | Fine-tune LR | Parameters |
|---|---|---|---|
| `xlmrlarge_tapt_lr2e5_model/` | `src/predict_blend_tapt_lr2e5.py` (primary) | 2e-5 | 559,893,507 |
| `xlmrlarge_tapt_model/` | `src/predict_blend_tapt.py` (secondary) | 1e-5 | 559,893,507 |

## Download

The weights are hosted on the Hugging Face Hub. Run these from the repository root:

```powershell
hf download HuGoBass09/apar-xlmr-large-tapt-lr2e5 --local-dir models/xlmrlarge_tapt_lr2e5_model
hf download HuGoBass09/apar-xlmr-large-tapt-lr1e5 --local-dir models/xlmrlarge_tapt_model
```

| Hub repository | Local directory |
|---|---|
| [`HuGoBass09/apar-xlmr-large-tapt-lr2e5`](https://huggingface.co/HuGoBass09/apar-xlmr-large-tapt-lr2e5) | `models/xlmrlarge_tapt_lr2e5_model/` |
| [`HuGoBass09/apar-xlmr-large-tapt-lr1e5`](https://huggingface.co/HuGoBass09/apar-xlmr-large-tapt-lr1e5) | `models/xlmrlarge_tapt_model/` |

`hf` comes with `huggingface_hub` (`python -m pip install -U huggingface_hub`).

Alternatively, retrain them: see [`../docs/xlmrlarge_tapt_lr2e5_model.md`](../docs/xlmrlarge_tapt_lr2e5_model.md)
and [`../docs/xlmrlarge_tapt_model.md`](../docs/xlmrlarge_tapt_model.md) (Colab notebooks in
[`../notebooks/`](../notebooks/)).
