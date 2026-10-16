# Documentation index

One document per model or blend. Each covers the approach, results, environment, reproduction
steps, runtime, reproducibility and the external libraries used.

| Document | What it covers | Score |
|---|---|---|
| [`blendtapt_lr2e5_model.md`](blendtapt_lr2e5_model.md) | **Primary** blend: TAPT XLM-R-large (lr=2e-5) + LinearSVC, `W_TAPT=0.35` | public 0.93446 / CV 0.9298 |
| [`blendtapt_model.md`](blendtapt_model.md) | Secondary blend: TAPT XLM-R-large (lr=1e-5) + LinearSVC, `W_TAPT=0.4`; temperature-scaling analysis | public 0.93120 / CV 0.9284 |
| [`xlmrlarge_tapt_lr2e5_model.md`](xlmrlarge_tapt_lr2e5_model.md) | Transformer component of the primary blend; fine-tuning learning-rate sweep | CV 0.9284 (solo) |
| [`xlmrlarge_tapt_model.md`](xlmrlarge_tapt_model.md) | Transformer component of the secondary blend; task-adaptive pretraining (TAPT) | CV 0.9272 (solo) |
| [`linear_model.md`](linear_model.md) | TF-IDF + LinearSVC component shared by both blends | CV 0.8879 (solo) |
| [`competition_compliance.md`](competition_compliance.md) | How the solution meets each competition rule | — |

For the ideas that were tried and rejected, see [`../EXPERIMENTS.md`](../EXPERIMENTS.md).
