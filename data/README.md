# Data

The competition dataset is **not included** in this repository and must not be committed
(it is excluded by `.gitignore`). Obtain it from the competition page and place the files here:

```
data/
  train.csv    columns: id, feedback, tag, label
  test.csv     columns: id, feedback, tag
```

| Column | Description |
|---|---|
| `id` | Row identifier |
| `feedback` | Free-text customer feedback (Azerbaijani, Russian, English; often written without diacritics) |
| `tag` | Optional part category (e.g. `Brakes`, `Battery`, `Kickstand`); empty for roughly 70% of rows |
| `label` | Target (`train.csv` only): `technical_support`, `customer_support` or `other` |

Scripts read `data/train.csv` and `data/test.csv` relative to the working directory, so run
them from the repository root.
