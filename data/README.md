# Data

The competition dataset belongs to Apar. It is **not included** in this repository and must not
be committed (it is excluded by `.gitignore`). The competition has ended and the dataset is no
longer publicly available, so the training and submission scripts cannot be re-run as-is. The
format the scripts expect is documented below, so you can run them on your own data with the
same columns. If you have a legitimate copy, place the files here:

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
