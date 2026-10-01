# FR-3 categorization proof of concept

Evidence for the numbers in *FR-3 Transaction Categorization — Feature Design*. This is throwaway experiment code: it reads truth tables directly and is not part of the product package.

## Reproduce

```sh
uv run sfc-data generate --spec configs/data/default.yaml --out data/synthetic/default.sqlite
uv run experiments/fr3_categorization/feasibility.py data/synthetic/default.sqlite
uv run experiments/fr3_categorization/plots.py      # charts, HTML report, screenshot; no retraining
```

Both scripts declare their own dependencies (PEP 723), including `fastembed`, so the project's lock file is unchanged. The first run downloads the embedding model (about 130 MB). A full run takes about 2.5 minutes on a laptop CPU.

## What it does

- **Splits** (Technical Design): known merchants are a stratified 20% of train users' transactions. Unseen merchants are test users' transactions at holdout merchants. "All test users" is every test-user transaction.
- **Training:** the other 80% of train users' transactions, at most 20,000 per class, seed 0. No label noise.
- **Models:** multinomial logistic regression with balanced class weights over three feature sets: character 2–4-grams of `merchant_raw`, a frozen `bge-small-en-v1.5` embedding of the normalized text, or both. Each also gets amount bin, sign, channel and hour.
- **Metrics:** macro F1 over the 12 spending categories (Income excluded), per-class F1, the largest error groups, and a 95% interval for unseen merchants from 1,000 bootstrap resamples of merchants within each category.

## Results

| File | What it is |
| --- | --- |
| [results/screenshots/report.png](results/screenshots/report.png) | Full-page screenshot of the HTML report: stat tiles, tables and every chart |
| [results/report.html](results/report.html) | The same report; open locally (it loads charts from `figures/`) |
| [results/figures/](results/figures/) | Each chart as a PNG |
| [results/feasibility.md](results/feasibility.md) | Text report with every table |
| [results/feasibility.json](results/feasibility.json) | Everything: per-class precision and recall, confusion matrices, per-merchant accuracy, bootstrap samples |

A second run gives identical metrics.

### Headline: FR-3's target is met; FR-4's is not

![Macro F1 by test set](results/figures/headline.png)

### The unseen-merchant gap is not sampling noise

No bootstrap resample of any model reaches 0.80; the best is 0.793.

![Bootstrap distribution](results/figures/unseen_bootstrap.png)

### Real chains are hardest; embeddings carry outside knowledge

![Accuracy by merchant type](results/figures/unseen_by_scope.png)

![Per-class F1](results/figures/per_class_f1.png)

![Unseen-merchant confusion](results/figures/unseen_confusion_both.png)

### Per merchant

![Per-merchant accuracy](results/figures/unseen_per_merchant.png)

### Dataset

![Transactions per category](results/figures/dataset_categories.png)

Charts use the dataviz reference palette; the three series colours were checked with its colour-vision validator (all pairs pass).

## Not done here

No hyperparameter search or confidence calibration, no keyword baseline, and no gradient-boosted tree or transformer candidates. Those belong to the FR-3 milestones.
