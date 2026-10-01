# FR-3 categorization proof of concept

Evidence for the numbers in *FR-3 Transaction Categorization — Feature Design*. This is throwaway experiment code: it reads truth tables directly and is not part of the product package.

## Reproduce

```sh
uv run sfc-data generate --spec configs/data/default.yaml --out data/synthetic/default.sqlite
uv run experiments/fr3_categorization/feasibility.py data/synthetic/default.sqlite
```

The script declares its own dependencies (PEP 723), including `fastembed`, so the project's lock file is unchanged. The first run downloads the embedding model (about 130 MB). A full run takes about 2.5 minutes on a laptop CPU.

## What it does

- **Splits** (Technical Design): known merchants are a stratified 20% of train users' transactions. Unseen merchants are test users' transactions at holdout merchants. "All test users" is every test-user transaction.
- **Training:** the other 80% of train users' transactions, at most 20,000 per class, seed 0. No label noise.
- **Models:** multinomial logistic regression with balanced class weights over three feature sets: character 2–4-grams of `merchant_raw`, a frozen `bge-small-en-v1.5` embedding of the normalized text, or both. Each also gets amount bin, sign, channel and hour.
- **Metrics:** macro F1 over the 12 spending categories (Income excluded), per-class F1, the largest error groups, and a 95% interval for unseen merchants from 1,000 bootstrap resamples of merchants within each category.

## Results

- [results/feasibility.md](results/feasibility.md): readable report.
- [results/feasibility.json](results/feasibility.json): everything, including per-class precision and recall for every test set.

A second run gives identical metrics.

## Not done here

No hyperparameter search or confidence calibration, no keyword baseline, and no gradient-boosted tree or transformer candidates. Those belong to the FR-3 milestones.
