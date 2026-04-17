# XGBoost — Breast Cancer Wisconsin (Diagnostic)

This project trains an **XGBoost** classifier on the Kaggle dataset **Breast Cancer Wisconsin (Diagnostic)** to predict whether a tumor is **malignant (M)** or **benign (B)**.

- **Dataset**: [Breast Cancer Wisconsin (Diagnostic) on Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) (`data.csv`)
- **Task**: Binary classification (`diagnosis` → B vs M, encoded as 0/1)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data

Place the Kaggle file as:

- `./data.csv`

After cleaning (drop `id`, empty `Unnamed: 32`, keep numeric features + target): **569 rows**, **30 features**, class mix **B = 357**, **M = 212** (~62.7% / ~37.3%).

## Run

Open and run the notebook:

- `xgboost_breast_cancer.ipynb`

Or execute it headlessly:

```bash
jupyter nbconvert --to notebook --execute xgboost_breast_cancer.ipynb --output executed.ipynb
```

## Results (from notebook run)

**Split (stratified 70% / 15% / 15%):** train **398** × 30, validation **85** × 30, test **86** × 30. Positive (malignant) rate stays ~**37.2%** across splits.

### Cross-validation (all labeled data)

| Metric | Value |
|--------|--------|
| 10-fold stratified CV accuracy (mean) | **0.9579** |
| 10-fold stratified CV accuracy (std) | **0.0295** |

### Hyperparameter search (train folds only)

| Metric | Value |
|--------|--------|
| Best `RandomizedSearchCV` accuracy (5-fold on train) | **0.9597** |

Example best parameters from that run (see notebook cell output for the full dict): `n_estimators=500`, `min_child_weight=5`, `subsample=1.0`, `reg_lambda=1.0`, `reg_alpha=0.1`, plus `max_depth` and other fields as printed.

### Held-out test set (single 86-sample split)

| Model stage | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------------|----------|-----------|--------|-----|---------|
| Baseline | 0.9767 | 1.0000 | 0.9375 | 0.9677 | 1.0000 |
| Tuned (fit on train only) | 0.9884 | 1.0000 | 0.9688 | 0.9841 | 1.0000 |
| Final (refit on train+val, then test) | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

**Validation snapshot (85 samples):** baseline and tuned both reached **~97.65%** accuracy on validation; baseline validation **ROC-AUC = 0.9912**, tuned validation **ROC-AUC ≈ 0.9906** (two malignant false negatives in that fold).

The notebook tunes with **`scoring="accuracy"`**, refits the best hyperparameters on **train + validation**, then evaluates **once** on the held-out **test** set (no test labels used during search).

## Discussion

- **Accuracy vs ROC-AUC:** Validation **ROC-AUC** was very high (~**0.99**) while **classification accuracy** on the same split was ~**97.7%** for baseline and tuned models; always report which metric you mean. Test **accuracy** for the tuned model was **0.9884** before the train+val refit, and **1.0** on this run’s test split for the final refit.
- **Errors before the final refit:** Baseline test had **2 false negatives** (malignant predicted benign), tuned train-only test had **1**; the final refit cleared those on this split. In clinical use, **false negatives** are usually the most costly mistake; reporting recall and confusion counts matters as much as accuracy.
- **Why CV accuracy (~0.96) differs from test (~0.99–1.0):** CV averages performance across many folds on overlapping data; the single test split is small (**86** cases), so metrics move in steps of about **1/86 ≈ 1.2%** per misclassified example. AUC can be **1.0** on a small holdout while accuracy is still below perfect.
- **Perfect test scores:** They are possible on this dataset but should be interpreted cautiously: **one** new random split could yield 97–99% again. For a course report, emphasize **procedure** (stratified split, tune on train, refit on train+val, report test once) and **error types**, not only a single accuracy scalar.

## Conclusions

1. **XGBoost fits this problem well:** With default-ish strong settings, validation accuracy was already **~97.7%**, with very high ROC-AUC (~**0.99**).
2. **Tuning + refit helped:** Randomized search improved **cross-validated accuracy on the training portion** to ~**0.96** mean CV on the benchmark cell and ~**0.96** best CV inside search; on the held-out test, the **train-only tuned** model reached **98.84%** accuracy vs **97.67%** baseline.
3. **Final train+val refit** used more labeled data (no peeking at the test set) and achieved **100%** accuracy and **1.0** ROC-AUC on **this** test split, with no false positives or false negatives in the confusion matrix.
4. **Takeaway for comparison with other groups (RF, SVM):** Report **the same split policy and metric** (accuracy vs AUC vs F1). Tree models need no feature scaling here; the main story is **high separability** of Wisconsin features and **honest evaluation** on a held-out set, not a single headline metric in isolation.
