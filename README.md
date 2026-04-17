# XGBoost — Breast Cancer Wisconsin (Diagnostic)

This project trains an **XGBoost** classifier on the Kaggle dataset **Breast Cancer Wisconsin (Diagnostic)** to predict whether a tumor is **malignant (M)** or **benign (B)**.

- **Dataset**: Kaggle — Breast Cancer Wisconsin Data (`data.csv`)
- **Task**: Binary classification (`diagnosis` column)

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

## Run

Open and run the notebook:

- `xgboost_breast_cancer.ipynb`

Or execute it headlessly:

```bash
jupyter nbconvert --to notebook --execute xgboost_breast_cancer.ipynb --output executed.ipynb
```

## Results (fill after running)

After **Run All** in the notebook, copy metrics here.

- **10-fold CV accuracy** (printed near the top of the workflow; headline number comparable to many Kaggle notebooks): mean __, std __
- **Baseline** (test): Accuracy __, F1 __, ROC-AUC __
- **Tuned, train split only** (test): Accuracy __, F1 __, ROC-AUC __
- **Final refit on train+val** (test; usually the best row): Accuracy __, F1 __, ROC-AUC __

The notebook tunes **`RandomizedSearchCV` with `scoring="accuracy"`**, refits the best hyperparameters on **train+validation** (more data than train-only), then reports metrics on the **held-out test** set once.

## Discussion (fill after running)

- Feature importance highlights the most predictive measurements (e.g., radius/texture/perimeter-related features tend to rank highly).
- In medical screening, **false negatives** (malignant predicted benign) are typically the highest-cost error; threshold selection can be adjusted to prioritize recall.
- Hyperparameter tuning improves generalization but should be evaluated strictly on a held-out test set to avoid optimistic bias.

