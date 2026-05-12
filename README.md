# Predicting F1 Pit Stops — Stable High-Score Solution

This repository contains a stable machine learning pipeline for the Kaggle competition **Playground Series S6E5: Predicting F1 Pit Stops**.

The goal of the competition is to predict whether a Formula 1 driver will make a pit stop on the next lap. The target variable is:

```text
PitNextLap
````

The notebook focuses on building a reliable high-score solution with correct cross-validation, fold-safe feature engineering, original dataset augmentation, and model blending.

---

## Project Goals

The main goals of this notebook are:

* Build a stable end-to-end training pipeline.
* Avoid target leakage during validation.
* Use the original dataset safely.
* Train several strong gradient boosting models.
* Generate separate submissions for each model.
* Build final blended submissions using OOF-based weights.
* Keep the notebook reproducible and easy to run on Kaggle.

---

## Competition Task

This is a binary classification problem.

Given race, driver, tyre, lap, stint, position, and degradation-related features, the model predicts the probability that a pit stop will happen on the next lap.

The evaluation metric is **ROC AUC**, so the final submission must contain probabilities, not class labels.

Correct:

```python
model.predict_proba(X_test)[:, 1]
```

Incorrect:

```python
model.predict(X_test)
```

---

## Dataset

The notebook uses:

1. Competition train dataset
2. Competition test dataset
3. Optional original dataset

Default Kaggle paths:

```python
PATHS = {
    "train": "/kaggle/input/competitions/playground-series-s6e5/train.csv",
    "test": "/kaggle/input/competitions/playground-series-s6e5/test.csv",
    "original": "/kaggle/input/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction/f1_strategy_dataset_v4.csv",
}
```

If the dataset paths are different in your Kaggle environment, change only the `PATHS` block.

---

## Notebook Structure

The notebook is organized into the following stages:

1. Import libraries and define global settings
2. Load train, test, and original data
3. Clean raw features
4. Add feature engineering
5. Align original dataset columns with competition data
6. Add fold-safe target and frequency encoding
7. Apply model-specific preprocessing
8. Train CatBoost, XGBoost, and LightGBM with cross-validation
9. Save single-model submissions
10. Create linear and rank-based blended submissions

---

## Feature Engineering

The notebook includes several types of features.

### Basic cleaning

The pipeline removes or fixes problematic columns and values:

* Drops `Normalized_TyreLife` from the original dataset if present.
* Converts selected numerical columns to categorical type.
* Handles extreme values in lap time and degradation columns.

### Binned features

The notebook creates binned versions of important continuous variables:

* `TyreLife_bin`
* `RaceProgress_bin`
* `LapNumber_bin`

These features help tree-based models capture non-linear thresholds.

### Interaction features

The pipeline creates domain-inspired interactions such as:

* tyre life interactions
* race progress interactions
* lap number interactions
* degradation-related combinations
* categorical n-gram style features

These features are useful because pit stop decisions are strongly dependent on tyre age, race phase, compound, and degradation behavior.

---

## Target and Frequency Encoding

The notebook uses fold-safe target and frequency encoding.

For each fold:

* target statistics are calculated only on the training part of the fold;
* validation data is never used to calculate target encoding;
* unseen categories are filled with global target mean or zero frequency.

This prevents target leakage and makes the OOF validation more trustworthy.

The generated features include:

```text
TE_<column>
FE_<column>
```

Where:

* `TE_` means target-encoded feature
* `FE_` means frequency-encoded feature

---

## Original Dataset Usage

The original dataset is used only as additional training data.

Important rule:

```text
Original data is added only to the training part of each fold.
```

It is not added to validation. This keeps OOF validation honest.

The notebook also aligns the original dataset columns with the competition dataset:

```python
X_orig = X_orig.reindex(columns=X.columns)
X_test = X_test.reindex(columns=X.columns)
```

This prevents feature mismatch errors.

---

## Models

The notebook trains three gradient boosting models:

### CatBoost

CatBoost receives categorical features as strings and uses native categorical handling.

Main advantages:

* strong with categorical variables;
* stable on tabular data;
* handles category interactions well.

### XGBoost

XGBoost receives a fully numeric matrix after preprocessing.

Main advantages:

* strong general-purpose boosting model;
* often performs very well on Kaggle tabular competitions;
* good candidate for blending.

### LightGBM

LightGBM is used as the third model for diversity.

Main advantages:

* fast training;
* strong performance on tabular data;
* useful for ensemble diversity.

---

## Cross-Validation

The notebook uses stratified K-fold validation:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

For every model, the notebook stores:

* out-of-fold predictions;
* test predictions averaged across folds;
* fold-level ROC AUC scores.

This allows proper model comparison and safe blending.

---

## Blending

The notebook creates two final ensemble variants.

### Linear Blend

The linear blend searches for the best model weights using OOF predictions.

Example:

```text
0.50 * CatBoost + 0.30 * XGBoost + 0.20 * LightGBM
```

The best weights are selected by maximizing OOF ROC AUC.

### Rank Blend

The rank blend converts predictions into ranks before averaging.

This can be more stable for ROC AUC because AUC depends on ranking, not probability calibration.

The notebook saves both:

```text
submission_linear_blend.csv
submission_rank_blend.csv
```

By default, the final `submission.csv` is created from the rank blend.

---

## Output Files

After running the notebook, the following files are generated:

```text
submission_cat.csv
submission_xgb.csv
submission_lgb.csv
submission_linear_blend.csv
submission_rank_blend.csv
submission.csv
```

The file used for final Kaggle submission is:

```text
submission.csv
```

---

## How to Run

1. Open the notebook on Kaggle.
2. Attach the competition dataset.
3. Attach the original F1 strategy dataset if available.
4. Run all cells from top to bottom.
5. Submit `submission.csv`.

If the original dataset is not available, set:

```python
USE_ORIGINAL = False
```

If GPU errors occur, set:

```python
USE_GPU = False
```

To train only one model for faster testing:

```python
RUN_MODELS = ["cat"]
```

For the full final run:

```python
RUN_MODELS = ["cat", "xgb", "lgb"]
```

---

## Important Implementation Details

### No target leakage

Target encoding is calculated only inside each fold using training data.

### Safe original dataset augmentation

The original dataset is added only to the training part of each fold.

### Correct AUC submission format

The submission uses probabilities, not hard class predictions.

### Separate model predictions

Each model has its own OOF and test predictions. This prevents accidental mixing of arrays between models.

### Rank-based final blend

The final submission uses rank blending by default because it is often more stable for ROC AUC competitions.

---

## Further Improvements

Possible ways to improve the score:

1. Run multiple seeds and average predictions.
2. Tune model hyperparameters separately for CatBoost, XGBoost, and LightGBM.
3. Try more careful feature selection based on OOF performance.
4. Add more domain-specific F1 strategy features.
5. Try different target encoding smoothing values.
6. Test linear blend and rank blend separately on the public leaderboard.
7. Add another diverse model only if it improves OOF and public leaderboard score.
8. Use hill climbing or greedy ensemble selection on OOF predictions.
