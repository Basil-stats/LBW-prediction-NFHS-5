# Predicting low birth weight from antenatal information using machine learning (NFHS-5, India)

## Contents

| File | Purpose |
|---|---|
| `codes.do` | Stata 17: builds the analytical sample from the NFHS-5 children's recode file (IAKR7EFL.DTA), codes all variables, produces Table 1 and the survey-weighted logistic regression (Table 2), and exports `data.csv` for Python |
| `LBW.py` | Python: trains and evaluates all models (Table 3, Table S2, Figs. 2 and S1–S4) |
| `requirements.txt` | Python package versions used |

## Data

The data are **not** included in this repository. NFHS-5 (2019–21) data are available free of charge from the DHS Program (https://dhsprogram.com/) after registration and approval. Download the India 2019–21 **Children's Recode** in Stata format (`IAKR7EFL.DTA`).

## How to reproduce

1. **Stata.** Edit the file paths at the top of `codes.do`, then run the whole file. It creates `data.csv`. Final analytical sample: 141,349 most recent births.
2. **Python 3.14.** Install the packages:
   ```
   pip install -r requirements.txt
   ```
3. Edit `DATA_PATH` and `OUT_DIR` at the top of each Python script, then run:
   ```
   python LBW.py
   ```

## Methods in brief

- **Models:** an antenatal model (Model A, 14 predictors available during pregnancy) and a full model (Model B, which adds post-delivery and survey-time variables).
- **Data split:** a single stratified 70/30 train–test split. The test set keeps the original outcome distribution and is used once.
- **Preprocessing and SMOTE:** both are steps inside an `imblearn` pipeline, so they are fitted only on the training part of each cross-validation fold.
- **Algorithms:** logistic regression, decision tree, random forest, XGBoost and a neural network, each trained with class weighting or with SMOTE.
- **Tuning and threshold:** hyperparameters are tuned by randomised search with stratified 5-fold cross-validation. The classification threshold (Youden's J) is chosen on out-of-fold training predictions.
- **Evaluation:** ROC-AUC and PR-AUC with bootstrap 95% CIs, sensitivity, specificity, PPV, NPV, F1, Brier score and calibration. Probabilities are recalibrated with Platt scaling.
- **Reporting:** follows TRIPOD+AI.

Random seed: 42.

## Contact

Basil Edolikkandy, International Institute for Population Sciences, Mumbai (eybasil@gmail.com)
