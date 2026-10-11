# Predictive Analytics Assignment

## Contents
- Case study: 30-day patient readmission prediction
- `01_simple_linear_regression.py`
- `02_logistic_regression.py`
- `03_svm_decision_tree.py`
- `04_naive_bayes.py`
- `05_market_basket_analysis.py`
- `06_breast_cancer_prediction.R`
- `07_hr_demand_prediction.R`
- `outputs/` for actual execution logs/screenshots

## Python setup
Install Python 3.10+ and run:
```bash
python -m pip install -r requirements.txt
```
Run each Python program:
```bash
python 01_simple_linear_regression.py
python 02_logistic_regression.py
python 03_svm_decision_tree.py
python 04_naive_bayes.py
python 05_market_basket_analysis.py
```

## R setup
Install R and RStudio. Open and run the two `.R` files. The breast-cancer program installs `mlbench` if needed. Internet may be required for package installation.

## Important notes
- The included datasets are demonstration data or public teaching datasets, not Johns Hopkins Hospital patient records.
- The HR dataset is illustrative. Replace it with real historical monthly staffing data for a meaningful forecast.
- The breast cancer model is for coursework only and must not be used for clinical diagnosis.
- Save your own real execution output and screenshots in `outputs/` before submission.
- Do not claim the illustrative readmission chart values are actual hospital results.
