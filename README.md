# Explainable Fraud Detection & Risk Prediction Using Machine Learning

An internship-ready machine learning project for fraud probability estimation, operational risk scoring, cost-sensitive decisions, and transaction-level explanations.

## What this demonstrates
- Imbalanced classification
- Leakage-aware train/validation/test splitting
- Logistic Regression baseline
- XGBoost nonlinear model
- PR-AUC, ROC-AUC, Brier score and threshold analysis
- Business-cost-aware decision thresholds
- 0–100 risk scoring
- SHAP transaction explanations
- Reproducible configuration and tests

## Quick start (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_generation
python -m src.train
python -m src.evaluate
python -m src.predict
python -m pytest
```

## Pipeline
`transaction data → feature engineering → baseline/model → calibrated/operating probability → risk score → approve/review/block → explanation`

The included generator creates synthetic data for reproducible portfolio use. Replace it with a legally usable real dataset before making empirical claims.

## Strong interview questions
1. Why is accuracy misleading in fraud detection?
2. Why use PR-AUC?
3. How do false-positive and false-negative costs affect the threshold?
4. Why must resampling happen only inside the training process?
5. Why does probability calibration matter for risk scores?
6. How would you monitor concept drift?
7. How would you prevent data leakage from post-transaction features?
8. How would you serve this model online at low latency?

## Resume bullet
**Explainable Fraud Detection & Risk Prediction | Python, Scikit-learn, XGBoost, SHAP** — Built an end-to-end fraud-risk pipeline with imbalance-aware modeling, cost-sensitive threshold optimization, operational risk scoring, and transaction-level SHAP explanations; evaluated using PR-AUC and business-cost trade-offs rather than accuracy alone.

## Disclaimer
Educational portfolio project. Not suitable for real financial decisions without extensive validation, governance, privacy review, monitoring, and domain-specific testing.
