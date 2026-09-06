from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

from .features import add_features, TARGET


def main():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---------------------------------------------------------
    # 1. Load raw data
    # ---------------------------------------------------------
    df = pd.read_csv(cfg["data"]["path"])

    X_raw = df.drop(columns=TARGET)
    y = df[TARGET]

    # ---------------------------------------------------------
    # 2. Train / validation / test split
    # ---------------------------------------------------------
    Xtr_raw, Xtmp_raw, ytr, ytmp = train_test_split(
        X_raw,
        y,
        test_size=(
            cfg["model"]["test_size"]
            + cfg["model"]["validation_size"]
        ),
        stratify=y,
        random_state=cfg["random_state"],
    )

    rel = cfg["model"]["test_size"] / (
        cfg["model"]["test_size"]
        + cfg["model"]["validation_size"]
    )

    Xv_raw, Xte_raw, yv, yte = train_test_split(
        Xtmp_raw,
        ytmp,
        test_size=rel,
        stratify=ytmp,
        random_state=cfg["random_state"],
    )

    # ---------------------------------------------------------
    # 3. Leakage-safe feature engineering
    # ---------------------------------------------------------
    amount_quantile = Xtr_raw["transaction_amount"].quantile(0.95)

    Xtr = add_features(
        Xtr_raw,
        amount_quantile=amount_quantile,
    )

    Xv = add_features(
        Xv_raw,
        amount_quantile=amount_quantile,
    )

    Xte = add_features(
        Xte_raw,
        amount_quantile=amount_quantile,
    )

    # ---------------------------------------------------------
    # 4. Logistic Regression baseline
    # ---------------------------------------------------------
    base = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=cfg["random_state"],
                ),
            ),
        ]
    )

    base.fit(Xtr, ytr)

    # ---------------------------------------------------------
    # 5. XGBoost
    # ---------------------------------------------------------
    sp = (ytr == 0).sum() / (ytr == 1).sum()

    xgb_model = XGBClassifier(
        n_estimators=cfg["model"]["n_estimators"],
        max_depth=cfg["model"]["max_depth"],
        learning_rate=cfg["model"]["learning_rate"],
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=sp,
        eval_metric="aucpr",
        random_state=cfg["random_state"],
        n_jobs=4,
    )

    xgb_model.fit(Xtr, ytr)

    # ---------------------------------------------------------
    # 6. Probability calibration
    #
    # Calibration uses only the training set through
    # internal 5-fold cross-validation.
    #
    # Validation and test remain completely untouched.
    # ---------------------------------------------------------
    calibrated_model = CalibratedClassifierCV(
        estimator=XGBClassifier(
            n_estimators=cfg["model"]["n_estimators"],
            max_depth=cfg["model"]["max_depth"],
            learning_rate=cfg["model"]["learning_rate"],
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=sp,
            eval_metric="aucpr",
            random_state=cfg["random_state"],
            n_jobs=4,
        ),
        method="sigmoid",
        cv=5,
    )

    calibrated_model.fit(Xtr, ytr)

    # ---------------------------------------------------------
    # 7. Validation comparison
    # ---------------------------------------------------------
    for name, m in [
        ("logistic", base),
        ("xgboost_raw", xgb_model),
        ("xgboost_calibrated", calibrated_model),
    ]:
        p = m.predict_proba(Xv)[:, 1]

        print(
            name,
            "PR-AUC",
            round(average_precision_score(yv, p), 4),
            "ROC-AUC",
            round(roc_auc_score(yv, p), 4),
        )

    # ---------------------------------------------------------
    # 8. Save artifacts
    # ---------------------------------------------------------
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(
        xgb_model,
        "reports/fraud_model_raw.joblib",
    )

    joblib.dump(
        calibrated_model,
        "reports/fraud_model.joblib",
    )

    joblib.dump(
        base,
        "reports/logistic_baseline.joblib",
    )

    joblib.dump(
        {
            "feature_columns": list(Xtr.columns),
            "amount_quantile": float(amount_quantile),
            "calibration_method": "sigmoid",
            "calibration_cv": 5,
        },
        "reports/metadata.joblib",
    )


if __name__ == "__main__":
    main()