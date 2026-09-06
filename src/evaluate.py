from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)

from sklearn.model_selection import train_test_split

from .features import add_features, TARGET


def choose_threshold(y, p, fp_cost, fn_cost):
    """Choose threshold using validation data only."""
    _, _, thresholds = precision_recall_curve(y, p)

    best = None

    for threshold in thresholds:
        predictions = (p >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        ).ravel()

        cost = fp * fp_cost + fn * fn_cost

        if best is None or cost < best["cost"]:
            best = {
                "cost": int(cost),
                "threshold": float(threshold),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "tn": int(tn),
            }

    return best


def calculate_cost(y, p, threshold, fp_cost, fn_cost):
    """Evaluate a fixed threshold on a dataset."""
    predictions = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    cost = fp * fp_cost + fn * fn_cost

    return {
        "cost": int(cost),
        "threshold": float(threshold),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
    }


def main():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # =========================================================
    # 1. Load raw data
    # =========================================================
    df = pd.read_csv(cfg["data"]["path"])

    X_raw = df.drop(columns=TARGET)
    y = df[TARGET]

    # =========================================================
    # 2. Reproduce EXACT train / validation / test split
    # =========================================================
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

    relative_test_size = cfg["model"]["test_size"] / (
        cfg["model"]["test_size"]
        + cfg["model"]["validation_size"]
    )

    Xv_raw, Xte_raw, yv, yte = train_test_split(
        Xtmp_raw,
        ytmp,
        test_size=relative_test_size,
        stratify=ytmp,
        random_state=cfg["random_state"],
    )

    # =========================================================
    # 3. Load training-only feature statistics
    # =========================================================
    metadata = joblib.load("reports/metadata.joblib")

    amount_quantile = metadata["amount_quantile"]

    Xv = add_features(
        Xv_raw,
        amount_quantile=amount_quantile,
    )

    Xte = add_features(
        Xte_raw,
        amount_quantile=amount_quantile,
    )

    # =========================================================
    # 4. Load trained model
    # =========================================================
    model = joblib.load("reports/fraud_model.joblib")

    # =========================================================
    # 5. Validation predictions
    # =========================================================
    p_val = model.predict_proba(Xv)[:, 1]

    validation_pr_auc = average_precision_score(
        yv,
        p_val,
    )

    validation_roc_auc = roc_auc_score(
        yv,
        p_val,
    )

    # =========================================================
    # 6. Select cost-sensitive threshold ON VALIDATION ONLY
    # =========================================================
    selected_threshold = choose_threshold(
        yv,
        p_val,
        cfg["costs"]["false_positive"],
        cfg["costs"]["false_negative"],
    )

    threshold = selected_threshold["threshold"]

    # =========================================================
    # 7. Final test predictions
    # =========================================================
    p_test = model.predict_proba(Xte)[:, 1]

    test_pr_auc = average_precision_score(
        yte,
        p_test,
    )

    test_roc_auc = roc_auc_score(
        yte,
        p_test,
    )

    test_brier = brier_score_loss(
        yte,
        p_test,
    )

    # =========================================================
    # 8. Apply validation-selected threshold to TEST
    # =========================================================
    test_cost = calculate_cost(
        yte,
        p_test,
        threshold,
        cfg["costs"]["false_positive"],
        cfg["costs"]["false_negative"],
    )

    # =========================================================
    # 9. Print validation results
    # =========================================================
    print("\n========================================")
    print("VALIDATION RESULTS")
    print("========================================")

    print(
        "Validation samples:",
        len(yv),
    )

    print(
        "Validation fraud rate:",
        round(yv.mean(), 4),
    )

    print(
        "Validation PR-AUC:",
        round(validation_pr_auc, 4),
    )

    print(
        "Validation ROC-AUC:",
        round(validation_roc_auc, 4),
    )

    # =========================================================
    # 10. Print threshold selection
    # =========================================================
    print("\n========================================")
    print("THRESHOLD SELECTION — VALIDATION ONLY")
    print("========================================")

    print(
        "Selected threshold:",
        round(threshold, 4),
    )

    print(
        "Validation cost:",
        selected_threshold["cost"],
    )

    print(
        "Validation false positives:",
        selected_threshold["fp"],
    )

    print(
        "Validation false negatives:",
        selected_threshold["fn"],
    )

    print(
        "Validation true positives:",
        selected_threshold["tp"],
    )

    print(
        "Validation true negatives:",
        selected_threshold["tn"],
    )

    # =========================================================
    # 11. Print FINAL held-out test results
    # =========================================================
    print("\n========================================")
    print("FINAL HELD-OUT TEST RESULTS")
    print("========================================")

    print(
        "Test samples:",
        len(yte),
    )

    print(
        "Test fraud rate:",
        round(yte.mean(), 4),
    )

    print(
        "Test PR-AUC:",
        round(test_pr_auc, 4),
    )

    print(
        "Test ROC-AUC:",
        round(test_roc_auc, 4),
    )

    print(
        "Test Brier score:",
        round(test_brier, 4),
    )

    # =========================================================
    # 12. Test cost at validation-selected threshold
    # =========================================================
    print("\n========================================")
    print("TEST COST AT VALIDATION THRESHOLD")
    print("========================================")

    print(
        "Threshold:",
        round(threshold, 4),
    )

    print(
        "Total cost:",
        test_cost["cost"],
    )

    print(
        "False positives:",
        test_cost["fp"],
    )

    print(
        "False negatives:",
        test_cost["fn"],
    )

    print(
        "True positives:",
        test_cost["tp"],
    )

    print(
        "True negatives:",
        test_cost["tn"],
    )

    # =========================================================
    # 13. Classification report at validation threshold
    # =========================================================
    test_predictions = (
        p_test >= threshold
    ).astype(int)

    print("\n========================================")
    print("CLASSIFICATION REPORT")
    print(
        f"TEST @ VALIDATION THRESHOLD {threshold:.4f}"
    )
    print("========================================")

    print(
        classification_report(
            yte,
            test_predictions,
            digits=4,
            zero_division=0,
        )
    )

    # =========================================================
    # 14. Classification report at standard 0.50
    # =========================================================
    predictions_050 = (
        p_test >= 0.50
    ).astype(int)

    print("\n========================================")
    print("CLASSIFICATION REPORT")
    print("TEST @ STANDARD THRESHOLD 0.5000")
    print("========================================")

    print(
        classification_report(
            yte,
            predictions_050,
            digits=4,
            zero_division=0,
        )
    )

    # =========================================================
    # 15. Precision-Recall curve
    # =========================================================
    precision, recall, _ = precision_recall_curve(
        yte,
        p_test,
    )

    Path("reports").mkdir(exist_ok=True)

    plt.figure(figsize=(7, 5))

    plt.plot(
        recall,
        precision,
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "Precision-Recall Curve — Held-Out Test Set"
    )

    plt.tight_layout()

    plt.savefig(
        "reports/precision_recall_curve.png",
        dpi=160,
    )

    plt.close()

    # =========================================================
    # 16. Save test predictions
    # =========================================================
    results = Xte_raw.copy()

    results["actual_fraud"] = yte.to_numpy()

    results["fraud_probability"] = p_test

    results["prediction_050"] = predictions_050

    results["prediction_cost_threshold"] = (
        test_predictions
    )

    results.to_csv(
        "reports/test_predictions.csv",
        index=False,
    )

    print("\n========================================")
    print("FILES SAVED")
    print("========================================")

    print(
        "reports/precision_recall_curve.png"
    )

    print(
        "reports/test_predictions.csv"
    )


if __name__ == "__main__":
    main()