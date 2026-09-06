import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from .features import add_features, TARGET


def main():
    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---------------------------------------------------------
    # 2. Load data
    # ---------------------------------------------------------
    df = pd.read_csv(
        cfg["data"]["path"]
    )

    X_raw = df.drop(columns=TARGET)
    y = df[TARGET]

    # ---------------------------------------------------------
    # 3. Reproduce exact train/validation/test split
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

    # ---------------------------------------------------------
    # 4. Leakage-safe feature engineering
    # ---------------------------------------------------------
    amount_quantile = Xtr_raw[
        "transaction_amount"
    ].quantile(0.95)

    Xv = add_features(
        Xv_raw,
        amount_quantile=amount_quantile,
    )

    Xte = add_features(
        Xte_raw,
        amount_quantile=amount_quantile,
    )

    # ---------------------------------------------------------
    # 5. Load models
    # ---------------------------------------------------------
    raw_model = joblib.load(
        "reports/fraud_model_raw.joblib"
    )

    calibrated_model = joblib.load(
        "reports/fraud_model.joblib"
    )

    # ---------------------------------------------------------
    # 6. Generate validation predictions
    # ---------------------------------------------------------
    raw_val_probability = raw_model.predict_proba(
        Xv
    )[:, 1]

    calibrated_val_probability = calibrated_model.predict_proba(
        Xv
    )[:, 1]

    # ---------------------------------------------------------
    # 7. Generate test predictions
    # ---------------------------------------------------------
    raw_test_probability = raw_model.predict_proba(
        Xte
    )[:, 1]

    calibrated_test_probability = calibrated_model.predict_proba(
        Xte
    )[:, 1]

    # ---------------------------------------------------------
    # 8. Brier scores
    # ---------------------------------------------------------
    raw_val_brier = brier_score_loss(
        yv,
        raw_val_probability,
    )

    calibrated_val_brier = brier_score_loss(
        yv,
        calibrated_val_probability,
    )

    raw_test_brier = brier_score_loss(
        yte,
        raw_test_probability,
    )

    calibrated_test_brier = brier_score_loss(
        yte,
        calibrated_test_probability,
    )

    print("\n========================================")
    print("CALIBRATION ANALYSIS")
    print("========================================")

    print("\nVALIDATION BRIER SCORE")
    print(
        f"Raw XGBoost:        {raw_val_brier:.4f}"
    )
    print(
        f"Calibrated XGBoost: {calibrated_val_brier:.4f}"
    )

    print("\nHELD-OUT TEST BRIER SCORE")
    print(
        f"Raw XGBoost:        {raw_test_brier:.4f}"
    )
    print(
        f"Calibrated XGBoost: {calibrated_test_brier:.4f}"
    )

    # ---------------------------------------------------------
    # 9. Calibration curves
    # ---------------------------------------------------------
    raw_fraction_pos, raw_mean_predicted = calibration_curve(
        yte,
        raw_test_probability,
        n_bins=10,
        strategy="quantile",
    )

    calibrated_fraction_pos, calibrated_mean_predicted = calibration_curve(
        yte,
        calibrated_test_probability,
        n_bins=10,
        strategy="quantile",
    )

    # ---------------------------------------------------------
    # 10. Save calibration plot
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        raw_mean_predicted,
        raw_fraction_pos,
        marker="o",
        label="Raw XGBoost",
    )

    plt.plot(
        calibrated_mean_predicted,
        calibrated_fraction_pos,
        marker="o",
        label="Calibrated XGBoost",
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraud frequency")
    plt.title("Fraud Probability Calibration")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "reports/calibration_curve.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------
    # 11. Save numerical calibration results
    # ---------------------------------------------------------
    calibration_results = pd.DataFrame(
        {
            "raw_predicted_probability":
                raw_mean_predicted,
            "raw_observed_frequency":
                raw_fraction_pos,
            "calibrated_predicted_probability":
                calibrated_mean_predicted,
            "calibrated_observed_frequency":
                calibrated_fraction_pos,
        }
    )

    calibration_results.to_csv(
        "reports/calibration_results.csv",
        index=False,
    )

    print("\nFILES SAVED")
    print("reports/calibration_curve.png")
    print("reports/calibration_results.csv")


if __name__ == "__main__":
    main()