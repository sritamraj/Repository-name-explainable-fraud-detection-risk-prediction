import pandas as pd
import yaml


def main():
    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---------------------------------------------------------
    # 2. Load held-out test predictions
    # ---------------------------------------------------------
    predictions = pd.read_csv(
        "reports/test_predictions.csv"
    )

    required_columns = [
        "actual_fraud",
        "fraud_probability",
        "prediction_050",
        "prediction_cost_threshold",
        "transaction_amount",
        "velocity_1h",
        "velocity_24h",
        "distance_from_home_km",
        "failed_attempts",
        "device_age_days",
        "merchant_risk",
        "night",
        "international",
        "new_device",
        "account_age_days",
        "customer_age",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns in test_predictions.csv: "
            + ", ".join(missing_columns)
        )

    # ---------------------------------------------------------
    # 3. Use the exact prediction produced during evaluation
    # ---------------------------------------------------------
    predictions["is_fraud"] = predictions[
        "actual_fraud"
    ].astype(int)

    predictions["prediction"] = predictions[
        "prediction_cost_threshold"
    ].astype(int)

    predictions["predicted_probability"] = predictions[
        "fraud_probability"
    ]

    # ---------------------------------------------------------
    # 4. Error categories
    # ---------------------------------------------------------
    false_positives = predictions[
        (predictions["is_fraud"] == 0)
        & (predictions["prediction"] == 1)
    ].copy()

    false_negatives = predictions[
        (predictions["is_fraud"] == 1)
        & (predictions["prediction"] == 0)
    ].copy()

    true_positives = predictions[
        (predictions["is_fraud"] == 1)
        & (predictions["prediction"] == 1)
    ].copy()

    true_negatives = predictions[
        (predictions["is_fraud"] == 0)
        & (predictions["prediction"] == 0)
    ].copy()

    # ---------------------------------------------------------
    # 5. Overall error analysis
    # ---------------------------------------------------------
    print("\n========================================")
    print("ERROR ANALYSIS")
    print("========================================")

    print(
        f"Test transactions: {len(predictions):,}"
    )

    print(
        f"False positives: {len(false_positives):,}"
    )

    print(
        f"False negatives: {len(false_negatives):,}"
    )

    print(
        f"True positives: {len(true_positives):,}"
    )

    print(
        f"True negatives: {len(true_negatives):,}"
    )

    # ---------------------------------------------------------
    # 6. False-negative analysis
    # ---------------------------------------------------------
    print("\n========================================")
    print("FALSE NEGATIVE ANALYSIS")
    print("========================================")

    if len(false_negatives) > 0:
        print(
            "Average fraud transaction amount:",
            round(
                false_negatives[
                    "transaction_amount"
                ].mean(),
                2,
            ),
        )

        print(
            "Average 24h velocity:",
            round(
                false_negatives[
                    "velocity_24h"
                ].mean(),
                2,
            ),
        )

        print(
            "International rate:",
            round(
                false_negatives[
                    "international"
                ].mean(),
                3,
            ),
        )

        print(
            "New-device rate:",
            round(
                false_negatives[
                    "new_device"
                ].mean(),
                3,
            ),
        )

        print(
            "Average failed attempts:",
            round(
                false_negatives[
                    "failed_attempts"
                ].mean(),
                2,
            ),
        )

        print(
            "Average predicted fraud probability:",
            round(
                false_negatives[
                    "predicted_probability"
                ].mean(),
                4,
            ),
        )

    else:
        print("No false negatives found.")

    # ---------------------------------------------------------
    # 7. False-positive analysis
    # ---------------------------------------------------------
    print("\n========================================")
    print("FALSE POSITIVE ANALYSIS")
    print("========================================")

    if len(false_positives) > 0:
        print(
            "Average legitimate transaction amount:",
            round(
                false_positives[
                    "transaction_amount"
                ].mean(),
                2,
            ),
        )

        print(
            "Average 24h velocity:",
            round(
                false_positives[
                    "velocity_24h"
                ].mean(),
                2,
            ),
        )

        print(
            "International rate:",
            round(
                false_positives[
                    "international"
                ].mean(),
                3,
            ),
        )

        print(
            "New-device rate:",
            round(
                false_positives[
                    "new_device"
                ].mean(),
                3,
            ),
        )

        print(
            "Average failed attempts:",
            round(
                false_positives[
                    "failed_attempts"
                ].mean(),
                2,
            ),
        )

        print(
            "Average predicted fraud probability:",
            round(
                false_positives[
                    "predicted_probability"
                ].mean(),
                4,
            ),
        )

    else:
        print("No false positives found.")

    # ---------------------------------------------------------
    # 8. Risk segmentation
    # ---------------------------------------------------------
    print("\n========================================")
    print("FRAUD RATE BY PREDICTED-RISK BAND")
    print("========================================")

    predictions["risk_band"] = pd.cut(
        predictions["predicted_probability"],
        bins=[
            -float("inf"),
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            float("inf"),
        ],
        labels=[
            "<1%",
            "1-5%",
            "5-10%",
            "10-25%",
            "25-50%",
            ">=50%",
        ],
    )

    risk_summary = (
        predictions.groupby(
            "risk_band",
            observed=False,
        )
        .agg(
            transactions=("is_fraud", "size"),
            frauds=("is_fraud", "sum"),
            fraud_rate=("is_fraud", "mean"),
            avg_probability=(
                "predicted_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        risk_summary.to_string(index=False)
    )

    # ---------------------------------------------------------
    # 9. Save error-analysis datasets
    # ---------------------------------------------------------
    false_positives.to_csv(
        "reports/false_positives.csv",
        index=False,
    )

    false_negatives.to_csv(
        "reports/false_negatives.csv",
        index=False,
    )

    risk_summary.to_csv(
        "reports/risk_band_analysis.csv",
        index=False,
    )

    print("\n========================================")
    print("FILES SAVED")
    print("========================================")

    print(
        "reports/false_positives.csv"
    )

    print(
        "reports/false_negatives.csv"
    )

    print(
        "reports/risk_band_analysis.csv"
    )


if __name__ == "__main__":
    main()