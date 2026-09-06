import joblib
import pandas as pd
import yaml

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


def evaluate_segment(name, mask, y, probability, threshold):
    if mask.sum() == 0:
        return None

    y_segment = y[mask]
    p_segment = probability[mask]
    pred_segment = (p_segment >= threshold).astype(int)

    return {
        "segment": name,
        "samples": int(mask.sum()),
        "fraud_rate": float(y_segment.mean()),
        "pr_auc": float(
            average_precision_score(y_segment, p_segment)
        ),
        "roc_auc": float(
            roc_auc_score(y_segment, p_segment)
        ),
        "precision": float(
            precision_score(
                y_segment,
                pred_segment,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_segment,
                pred_segment,
                zero_division=0,
            )
        ),
    }


def main():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(
        cfg["data"]["path"]
    )

    predictions = pd.read_csv(
        "reports/test_predictions.csv"
    )

    model = joblib.load(
        "reports/fraud_model.joblib"
    )

    metadata = joblib.load(
        "reports/metadata.joblib"
    )

    # Reconstruct test features using the training-only
    # amount threshold stored in metadata.
    feature_columns = metadata["feature_columns"]
    amount_quantile = metadata["amount_quantile"]

    from .features import add_features

    X_raw = df.drop(columns=["is_fraud"])

    X = add_features(
        X_raw,
        amount_quantile=amount_quantile,
    )

    # The evaluation script already generated the exact
    # test predictions. Use those rows through the saved
    # prediction file instead of creating a new split.
    y = predictions["actual_fraud"].astype(int)
    probability = predictions["fraud_probability"]

    threshold = 0.05

    # Align segment features with the saved test predictions.
    #
    # The evaluation file contains the original test rows
    # in the same order as the test split. We reproduce the
    # exact split used by train/evaluate.
    from sklearn.model_selection import train_test_split

    Xtr_raw, Xtmp_raw, ytr, ytmp = train_test_split(
        X_raw,
        df["is_fraud"],
        test_size=(
            cfg["model"]["test_size"]
            + cfg["model"]["validation_size"]
        ),
        stratify=df["is_fraud"],
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

    Xte = Xte_raw.reset_index(drop=True)
    yte = yte.reset_index(drop=True)

    # Safety check: saved predictions must match test labels.
    if not yte.equals(y.reset_index(drop=True)):
        raise ValueError(
            "Saved predictions do not match the reproduced test split."
        )

    # Segment definitions
    segments = {
        "all_test": pd.Series(
            True,
            index=Xte.index,
        ),

        "domestic": Xte["international"] == 0,

        "international": Xte["international"] == 1,

        "existing_device": Xte["new_device"] == 0,

        "new_device": Xte["new_device"] == 1,

        "low_amount_<100": Xte["transaction_amount"] < 100,

        "high_amount_>=100": Xte["transaction_amount"] >= 100,

        "low_velocity_<5": Xte["velocity_24h"] < 5,

        "high_velocity_>=5": Xte["velocity_24h"] >= 5,

        "low_merchant_risk_<0.25": Xte["merchant_risk"] < 0.25,

        "high_merchant_risk_>=0.25": (
            Xte["merchant_risk"] >= 0.25
        ),
    }

    results = []

    for name, mask in segments.items():
        result = evaluate_segment(
            name=name,
            mask=mask.to_numpy(),
            y=yte.to_numpy(),
            probability=probability.to_numpy(),
            threshold=threshold,
        )

        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(results)

    print("\n========================================")
    print("ROBUSTNESS ANALYSIS")
    print("========================================")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "fraud_rate": "{:.3%}".format,
                "pr_auc": "{:.4f}".format,
                "roc_auc": "{:.4f}".format,
                "precision": "{:.3%}".format,
                "recall": "{:.3%}".format,
            },
        )
    )

    results_df.to_csv(
        "reports/robustness_analysis.csv",
        index=False,
    )

    print("\n========================================")
    print("FILES SAVED")
    print("========================================")
    print("reports/robustness_analysis.csv")


if __name__ == "__main__":
    main()