import joblib
import pandas as pd
import yaml

from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from .features import add_features, TARGET


def evaluate_threshold(y, probability, threshold, fp_cost, fn_cost):
    prediction = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        prediction,
        labels=[0, 1],
    ).ravel()

    cost = (
        fp_cost * fp
        + fn_cost * fn
    )

    return {
        "threshold": threshold,
        "precision": precision_score(
            y,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            prediction,
            zero_division=0,
        ),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
        "total_cost": cost,
    }


def main():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(
        cfg["data"]["path"]
    )

    X_raw = df.drop(columns=TARGET)
    y = df[TARGET]

    # Reproduce exact train/validation/test split.
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

    model = joblib.load(
        "reports/fraud_model.joblib"
    )

    validation_probability = model.predict_proba(
        Xv
    )[:, 1]

    test_probability = model.predict_proba(
        Xte
    )[:, 1]

    fp_cost = cfg["costs"]["false_positive"]
    fn_cost = cfg["costs"]["false_negative"]

    thresholds = [
        0.01,
        0.02,
        0.03,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.50,
    ]

    validation_results = []

    for threshold in thresholds:
        result = evaluate_threshold(
            yv.to_numpy(),
            validation_probability,
            threshold,
            fp_cost,
            fn_cost,
        )

        validation_results.append(result)

    validation_df = pd.DataFrame(
        validation_results
    )

    selected_row = validation_df.loc[
        validation_df["total_cost"].idxmin()
    ]

    selected_threshold = float(
        selected_row["threshold"]
    )

    test_result = evaluate_threshold(
        yte.to_numpy(),
        test_probability,
        selected_threshold,
        fp_cost,
        fn_cost,
    )

    print("\n========================================")
    print("THRESHOLD / COST SENSITIVITY")
    print("========================================")

    print("\nVALIDATION RESULTS")

    print(
        validation_df.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.3%}".format,
                "recall": "{:.3%}".format,
            },
        )
    )

    print("\n========================================")
    print("SELECTED THRESHOLD")
    print("========================================")

    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Validation cost: "
        f"{int(selected_row['total_cost']):,}"
    )

    print("\n========================================")
    print("HELD-OUT TEST RESULT")
    print("========================================")

    print(
        f"Test threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Test precision: "
        f"{test_result['precision']:.3%}"
    )

    print(
        f"Test recall: "
        f"{test_result['recall']:.3%}"
    )

    print(
        f"Test false positives: "
        f"{test_result['false_positives']:,}"
    )

    print(
        f"Test false negatives: "
        f"{test_result['false_negatives']:,}"
    )

    print(
        f"Test total cost: "
        f"{test_result['total_cost']:,}"
    )

    validation_df.to_csv(
        "reports/threshold_analysis.csv",
        index=False,
    )

    print("\nFILES SAVED")
    print(
        "reports/threshold_analysis.csv"
    )


if __name__ == "__main__":
    main()