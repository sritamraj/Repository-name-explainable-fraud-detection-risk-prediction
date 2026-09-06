import joblib
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split

from .features import add_features, TARGET


def calculate_cost(
    y_true,
    probability,
    threshold,
    fp_cost,
    fn_cost,
):
    prediction = (
        probability >= threshold
    ).astype(int)

    fp = (
        (y_true == 0)
        & (prediction == 1)
    ).sum()

    fn = (
        (y_true == 1)
        & (prediction == 0)
    ).sum()

    cost = (
        fp_cost * fp
        + fn_cost * fn
    )

    return int(cost), int(fp), int(fn)


def main():

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(
        cfg["data"]["path"]
    )

    X_raw = df.drop(
        columns=TARGET
    )

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

    relative_test_size = (
        cfg["model"]["test_size"]
        / (
            cfg["model"]["test_size"]
            + cfg["model"]["validation_size"]
        )
    )

    Xv_raw, Xte_raw, yv, yte = train_test_split(
        Xtmp_raw,
        ytmp,
        test_size=relative_test_size,
        stratify=ytmp,
        random_state=cfg["random_state"],
    )

    amount_quantile = (
        Xtr_raw["transaction_amount"]
        .quantile(0.95)
    )

    Xv = add_features(
        Xv_raw,
        amount_quantile=amount_quantile,
    )

    model = joblib.load(
        "reports/fraud_model.joblib"
    )

    probability = model.predict_proba(
        Xv
    )[:, 1]

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

    fp_cost = cfg["costs"]["false_positive"]

    fn_costs = [
        50,
        100,
        200,
    ]

    results = []

    for fn_cost in fn_costs:

        best = None

        for threshold in thresholds:

            cost, fp, fn = calculate_cost(
                yv.to_numpy(),
                probability,
                threshold,
                fp_cost,
                fn_cost,
            )

            row = {
                "false_positive_cost": fp_cost,
                "false_negative_cost": fn_cost,
                "threshold": threshold,
                "validation_cost": cost,
                "false_positives": fp,
                "false_negatives": fn,
            }

            results.append(row)

            if (
                best is None
                or cost < best["validation_cost"]
            ):
                best = row

        print(
            "\n========================================"
        )

        print(
            f"FN COST = {fn_cost}"
        )

        print(
            "========================================"
        )

        print(
            f"Optimal threshold: "
            f"{best['threshold']:.2f}"
        )

        print(
            f"Minimum validation cost: "
            f"{best['validation_cost']:,}"
        )

        print(
            f"False positives: "
            f"{best['false_positives']:,}"
        )

        print(
            f"False negatives: "
            f"{best['false_negatives']:,}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "reports/cost_sensitivity.csv",
        index=False,
    )

    print(
        "\n========================================"
    )

    print(
        "FILES SAVED"
    )

    print(
        "========================================"
    )

    print(
        "reports/cost_sensitivity.csv"
    )


if __name__ == "__main__":
    main()