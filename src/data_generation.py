from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def make_dataset(
    n_samples=30000,
    fraud_rate=0.035,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    # ---------------------------------------------------------
    # Customer / account characteristics
    # ---------------------------------------------------------
    customer_age = rng.normal(
        36,
        11,
        n_samples,
    ).clip(18, 80)

    account_age_days = rng.gamma(
        3.5,
        260,
        n_samples,
    ).clip(1, 5000)

    device_age_days = rng.gamma(
        4,
        100,
        n_samples,
    ).clip(0.1, 2500)

    # ---------------------------------------------------------
    # Transaction characteristics
    # ---------------------------------------------------------
    transaction_amount = rng.lognormal(
        3.7,
        1.0,
        n_samples,
    ).clip(1, 10000)

    velocity_1h = rng.poisson(
        1.7,
        n_samples,
    )

    velocity_24h = (
        velocity_1h
        + rng.poisson(3.0, n_samples)
    )

    distance_from_home_km = rng.exponential(
        18,
        n_samples,
    ).clip(0, 500)

    failed_attempts = rng.poisson(
        0.25,
        n_samples,
    )

    merchant_risk = rng.beta(
        2,
        8,
        n_samples,
    )

    night = rng.binomial(
        1,
        0.22,
        n_samples,
    )

    international = rng.binomial(
        1,
        0.12,
        n_samples,
    )

    new_device = rng.binomial(
        1,
        0.15,
        n_samples,
    )

    # ---------------------------------------------------------
    # Behavioral / nonlinear signals
    # ---------------------------------------------------------

    # Transaction amount relative to typical account behavior.
    amount_excess = np.maximum(
        transaction_amount - 150,
        0,
    ) / 150

    # Very high transaction velocity.
    velocity_burst = np.maximum(
        velocity_1h - 3,
        0,
    ) ** 2

    # Suspicious geographic behavior.
    distance_anomaly = np.maximum(
        distance_from_home_km - 50,
        0,
    ) / 50

    # Young accounts are more vulnerable to certain patterns.
    young_account = (
        account_age_days < 180
    ).astype(int)

    # Very new devices are more suspicious.
    very_new_device = (
        device_age_days < 30
    ).astype(int)

    # ---------------------------------------------------------
    # Fraud interaction effects
    # ---------------------------------------------------------

    international_new_device = (
        international * new_device
    )

    velocity_failed_attempts = (
        velocity_1h * failed_attempts
    )

    amount_velocity_interaction = (
        amount_excess * np.maximum(
            velocity_1h - 1,
            0,
        )
    )

    distance_international = (
        distance_anomaly * international
    )

    young_account_new_device = (
        young_account * new_device
    )

    merchant_amount_interaction = (
        merchant_risk
        * np.log1p(transaction_amount)
    )

    # ---------------------------------------------------------
    # Nonlinear fraud score
    # ---------------------------------------------------------
    raw_score = (
        -3.8
        + 1.2 * merchant_risk
        + 0.45 * np.log1p(transaction_amount)
        + 0.18 * velocity_24h
        + 0.35 * failed_attempts
        + 0.006 * distance_from_home_km
        + 0.65 * international
        + 0.85 * new_device
        + 0.45 * night
        + 1.80 * international_new_device
        + 0.55 * velocity_failed_attempts
        + 0.32 * amount_velocity_interaction
        + 0.70 * distance_international
        + 0.65 * young_account_new_device
        + 0.45 * merchant_amount_interaction
        + 0.80 * very_new_device
        + 0.12 * velocity_burst
        + 0.20 * distance_anomaly
        - 0.00012 * account_age_days
        - 0.00025 * device_age_days
    )

    # ---------------------------------------------------------
    # Convert score to probability
    # ---------------------------------------------------------
    raw_probability = 1 / (
        1 + np.exp(-raw_score)
    )

    # Calibrate intercept by binary search so the
    # expected fraud prevalence approximately matches
    # the requested fraud_rate.
    low = -10.0
    high = 10.0

    for _ in range(60):
        shift = (low + high) / 2

        probability = 1 / (
            1
            + np.exp(
                -(raw_score + shift)
            )
        )

        if probability.mean() > fraud_rate:
            high = shift
        else:
            low = shift

    final_shift = (low + high) / 2

    probability = 1 / (
        1
        + np.exp(
            -(raw_score + final_shift)
        )
    )

    # ---------------------------------------------------------
    # Sample fraud labels
    # ---------------------------------------------------------
    is_fraud = rng.binomial(
        1,
        probability,
    )

    # ---------------------------------------------------------
    # Build dataset
    # ---------------------------------------------------------
    return pd.DataFrame(
        {
            "customer_age": customer_age.round(1),
            "account_age_days": account_age_days.round(),
            "transaction_amount": transaction_amount.round(2),
            "velocity_1h": velocity_1h,
            "velocity_24h": velocity_24h,
            "distance_from_home_km": distance_from_home_km.round(2),
            "failed_attempts": failed_attempts,
            "device_age_days": device_age_days.round(),
            "merchant_risk": merchant_risk.round(4),
            "night": night,
            "international": international,
            "new_device": new_device,
            "is_fraud": is_fraud,
        }
    )


def main():
    with open(
        "config.yaml",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    df = make_dataset(
        n_samples=cfg["data"]["n_samples"],
        fraud_rate=cfg["data"]["fraud_rate"],
        random_state=cfg["random_state"],
    )

    path = Path(
        cfg["data"]["path"]
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved {len(df):,} rows to {path}; "
        f"fraud rate={df.is_fraud.mean():.3%}"
    )


if __name__ == "__main__":
    main()