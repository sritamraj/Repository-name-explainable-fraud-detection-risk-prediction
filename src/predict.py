import joblib
import pandas as pd

from .features import add_features


MODEL_PATH = "reports/fraud_model.joblib"
METADATA_PATH = "reports/metadata.joblib"


REQUIRED_COLUMNS = [
    "customer_age",
    "account_age_days",
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
]


def risk_decision(probability, review=0.35, block=0.75):
    """Map fraud probability to an operational risk decision."""
    if probability >= block:
        return "block"
    if probability >= review:
        return "review"
    return "approve"


def validate_transaction(transaction):
    """Validate that all model input fields are present."""
    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in transaction
    ]

    if missing:
        raise ValueError(
            f"Missing required transaction fields: {missing}"
        )


def score_transaction(transaction, review=0.35, block=0.75):
    """Score one transaction using the trained calibrated model."""

    validate_transaction(transaction)

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    amount_quantile = metadata["amount_quantile"]
    feature_columns = metadata["feature_columns"]

    row = pd.DataFrame([transaction])
    row = add_features(
        row,
        amount_quantile=amount_quantile,
    )

    row = row[feature_columns]

    probability = float(
        model.predict_proba(row)[0, 1]
    )

    return {
        "fraud_probability": round(probability, 4),
        "risk_score": round(probability * 100, 2),
        "decision": risk_decision(
            probability,
            review=review,
            block=block,
        ),
    }


if __name__ == "__main__":
    example_transaction = {
        "customer_age": 29,
        "account_age_days": 45,
        "transaction_amount": 2400,
        "velocity_1h": 5,
        "velocity_24h": 12,
        "distance_from_home_km": 160,
        "failed_attempts": 2,
        "device_age_days": 3,
        "merchant_risk": 0.65,
        "night": 1,
        "international": 1,
        "new_device": 1,
    }

    result = score_transaction(example_transaction)

    print("Fraud Risk Prediction")
    print("----------------------")
    print(f"Fraud probability : {result['fraud_probability']}")
    print(f"Risk score        : {result['risk_score']}/100")
    print(f"Decision           : {result['decision']}")