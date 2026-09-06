import pandas as pd
import pytest

from src.features import add_features
from src.predict import risk_decision, validate_transaction


VALID_TRANSACTION = {
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


def test_features():
    df = pd.DataFrame([VALID_TRANSACTION])

    result = add_features(
        df,
        amount_quantile=208.686,
    )

    assert "velocity_ratio" in result.columns
    assert "amount_per_account_day" in result.columns
    assert "device_account_age_ratio" in result.columns
    assert "high_velocity_flag" in result.columns
    assert "large_amount_flag" in result.columns


def test_decisions():
    assert risk_decision(0.10) == "approve"
    assert risk_decision(0.50) == "review"
    assert risk_decision(0.90) == "block"


def test_validation_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_transaction({"customer_age": 29})


def test_validation_accepts_complete_transaction():
    validate_transaction(VALID_TRANSACTION)