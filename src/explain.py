import joblib
import pandas as pd
import shap

from .features import add_features


def explain_transaction(transaction, top_k=8):
    """
    Generate SHAP reason codes for a single transaction.

    SHAP explains the underlying XGBoost risk score.
    The calibrated model remains responsible for the final
    probability used by the prediction pipeline.
    """

    model = joblib.load("reports/fraud_model_raw.joblib")
    metadata = joblib.load("reports/metadata.joblib")

    amount_quantile = metadata["amount_quantile"]

    row = add_features(
        pd.DataFrame([transaction]),
        amount_quantile=amount_quantile,
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(row)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    values = shap_values[0]

    result = pd.DataFrame(
        {
            "feature": row.columns,
            "value": row.iloc[0].values,
            "shap_value": values,
        }
    )

    result["abs_shap"] = result["shap_value"].abs()

    return (
        result
        .sort_values("abs_shap", ascending=False)
        .head(top_k)
        [["feature", "value", "shap_value"]]
    )