import joblib,pandas as pd,shap
from .features import add_features

def explain_transaction(transaction,top_k=8):
    model=joblib.load('reports/fraud_model.joblib'); row=add_features(pd.DataFrame([transaction])); values=shap.TreeExplainer(model).shap_values(row)[0]
    return pd.DataFrame({'feature':row.columns,'value':row.iloc[0].values,'shap_value':values,'abs_shap':abs(values)}).sort_values('abs_shap',ascending=False).head(top_k)[['feature','value','shap_value']]
