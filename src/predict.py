import joblib,pandas as pd
from .features import add_features

def risk_decision(p,review=.35,block=.75): return 'block' if p>=block else ('review' if p>=review else 'approve')
def score_transaction(transaction,review=.35,block=.75):
    model=joblib.load('reports/fraud_model.joblib'); row=add_features(pd.DataFrame([transaction])); p=float(model.predict_proba(row)[0,1]); return {'fraud_probability':round(p,4),'risk_score':round(p*100,2),'decision':risk_decision(p,review,block)}
if __name__=='__main__':
    print(score_transaction({'customer_age':29,'account_age_days':45,'transaction_amount':2400,'velocity_1h':5,'velocity_24h':12,'distance_from_home_km':160,'failed_attempts':2,'device_age_days':3,'merchant_risk':.65,'night':1,'international':1,'new_device':1}))
