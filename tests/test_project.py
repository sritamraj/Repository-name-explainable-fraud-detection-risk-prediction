import pandas as pd
from src.features import add_features
from src.predict import risk_decision

def test_features():
    df=pd.DataFrame([{'customer_age':30,'account_age_days':100,'transaction_amount':100,'velocity_1h':1,'velocity_24h':3,'distance_from_home_km':2,'failed_attempts':0,'device_age_days':50,'merchant_risk':.1,'night':0,'international':0,'new_device':0}])
    assert 'velocity_ratio' in add_features(df).columns

def test_decisions():
    assert risk_decision(.1)=='approve'; assert risk_decision(.5)=='review'; assert risk_decision(.9)=='block'
