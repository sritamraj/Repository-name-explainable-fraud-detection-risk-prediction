from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def make_dataset(n_samples=30000, fraud_rate=0.035, random_state=42):
    rng=np.random.default_rng(random_state)
    customer_age=rng.normal(36,11,n_samples).clip(18,80)
    account_age_days=rng.gamma(3.5,260,n_samples).clip(1,5000)
    transaction_amount=rng.lognormal(3.7,1.0,n_samples).clip(1,10000)
    velocity_1h=rng.poisson(1.7,n_samples)
    velocity_24h=velocity_1h+rng.poisson(3.0,n_samples)
    distance_from_home_km=rng.exponential(18,n_samples).clip(0,500)
    failed_attempts=rng.poisson(.25,n_samples)
    device_age_days=rng.gamma(4,100,n_samples).clip(.1,2500)
    merchant_risk=rng.beta(2,8,n_samples)
    night=rng.binomial(1,.22,n_samples)
    international=rng.binomial(1,.12,n_samples)
    new_device=rng.binomial(1,.15,n_samples)
    amount_z=np.abs((transaction_amount-np.median(transaction_amount))/(np.std(transaction_amount)+1e-8))
    logit=(-5+1.8*merchant_risk+.65*np.log1p(transaction_amount)+.28*velocity_24h+.45*failed_attempts+.012*distance_from_home_km+.9*international+1.1*new_device+.7*night+.45*(international*new_device)+.22*amount_z-.00015*account_age_days-.00035*device_age_days)
    p=1/(1+np.exp(-logit))
    shift=np.log(fraud_rate/(1-fraud_rate))-np.log(np.mean(p)/(1-np.mean(p)))
    p=1/(1+np.exp(-(logit+shift)))
    y=rng.binomial(1,p)
    return pd.DataFrame({'customer_age':customer_age.round(1),'account_age_days':account_age_days.round(),'transaction_amount':transaction_amount.round(2),'velocity_1h':velocity_1h,'velocity_24h':velocity_24h,'distance_from_home_km':distance_from_home_km.round(2),'failed_attempts':failed_attempts,'device_age_days':device_age_days.round(),'merchant_risk':merchant_risk.round(4),'night':night,'international':international,'new_device':new_device,'is_fraud':y})

def main():
    with open('config.yaml',encoding='utf-8') as f: cfg=yaml.safe_load(f)
    df=make_dataset(cfg['data']['n_samples'],cfg['data']['fraud_rate'],cfg['random_state'])
    path=Path(cfg['data']['path']); path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False)
    print(f'Saved {len(df):,} rows to {path}; fraud rate={df.is_fraud.mean():.3%}')
if __name__=='__main__': main()
