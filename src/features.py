import pandas as pd
TARGET='is_fraud'

def add_features(df):
    out=df.copy()
    out['amount_per_account_day']=out.transaction_amount/(out.account_age_days+1)
    out['velocity_ratio']=out.velocity_1h/(out.velocity_24h+1)
    out['device_account_age_ratio']=out.device_age_days/(out.account_age_days+1)
    out['high_velocity_flag']=(out.velocity_24h>=8).astype(int)
    out['large_amount_flag']=(out.transaction_amount>=out.transaction_amount.quantile(.95)).astype(int)
    return out
