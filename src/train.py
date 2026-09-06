from pathlib import Path
import joblib,pandas as pd,yaml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from xgboost import XGBClassifier
from .features import add_features,TARGET

def main():
    with open('config.yaml',encoding='utf-8') as f: cfg=yaml.safe_load(f)
    df=add_features(pd.read_csv(cfg['data']['path'])); X=df.drop(columns=TARGET); y=df[TARGET]
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=cfg['model']['test_size']+cfg['model']['validation_size'],stratify=y,random_state=cfg['random_state'])
    rel=cfg['model']['test_size']/(cfg['model']['test_size']+cfg['model']['validation_size'])
    Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=rel,stratify=ytmp,random_state=cfg['random_state'])
    base=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(class_weight='balanced',max_iter=2000,random_state=cfg['random_state']))])
    base.fit(Xtr,ytr)
    sp=(ytr==0).sum()/(ytr==1).sum()
    model=XGBClassifier(n_estimators=cfg['model']['n_estimators'],max_depth=cfg['model']['max_depth'],learning_rate=cfg['model']['learning_rate'],subsample=.85,colsample_bytree=.85,scale_pos_weight=sp,eval_metric='aucpr',random_state=cfg['random_state'],n_jobs=4)
    model.fit(Xtr,ytr)
    for name,m in [('logistic',base),('xgboost',model)]:
        p=m.predict_proba(Xv)[:,1]; print(name,'PR-AUC',round(average_precision_score(yv,p),4),'ROC-AUC',round(roc_auc_score(yv,p),4))
    Path('reports').mkdir(exist_ok=True); joblib.dump(model,'reports/fraud_model.joblib'); joblib.dump(base,'reports/logistic_baseline.joblib'); joblib.dump({'feature_columns':list(X.columns)},'reports/metadata.joblib')
if __name__=='__main__': main()
