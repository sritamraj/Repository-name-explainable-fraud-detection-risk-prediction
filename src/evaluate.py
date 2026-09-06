from pathlib import Path
import joblib,pandas as pd,yaml,matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,brier_score_loss,confusion_matrix,classification_report
from .features import add_features,TARGET

def choose_threshold(y,p,fp_cost,fn_cost):
    _,_,ts=precision_recall_curve(y,p); best=None
    for t in ts:
        tn,fp,fn,tp=confusion_matrix(y,(p>=t).astype(int)).ravel(); cost=fp*fp_cost+fn*fn_cost
        if best is None or cost<best[0]: best=(cost,float(t),int(fp),int(fn))
    return best

def main():
    with open('config.yaml',encoding='utf-8') as f: cfg=yaml.safe_load(f)
    df=add_features(pd.read_csv(cfg['data']['path'])); model=joblib.load('reports/fraud_model.joblib'); X=df.drop(columns=TARGET); y=df[TARGET]; p=model.predict_proba(X)[:,1]
    print('PR-AUC:',round(average_precision_score(y,p),4)); print('ROC-AUC:',round(roc_auc_score(y,p),4)); print('Brier:',round(brier_score_loss(y,p),4)); print('Cost-optimal threshold:',choose_threshold(y,p,cfg['costs']['false_positive'],cfg['costs']['false_negative'])); print(classification_report(y,(p>=.5).astype(int),digits=4))
    precision,recall,_=precision_recall_curve(y,p); Path('reports').mkdir(exist_ok=True); plt.figure(figsize=(7,5)); plt.plot(recall,precision); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision–Recall Curve'); plt.tight_layout(); plt.savefig('reports/precision_recall_curve.png',dpi=160); plt.close()
if __name__=='__main__': main()
