"""Exp2: final F0.5 vs candidate budget K (per source) for the learned blocking ranker H.
Stage-1 (Exp1 LightGBM, unchanged) was trained/cross-fitted ONCE on the top-30-per-source candidates (oof_scores_H30.parquet); each K evaluates the subset rank_H<=K
with per-K cross-fitted thresholds (same protocol as Exp1). Reports overall / India / US F0.5, micro P/R, singleton acc, candidate counts."""
import sys,json,numpy as np,pandas as pd
sys.path.insert(0,'pilot')
from metric import f05_from_pred,f05_per_query,query_counts
OUT='pilot/exp2/'; BENCH='pilot/exp1/'; G=np.round(np.arange(0.30,0.9501,0.025),3)
S1=pd.read_parquet(BENCH+'bench_s1.parquet',columns=['id','country','fold']); nq=len(S1); qmap=dict(zip(S1.id,range(nq))); qfold=S1.fold.values.astype(int); qc=S1.country.values
ntrue=pd.read_parquet(BENCH+'ntrue.parquet').ntrue.reindex(S1.id).fillna(0).values.astype(float)
O=pd.read_parquet(OUT+'oof_scores_H30.parquet',columns=['q_id','p_id','src','country','fold','label','s_B0'])
rk=pd.concat([pd.read_parquet(OUT+f'pairfeat_{c}_{s}.parquet',columns=['q_id','p_id','rank_H']) for c in ('India','US') for s in ('s2','s3')],ignore_index=True)
O=O.merge(rk,on=['q_id','p_id'],how='left'); assert O.rank_H.notna().all()
res={}
for K in (5,10,15,20,30):
    d=O[O.rank_H<=K].reset_index(drop=True); qi=d.q_id.map(qmap).values.astype(np.int64); lab=d.label.values; s=d.s_B0.values; rf=d.fold.values.astype(int)
    in_c=np.bincount(qi[lab==1],minlength=nq).astype(float)
    Fq=np.stack([f05_from_pred(qi,s>=t,lab,ntrue,nq) for t in G]); fcf=np.zeros(nq); thr=[]
    for k in range(5):
        sel=qfold!=k; ti=int(Fq[:,sel].mean(1).argmax()); thr.append(float(G[ti])); fcf[qfold==k]=Fq[ti,qfold==k]
    pred=s>=np.array(thr)[rf]; tp,npred=query_counts(qi,pred,lab,nq); single=ntrue==0; cnt=np.bincount(qi,minlength=nq)
    r=dict(F05=float(fcf.mean()),F05_India=float(fcf[qc=='India'].mean()),F05_US=float(fcf[qc=='US'].mean()),precision_micro=float(tp.sum()/max(npred.sum(),1)),recall_micro=float(tp.sum()/ntrue.sum()),
           singleton_acc=float((npred[single]==0).mean()),candidate_recall=float(lab.sum()/ntrue.sum()),cands_per_entity_mean=float(cnt.mean()),cands_per_entity_p95=float(np.percentile(cnt,95)),cands_per_entity_max=int(cnt.max()),thr=thr,
           false_merges=int(npred.sum()-tp.sum()),missed_model_rejected=int(((lab==1)&~pred).sum()),missed_blocking_lost=int((ntrue-in_c).sum()))
    for src in ('s2','s3'): m=(d.src.values==src); r[f'recall_of_candidates_{src}']=float((pred&(lab==1)&m).sum()/max(((lab==1)&m).sum(),1)); r[f'candidate_recall_{src}']=float(((lab==1)&m).sum()/max((ntrue.sum()/2),1))
    res[K]=r; print(K,{k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items() if k in('F05','F05_India','F05_US','precision_micro','recall_micro','singleton_acc','candidate_recall','cands_per_entity_mean','cands_per_entity_p95')},flush=True)
json.dump(res,open(OUT+'kcurve.json','w'),indent=1,default=float)
