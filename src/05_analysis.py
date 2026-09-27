import sys,json,numpy as np,pandas as pd,lightgbm as lgb
sys.path.insert(0,'pilot')
from metric import evaluate
pd.set_option('display.width',250,'display.max_colwidth',70)
gp=pd.read_parquet('pilot/gt_pairs.parquet'); s1=pd.read_parquet('pilot/s1_n.parquet').set_index('id')
X=pd.concat([pd.read_parquet(f'pilot/feat_{c}_{s}_K30.parquet') for c in ('US','India') for s in ('s2','s3')],ignore_index=True)
g=X.groupby('q_id').fused; X['fused_rank_all']=g.rank(ascending=False,method='first'); X['fused_gap_top_all']=X.fused-g.transform('max')
ev=pd.Index(sum([list(s1[s1.country==c].reset_index().sample(frac=1,random_state=0).head(20000).id) for c in ('US','India')],[]))
truth_n=gp[gp.s1.isin(ev)].groupby('s1').size(); cty=s1.country
h=lambda s:int(s[-3:])%10
val_q=ev[[h(q)>=7 for q in ev]]; tr_q=ev[[h(q)<7 for q in ev]]
meta=['q_id','p_id','label','country','src']; feats=[c for c in X.columns if c not in meta]
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=8)
def fit(d,f=feats): return lgb.train(P,lgb.Dataset(d[f],d.label),400)
def best(va,s,qs,tn=truth_n):
    b=(-1,)
    for t in np.r_[np.arange(0.3,0.95,0.05)]:
        r,_=evaluate(va.q_id.values,s>=t,va.label.values,tn,qs)
        if r['F05']>b[0]: b=(r['F05'],float(t),r)
    return b
tr=X[X.q_id.isin(set(tr_q))]; va=X[X.q_id.isin(set(val_q))].reset_index(drop=True)
m=fit(tr); va['s']=m.predict(va[feats]); b0=best(va,va.s.values,val_q); print('base',round(b0[0],4),'thr',b0[1]); thr=b0[1]; out={'base':b0[0]}
# ---- 1. candidate-budget Pareto: K candidates per source per entity (by fused rank), model retrained on same K
print('\nK per source | avg cands/entity | cand recall | F0.5 | P | R')
out['K']={}
for K in (1,2,3,5,8,10,15,20,30):
    trK=tr[tr.fused_rank<=K]; vaK=va[va.fused_rank<=K].reset_index(drop=True); mK=fit(trK); sK=mK.predict(vaK[feats]); bK=best(vaK,sK,val_q)
    cr=vaK.label.sum()/truth_n.reindex(val_q).sum(); print(K,'|',round(len(vaK)/len(val_q),1),'|',round(cr,4),'|',round(bK[0],4),'|',round(bK[2]['precision_micro'],4),'|',round(bK[2]['recall_micro'],4)); out['K'][K]=dict(avg_cands=len(vaK)/len(val_q),cand_recall=float(cr),F05=bK[0],P=bK[2]['precision_micro'],R=bK[2]['recall_micro'],thr=bK[1])
# ---- 2. proper hard-negative variants (positives + top-5 hard + random sample of the rest, so the model still sees the tail)
print('\nhard-negative variants (train subsets, all val candidates scored):')
pos=tr[tr.label==1]; neg=tr[tr.label==0]; hard=neg[neg.fused_rank_all<=5]; rest=neg[neg.fused_rank_all>5]
for nm,d in {'all top-30 negatives (current)':tr,'random negatives only (3x pos)':pd.concat([pos,neg.sample(3*len(pos),random_state=0)]),'top5 hard + 1x random tail':pd.concat([pos,hard,rest.sample(len(pos),random_state=0)]),'top5 hard only':pd.concat([pos,hard]),'top10 hard + tail 1x':pd.concat([pos,neg[neg.fused_rank_all<=10],neg[neg.fused_rank_all>10].sample(len(pos),random_state=0)])}.items():
    mm=fit(d); bb=best(va,mm.predict(va[feats]),val_q); print(f'{nm:38s} F0.5 {bb[0]:.4f} P {bb[2]["precision_micro"]:.4f} R {bb[2]["recall_micro"]:.4f} singleton {bb[2]["singleton_acc"]:.4f}'); out.setdefault('hardneg',{})[nm]=bb[0]
# ---- 3. leave-one-country-out (open-set format robustness proxy)
print('\nleave-one-country-out:')
for trc,tec in (('US','India'),('India','US')):
    qtr=[q for q in tr_q if cty[q]==trc]; qte=[q for q in ev if cty[q]==tec]
    d=X[X.q_id.isin(set(qtr))]; v=X[X.q_id.isin(set(qte))].reset_index(drop=True); mm=fit(d); sv=mm.predict(v[feats])
    r_at_own=None; bb=best(v,sv,pd.Index(qte)); r_fix,_=evaluate(v.q_id.values,sv>=thr,v.label.values,truth_n,pd.Index(qte))
    # in-country reference: model trained on same-country train queries, validated on same val queries
    vq=[q for q in val_q if cty[q]==tec]; vv=va[va.q_id.isin(set(vq))]; ref=best(vv,vv.s.values,pd.Index(vq))
    d2=X[X.q_id.isin(set(q for q in tr_q if cty[q]==tec))]; m2=fit(d2); ref2=best(vv,m2.predict(vv[feats]),pd.Index(vq))
    print(f'train {trc} -> test {tec}: F0.5 (best thr {bb[1]:.2f}) {bb[0]:.4f} | at thr fixed from other-country val {thr:.2f}: {r_fix["F05"]:.4f} | joint model on {tec} val {ref[0]:.4f} | {tec}-only model {ref2[0]:.4f}')
    out.setdefault('loco',{})[f'{trc}->{tec}']=dict(F05_best=bb[0],F05_fixed_thr=r_fix['F05'],joint_model=ref[0],same_country_model=ref2[0])
# ---- 4. error analysis at base threshold
va['pred']=va.s>=thr
fn=va[(va.label==1)&(~va.pred)]; fp=va[(va.label==0)&(va.pred)]
tot_true=truth_n.reindex(val_q).sum(); print(f'\nval true pairs {tot_true:.0f}: candidates cover {va.label.sum()}, TP {((va.label==1)&va.pred).sum()}, FN-with-candidate {len(fn)}, FN-blocking-lost {tot_true-va.label.sum():.0f}; FP {len(fp)}')
def prof(d,tag):
    print(tag,'n',len(d),'| addr missing p:',round(d.a_miss_p.mean(),3),'| a_eq',round(d.a_eq.mean(),3),'| a_num_conflict',round(d.a_num_conflict.mean(),3),'| n_eq_nc',round(d.n_eq_nc.mean(),3),'| by src s3',round((d.src=='s3').mean(),3),'| India',round((d.country=='India').mean(),3),'| script_diff',round(d.n_script_diff.mean(),3))
prof(fn,'FN'); prof(fp,'FP'); prof(va[(va.label==1)&va.pred],'TP')
n=pd.read_parquet('pilot/s1_n.parquet').set_index('id'); pp=pd.concat([pd.read_parquet('pilot/s2_n.parquet'),pd.read_parquet('pilot/s3_n.parquet')]).set_index('id')
for tag,d in (('FALSE NEGATIVES',fn),('FALSE POSITIVES',fp)):
    print('\n',tag)
    for r in d.sample(10,random_state=2).itertuples():
        print(f' s={r.s:.2f} [{r.country}/{r.src}] Q:',n.loc[r.q_id,'name_raw'],'|',n.loc[r.q_id,'addr_raw'][:70],'\n                 P:',pp.loc[r.p_id,'name_raw'],'|',pp.loc[r.p_id,'addr_raw'][:70])
json.dump(out,open('pilot/report_analysis.json','w'),default=float,indent=1)
