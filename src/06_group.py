"""Group-consistency ablation: does evidence from *other* candidates of the same S1 entity (S2<->S3 agreement) raise macro F0.5?
Stage 1 = pairwise model (out-of-fold scores on train, full model on val). Stage 2 adds:
  anchor features  : similarity between candidate p and the most confident *other* candidate of the same S1 entity (name+address)
  aggregate feats  : stage-1 score of p, #other confident candidates, top-1/2 stage-1 scores of the entity, rank of p's stage-1 score."""
import sys,numpy as np,pandas as pd,lightgbm as lgb,time
sys.path.insert(0,'pilot')
from metric import evaluate
from feats import build
T0=time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
gp=pd.read_parquet('pilot/gt_pairs.parquet'); s1=pd.read_parquet('pilot/s1_n.parquet').set_index('id')
X=pd.concat([pd.read_parquet(f'pilot/feat_{c}_{s}_K30.parquet') for c in ('US','India') for s in ('s2','s3')],ignore_index=True)
g=X.groupby('q_id').fused; X['fused_rank_all']=g.rank(ascending=False,method='first'); X['fused_gap_top_all']=X.fused-g.transform('max')
ev=pd.Index(sum([list(s1[s1.country==c].reset_index().sample(frac=1,random_state=0).head(20000).id) for c in ('US','India')],[]))
truth_n=gp[gp.s1.isin(ev)].groupby('s1').size(); h=lambda s:int(s[-3:])%10
val_q=ev[[h(q)>=7 for q in ev]]; tr_q=set(ev[[h(q)<7 for q in ev]])
meta=['q_id','p_id','label','country','src']; feats=[c for c in X.columns if c not in meta]
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=8)
fit=lambda d,f:lgb.train(P,lgb.Dataset(d[f],d.label),400)
def best(v,s,qs):
    b=(-1,)
    for t in np.r_[np.arange(0.3,0.96,0.05)]:
        r,_=evaluate(v.q_id.values,s>=t,v.label.values,truth_n,qs)
        if r['F05']>b[0]: b=(r['F05'],float(t),r)
    return b
X['tr']=X.q_id.isin(tr_q); tr=X[X.tr].reset_index(drop=True); va=X[~X.tr & X.q_id.isin(set(val_q))].reset_index(drop=True)
# ---- stage-1 scores: 2-fold OOF on train, full-train model on val
fold=tr.q_id.map(lambda s:int(s[-4])%2).values
tr['s1']=0.0
for k in (0,1):
    m=fit(tr[fold!=k],feats); tr.loc[fold==k,'s1']=m.predict(tr.loc[fold==k,feats])
m1=fit(tr,feats); va['s1']=m1.predict(va[feats]); b1=best(va,va.s1.values,val_q); log('stage-1 (pairwise) val F0.5',round(b1[0],4),'thr',b1[1])
# ---- anchor features
pool=pd.concat([pd.read_parquet('pilot/s2_n.parquet'),pd.read_parquet('pilot/s3_n.parquet')]).set_index('id')
def add_group(d):
    d=d.copy(); d['rk1']=d.groupby('q_id').s1.rank(ascending=False,method='first')
    top=d.sort_values(['q_id','s1'],ascending=[True,False]).groupby('q_id')
    d['top1']=top.s1.transform('max') if False else d.groupby('q_id').s1.transform('max')
    srt=d.sort_values(['q_id','s1'],ascending=[True,False]); sec=srt.groupby('q_id').s1.nth(1); sec=pd.Series(sec.values,index=srt.loc[sec.index,'q_id'].values)
    d['top2']=d.q_id.map(sec).fillna(0).values; d['n_conf5']=d.groupby('q_id').s1.transform(lambda x:(x>0.5).sum()); d['n_conf9']=d.groupby('q_id').s1.transform(lambda x:(x>0.9).sum())
    # anchor = best-scoring candidate of the entity other than p itself
    a1=srt.groupby('q_id').head(2); a1['r']=a1.groupby('q_id').cumcount(); A1=a1[a1.r==0].set_index('q_id'); A2=a1[a1.r==1].set_index('q_id')
    a_id=np.where(d.p_id.values==d.q_id.map(A1.p_id).values,d.q_id.map(A2.p_id).values,d.q_id.map(A1.p_id).values)
    a_s=np.where(d.p_id.values==d.q_id.map(A1.p_id).values,d.q_id.map(A2.s1).values,d.q_id.map(A1.s1).values)
    d['anc_s']=np.nan_to_num(a_s.astype(float),nan=0.0); has=pd.notna(a_id)&(a_id!=None)
    Fa=pd.DataFrame({'q_id':np.where(has,a_id,d.p_id.values),'p_id':d.p_id.values,'cos_nw':0.,'cos_aw':0.,'cos_nc':0.,'cos_ac':0.,'cos_all':0.,'src':d.src.values})
    Xa=[]
    for a in range(0,len(Fa),300000): Xa.append(build(Fa.iloc[a:a+300000],pool,pool))
    Xa=pd.concat(Xa,ignore_index=True)
    keep=['n_eq_nc','n_lev_nc','n_lev_ns','n_jw_nc','n_tsr','n_jac','n_prefix','n_contains','a_eq','a_lev','a_tsr','a_jac','a_num_ov','a_house_eq','a_miss_q','a_state_eq','a_num_conflict']
    for c in keep: d['anc_'+c]=np.where(has,Xa[c].values,np.nan)
    d['anc_conf']=d.anc_s*has
    return d
GF=['s1','rk1','top1','top2','n_conf5','n_conf9','anc_s']; AF=['anc_'+c for c in ['n_eq_nc','n_lev_nc','n_lev_ns','n_jw_nc','n_tsr','n_jac','n_prefix','n_contains','a_eq','a_lev','a_tsr','a_jac','a_num_ov','a_house_eq','a_miss_q','a_state_eq','a_num_conflict']]
tr2=add_group(tr); log('train group feats'); va2=add_group(va); log('val group feats')
res={'stage1_pairwise':b1[0]}
for nm,f in {'stage2_aggregates_only':feats+GF[:6],'stage2_anchor_only':feats+['s1']+AF+['anc_s'],'stage2_full(agg+anchor)':feats+GF+AF}.items():
    m=fit(tr2,f); s=m.predict(va2[f]); b=best(va2,s,val_q); res[nm]=b[0]; log(nm,'F0.5',round(b[0],4),'thr',b[1],'P',round(b[2]['precision_micro'],4),'R',round(b[2]['recall_micro'],4),'singleton',round(b[2]['singleton_acc'],4))
print(res)
