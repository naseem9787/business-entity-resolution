"""Pairwise matching experiments on the pilot candidates.
usage: python pilot/04_match.py <countries comma> <KFIN>   (features cached in pilot/feat_*.parquet)"""
import sys,os,json,time,numpy as np,pandas as pd,lightgbm as lgb
sys.path.insert(0,'pilot')
from feats import build,idf_tables
from metric import evaluate
countries=sys.argv[1].split(','); KFIN=int(sys.argv[2]); T0=time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
gp=pd.read_parquet('pilot/gt_pairs.parquet')
s1=pd.read_parquet('pilot/s1_n.parquet').set_index('id')
srcs={s:pd.read_parquet(f'pilot/{s}_n.parquet') for s in ('s2','s3')}
XS=[]
for c in countries:
    for s in ('s2','s3'):
        fn=f'pilot/feat_{c}_{s}_K{KFIN}.parquet'
        if os.path.exists(fn): XS.append(pd.read_parquet(fn)); continue
        F=pd.read_parquet(f'pilot/cand_{c}_{s}.parquet'); F['fused']=F.cos_nw+2*F.cos_aw+F.cos_nc+2*F.cos_ac
        F=F.sort_values(['q_id','fused'],ascending=[True,False]); F=F[F.groupby('q_id').cumcount()<KFIN].reset_index(drop=True)
        P=srcs[s][srcs[s].country==c].set_index('id'); Q=s1[s1.country==c]
        idn,ida=idf_tables([Q.reset_index(),P.reset_index()])
        X=[];
        for a in range(0,len(F),300000):
            Fc=F.iloc[a:a+300000]; X.append(build(Fc,Q,P,idn,ida))
        X=pd.concat(X,ignore_index=True)
        # context features must be computed on the whole candidate list per query, not per chunk
        f=X.fused.values; g=pd.Series(f).groupby(F.q_id.values)
        X['fused_rank']=g.rank(ascending=False,method='first').values; X['fused_gap_top']=f-g.transform('max').values; X['n_cands']=g.transform('size').values
        X['q_id']=F.q_id.values; X['p_id']=F.p_id.values; X['label']=F.label.values; X['country']=c; X['src']=s
        X.to_parquet(fn); XS.append(X); log('features',c,s,X.shape)
X=pd.concat(XS,ignore_index=True)
# per-query cross-source context (rank among S2+S3 candidates)
g=X.groupby('q_id').fused; X['fused_rank_all']=g.rank(ascending=False,method='first'); X['fused_gap_top_all']=X.fused-g.transform('max')
# truth for all queries in the evaluated set
qids=X.q_id.unique(); allq=pd.Index(sorted(set(X.q_id)|set()))
ev_q=[]  # all evaluated queries: the sampled queries for those countries (first 20000 shuffled per country as in blocking)
for c in countries:
    ev_q+=list(s1[s1.country==c].reset_index().sample(frac=1,random_state=0).head(int(os.environ.get('NQ','20000'))).id)
ev_q=pd.Index(ev_q); truth_n=gp[gp.s1.isin(ev_q)].groupby('s1').size()
h=lambda s: int(s[-3:])%10  # deterministic query split
isv=ev_q.map(lambda s:h(s)>=7); val_q=ev_q[isv]; tr_q=ev_q[~isv]
tr=X[X.q_id.isin(set(tr_q))]; va=X[X.q_id.isin(set(val_q))].reset_index(drop=True)
log('train pairs',len(tr),'pos',tr.label.sum(),'| val pairs',len(va),'pos',va.label.sum(),'| val queries',len(val_q),'| truth pairs total(val)',int(truth_n.reindex(val_q).fillna(0).sum()))
meta=['q_id','p_id','label','country','src']
allf=[c for c in X.columns if c not in meta]
def fit(feats,trd,params=None,rounds=400):
    p=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=8)
    if params: p.update(params)
    return lgb.train(p,lgb.Dataset(trd[feats],trd.label),rounds)
def sweep(va,score,tag,mask_extra=None):
    best=(-1,None); rows=[]
    for t in np.r_[np.arange(0.1,0.9,0.05),0.9,0.93,0.95,0.97,0.98,0.99]:
        r,_=evaluate(va.q_id.values,score>=t,va.label.values,truth_n,val_q); rows.append((round(float(t),3),r['F05'],r['precision_micro'],r['recall_micro']));
        if r['F05']>best[0]: best=(r['F05'],float(t),r)
    return best,rows
res={}
def run(name,feats,trd,va=va,params=None):
    m=fit(feats,trd,params); s=m.predict(va[feats]); b,rows=sweep(va,s,name); res[name]=dict(best_F05=b[0],thr=b[1],detail=b[2],sweep=rows,n_feat=len(feats)); log(name,'best F0.5',round(b[0],4),'thr',b[1],'P',round(b[2]['precision_micro'],4),'R',round(b[2]['recall_micro'],4),'singleton_acc',round(b[2]['singleton_acc'],4)); return m,s
# --- 0. simple rule baseline (no ML): fused cosine threshold
for c in ['cos_all','fused']:
    b,_=sweep(va,va[c].values/(va[c].max() if c=='fused' else 1),c); res['rule_'+c]=dict(best_F05=b[0],thr=b[1],detail=b[2]); log('rule',c,round(b[0],4))
# --- 1. main LightGBM, all features, all top-K candidates as (hard) negatives
m,s=run('lgbm_all',allf,tr)
imp=pd.Series(m.feature_importance('gain'),allf).sort_values(ascending=False); res['importance']=(imp/imp.sum()).round(4).head(25).to_dict()
# --- 2. hard-negative ablation: only random negatives (+ all positives)
rng=np.random.default_rng(0); neg=tr[tr.label==0]; rneg=neg.sample(min(len(neg),int(tr.label.sum()*3)),random_state=0)
run('lgbm_random_negatives_only',allf,pd.concat([tr[tr.label==1],rneg]))
# hard negatives only among top-5 fused ranks + random
hn=tr[(tr.label==0)&(tr.fused_rank_all<=5)]; run('lgbm_hard_top5_negs+pos',allf,pd.concat([tr[tr.label==1],hn]))
# --- 3. representation ablations
notr=[c for c in allf if not c.endswith('_nt')]; run('lgbm_no_translit_feats',notr,tr)
run('lgbm_name_only',[c for c in allf if c.startswith('n_') or c in('name_freq_q','name_freq_p','src3')],tr)
run('lgbm_addr_only',[c for c in allf if c.startswith('a_') or c=='src3'],tr)
ctx=['fused_rank','fused_gap_top','n_cands','fused_rank_all','fused_gap_top_all']; run('lgbm_no_context_feats',[c for c in allf if c not in ctx],tr)
# --- 4. decision rules on best model score
va['s']=s
def rule_eval(name,mask):
    r,f=evaluate(va.q_id.values,mask,va.label.values,truth_n,val_q); res[name]=dict(F05=r['F05'],P=r['precision_micro'],R=r['recall_micro'],singleton_acc=r['singleton_acc']); log(name,round(r['F05'],4),'P',round(r['precision_micro'],4),'R',round(r['recall_micro'],4))
b0=res['lgbm_all']['thr']
mx=va.groupby('q_id').s.transform('max')
for tt in (0.5,0.6,0.7,0.8): rule_eval(f'thr{tt}',va.s.values>=tt)
for t in (0.5,0.7,0.85,0.9,0.95): rule_eval(f'top1>={t}_and_s>={b0:.2f}',(mx.values>=t)&(va.s.values>=b0))
for src in ('s2','s3'):    # per-source thresholds
    for d in (-0.15,-0.1,0.1,0.15):
        th=np.where(va.src.values==src,b0+d,b0); rule_eval(f'thr_{src}{d:+.2f}',va.s.values>=th)
for rel in (0.3,0.5,0.7): rule_eval(f'relative>= {rel}*top1',(va.s.values>=b0)&(va.s.values>=rel*mx.values))
# --- 5. per-country / per-source / by-#matches breakdown at best threshold
pm=va.s.values>=b0
for c in countries:
    qs=[q for q in val_q if s1.loc[q,'country']==c]; msk=va.q_id.isin(set(qs)).values
    r,_=evaluate(va.q_id.values[msk],pm[msk],va.label.values[msk],truth_n,pd.Index(qs)); res['country_'+c]=r; log('country',c,round(r['F05'],4),'P',round(r['precision_micro'],4),'R',round(r['recall_micro'],4),'singleton',r['singleton_acc'])
_,fq=evaluate(va.q_id.values,pm,va.label.values,truth_n,val_q)
nt=truth_n.reindex(val_q).fillna(0).values; res['F05_by_nmatches']={k:float(fq[(nt>=a)&(nt<=b)].mean()) for k,(a,b) in {'0':(0,0),'1':(1,1),'2':(2,2),'3-5':(3,5),'6+':(6,99)}.items()}
log('by #matches',res['F05_by_nmatches'])
# candidate ceiling at KFIN for these val queries
res['candidate_recall_at_KFIN']=float(va.label.sum()/max(truth_n.reindex(val_q).fillna(0).sum(),1)); res['avg_cands_per_query']=float(len(va)/len(val_q))
json.dump(res,open(f'pilot/report_match_{"_".join(countries)}_K{KFIN}.json','w'),default=float,indent=1); va.to_parquet(f'pilot/val_scored_{"_".join(countries)}_K{KFIN}.parquet'); log('saved')
