"""Exp2 step 4: pair-feature store. Builds the SAME features as Exp1 (feats.build, unchanged) once for every row selected by any compared config
(H top-30 | G top-15 | A top-15 per source), so that each config's candidate file can be assembled without recomputing features.
Output: pilot/exp2/pairfeat_{country}_{src}.parquet (features without context columns + rank_A/G/H + labels)."""
import sys,os,gc,time,numpy as np,pandas as pd
sys.path.insert(0,'pilot')
from feats import build,idf_tables
OUT='pilot/exp2/'; BENCH='pilot/exp1/'; T0=time.time(); KH=int(os.environ.get('KH','30')); KGA=15
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
def rank_within(qi,score,allowed):
    idx=np.flatnonzero(allowed); o=np.lexsort((-score[idx],qi[idx])); idx=idx[o]; q=qi[idx]; st=np.r_[True,q[1:]!=q[:-1]]; s=np.flatnonzero(st); g=np.cumsum(st)-1
    r=np.full(len(qi),9999,np.int32); r[idx]=np.arange(len(idx))-s[g]+1; return r
S1=pd.read_parquet(BENCH+'bench_s1.parquet'); POOL={s:pd.read_parquet(BENCH+f'bench_{s}.parquet') for s in ('s2','s3')}
combos=sys.argv[1:] or [f'{c}/{s}' for c in ('India','US') for s in ('s2','s3')]
for combo in combos:
    country,src=combo.split('/'); D=pd.read_parquet(OUT+f'union_{country}_{src}.parquet'); R=pd.read_parquet(OUT+f'ranked_{country}_{src}.parquet'); qi=D.qi.values
    fused=R.fused.values; A=np.minimum.reduce([D.r_nw.values,D.r_aw.values,D.r_nc.values,D.r_ac.values])<999
    rA=rank_within(qi,fused,A); rG=rank_within(qi,R.score_G.values,np.ones(len(D),bool)); rH=rank_within(qi,R.score_H.values,np.ones(len(D),bool))
    keep=(rH<=KH)|(rG<=KGA)|(rA<=KGA); log(combo,'rows kept',int(keep.sum()),'of',len(D),'per query',round(keep.sum()/(qi.max()+1),1))
    F=D.loc[keep,['qi','pj','cos_nw','cos_aw','cos_nc','cos_ac','cos_all','label']].reset_index(drop=True); F['rank_A']=rA[keep].astype(np.int16); F['rank_G']=rG[keep].astype(np.int16); F['rank_H']=rH[keep].astype(np.int16); F['score_H']=R.score_H.values[keep]
    del D,R,fused,A,rA,rG,rH,qi; gc.collect()
    Q=S1[S1.country==country].reset_index(drop=True); P=POOL[src][POOL[src].country==country].reset_index(drop=True)
    F['q_id']=Q.id.values[F.qi.values]; F['p_id']=P.id.values[F.pj.values]; F['fold']=Q.fold.values[F.qi.values]; F['src']=src
    Qi=Q.set_index('id'); Pi=P.set_index('id'); idn,ida=idf_tables([Q,P]); X=[]
    for a in range(0,len(F),300000): X.append(build(F.iloc[a:a+300000],Qi,Pi,idn,ida))
    X=pd.concat(X,ignore_index=True)
    for c in ('q_id','p_id','fold','label','rank_A','rank_G','rank_H','score_H','qi'): X[c]=F[c].values
    X['country']=country; X['src']=src
    X.drop(columns=['fused_rank','fused_gap_top','n_cands']).to_parquet(OUT+f'pairfeat_{country}_{src}.parquet'); log(combo,'features',X.shape); del F,X,Q,P,Qi,Pi; gc.collect()
