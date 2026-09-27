"""Exp3: rebuild the Exp2 H@10 stage-1 feature file from CORRECTED normalisation (norm3 frames), for exactly the same candidate pairs (no new retrieval).
Same feature code (feats.build, unchanged), same TF-IDF definitions as 20_retrieval; only the text representation of native-script records differs.
-> pilot/exp3/feats_H_K10_fix.parquet  (same schema as pilot/exp2/feats_H_K10.parquet)"""
import sys,gc,time,numpy as np,pandas as pd
sys.path.insert(0,'pilot')
from blk import tfidf_word,tfidf_char
from feats import build,idf_tables
E2='pilot/exp2/'; E3='pilot/exp3/'; T0=time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
def rowdot(A,B,qi,pj,ch=1_000_000):
    out=np.empty(len(qi),np.float32)
    for s in range(0,len(qi),ch): out[s:s+ch]=np.asarray(A[qi[s:s+ch]].multiply(B[pj[s:s+ch]]).sum(1)).ravel()
    return out
tok=lambda p,x: x.map(lambda s:' '.join(p+t for t in s.split()))
O=pd.read_parquet(E2+'feats_H_K10.parquet',columns=['q_id','p_id','label','country','src','fold'])
S1=pd.read_parquet(E3+'bench3_s1.parquet'); POOL={s:pd.read_parquet(E3+f'bench3_{s}.parquet') for s in ('s2','s3')}; parts=[]
for country in ('India','US'):
    for src in ('s2','s3'):
        R=O[(O.country==country)&(O.src==src)].reset_index(drop=True); Q=S1[S1.country==country].reset_index(drop=True); P=POOL[src][POOL[src].country==country].reset_index(drop=True)
        qmap=dict(zip(Q.id,range(len(Q)))); pmap=dict(zip(P.id,range(len(P)))); qi=R.q_id.map(qmap).values.astype(np.int64); pj=R.p_id.map(pmap).values.astype(np.int64)
        Pw,Qw=tfidf_word(P.nc,Q.nc); Pa,Qa=tfidf_word(P.ac,Q.ac); Pnc,Qnc=tfidf_char(P.nc.str.replace(' ',''),Q.nc.str.replace(' ',''),(4,4)); Pac,Qac=tfidf_char(P.ac,Q.ac,(4,4),'char_wb')
        Pall,Qall=tfidf_word(tok('n_',P.nc)+' '+tok('a_',P.ac),tok('n_',Q.nc)+' '+tok('a_',Q.ac))
        F=R.copy(); F['cos_nw']=rowdot(Qw,Pw,qi,pj); F['cos_aw']=rowdot(Qa,Pa,qi,pj); F['cos_nc']=rowdot(Qnc,Pnc,qi,pj); F['cos_ac']=rowdot(Qac,Pac,qi,pj); F['cos_all']=rowdot(Qall,Pall,qi,pj)
        del Pw,Qw,Pa,Qa,Pnc,Qnc,Pac,Qac,Pall,Qall; gc.collect(); log(country,src,'cosines recomputed',len(F))
        Qi=Q.set_index('id'); Pi=P.set_index('id'); idn,ida=idf_tables([Q,P]); X=pd.concat([build(F.iloc[a:a+300000],Qi,Pi,idn,ida) for a in range(0,len(F),300000)],ignore_index=True)
        for c in ('q_id','p_id','label','country','src','fold'): X[c]=F[c].values
        X=X.sort_values(['q_id','fused'],ascending=[True,False],kind='stable').reset_index(drop=True); f=X.fused.values; g=pd.Series(f).groupby(X.q_id.values)
        X['fused_rank']=g.rank(ascending=False,method='first').values; X['fused_gap_top']=f-g.transform('max').values; X['n_cands']=g.transform('size').values; parts.append(X); log(country,src,'features',X.shape); del F,X; gc.collect()
X=pd.concat(parts,ignore_index=True); g=X.groupby('q_id').fused; X['fused_rank_all']=g.rank(ascending=False,method='first'); X['fused_gap_top_all']=X.fused-g.transform('max')
ref=pd.read_parquet(E2+'feats_H_K10.parquet',columns=None,nrows=1) if False else None
X.to_parquet(E3+'feats_H_K10_fix.parquet'); log('saved',X.shape,'label sum',int(X.label.sum()))
