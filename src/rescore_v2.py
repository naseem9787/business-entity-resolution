"""HYBRID v2: keep the PRODUCTION candidate set (old-normalisation retrieval + ranker, cached in <work>/scored and <work>/parts) and RE-SCORE it
with norm3 text + base features + v2 per-shard features + v2 stage-1. No retrieval is re-run -> candidate_pairs are identical to the production candidates.
usage (driver): python pilot/rescore_v2.py <prod_work_dir> <out_feats_dir>     (writes one feature parquet per shard; resumable)"""
import os,sys,glob,time,gc,importlib,multiprocessing as mp
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'exp3')); sys.path.insert(0,HERE)
import norm3; sys.modules['norm']=norm3
import numpy as np,pandas as pd
import make_submission as ms, v2feats
from feats import build,idf_tables
TRANSLIT=os.path.join(HERE,'final_optimization','02_ablation','translit_map.json')
def _renorm(df):
    prep=importlib.import_module('01_prep'); d=pd.DataFrame({'entity_id':df.id.values,'business_name':df.name_raw.values,'business_address':df.addr_raw.values,'country':df.country.values}); return prep.work(d)
def renorm(df,pool):
    ch=[df.iloc[i:i+50000] for i in range(0,len(df),50000)]; return pd.concat(pool.map(_renorm,ch),ignore_index=True) if len(ch)>1 else _renorm(df)
def rowdot(A,B,qi,pj,ch=1_000_000):
    out=np.empty(len(qi),np.float32)
    for s in range(0,len(qi),ch): out[s:s+ch]=np.asarray(A[qi[s:s+ch]].multiply(B[pj[s:s+ch]]).sum(1)).ravel()
    return out
def rescore_shard(Q,Ps,cand):
    """Q, Ps[src]: norm3-normalised frames; cand: q_id,p_id,src (production candidates of this shard)"""
    parts=[]; PP=[]
    for src in ('s2','s3'):
        F=cand[cand.src==src].reset_index(drop=True); P=Ps.get(src)
        if P is None or len(F)==0: continue
        qm=dict(zip(Q.id,range(len(Q)))); pm=dict(zip(P.id,range(len(P)))); qi=F.q_id.map(qm).values.astype(np.int64); pj=F.p_id.map(pm).values.astype(np.int64)
        Pw,Qw=ms.SW(P.nc,Q.nc); Pa,Qa=ms.SW(P.ac,Q.ac); Pnc,Qnc=ms.SC(P.nc.str.replace(' ',''),Q.nc.str.replace(' ',''),(4,4)); Pac,Qac=ms.SC(P.ac,Q.ac,(4,4),'char_wb')
        Pall,Qall=ms.SW(ms.tok('n_',P.nc)+' '+ms.tok('a_',P.ac),ms.tok('n_',Q.nc)+' '+ms.tok('a_',Q.ac))
        F['cos_nw']=rowdot(Qw,Pw,qi,pj); F['cos_aw']=rowdot(Qa,Pa,qi,pj); F['cos_nc']=rowdot(Qnc,Pnc,qi,pj); F['cos_ac']=rowdot(Qac,Pac,qi,pj); F['cos_all']=rowdot(Qall,Pall,qi,pj); del Pw,Qw,Pa,Qa,Pnc,Qnc,Pac,Qac,Pall,Qall
        Qi=Q.set_index('id'); Pi=P.set_index('id'); idn,ida=idf_tables([Q,P]); r=ms.CFG['N_REF']/(len(Q)+len(P)); idn['__cnt__']={k:v*r for k,v in idn['__cnt__'].items()}
        X=pd.concat([build(F.iloc[s:s+300000],Qi,Pi,idn,ida) for s in range(0,len(F),300000)],ignore_index=True); X['q_id']=F.q_id.values; X['p_id']=F.p_id.values; X['src']=src
        X=X.sort_values(['q_id','fused'],ascending=[True,False],kind='stable').reset_index(drop=True); f=X.fused.values; g=pd.Series(f).groupby(X.q_id.values)
        X['fused_rank']=g.rank(ascending=False,method='first').values; X['fused_gap_top']=f-g.transform('max').values; X['n_cands']=g.transform('size').values; parts.append(X)
        PP.append(P[['id','country','nc','ac','an','nn','addr_raw','name_raw']].assign(src=src))
    X=pd.concat(parts,ignore_index=True); g=X.groupby('q_id').fused; X['fused_rank_all']=g.rank(ascending=False,method='first'); X['fused_gap_top_all']=X.fused-g.transform('max')
    P=pd.concat(PP,ignore_index=True).drop_duplicates('id'); X=v2feats.add(X,Q,P,TRANSLIT,norm3.LEGAL); Pi=P.set_index('id')
    X['p_empty']=(Pi.addr_raw.reindex(X.p_id).values=='').astype(np.int8); X['p_native']=Pi.name_raw.reindex(X.p_id).str.contains('[^\x00-ɏ]',regex=True).fillna(False).values.astype(np.int8)
    return X
def drive(work,out,model_dir=None):
    import lightgbm as lgb,json as _j
    M=lgb.Booster(model_file=os.path.join(model_dir,'stage1.txt')) if model_dir else None; FE=_j.load(open(os.path.join(model_dir,'meta.json')))['features'] if model_dir else None
    import argparse; os.makedirs(out,exist_ok=True); a=argparse.Namespace(work=work); pool=mp.Pool(4); T0=time.time()
    shards=sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(work,'scored','*.parquet'))); nost={}
    for i,sh in enumerate(shards,1):
        fo=os.path.join(out,sh+'.parquet')
        if os.path.exists(fo): continue
        c,k=sh.split('__'); t=time.time(); cand=pd.read_parquet(os.path.join(work,'scored',sh+'.parquet'),columns=['q_id','p_id','src'])
        if len(cand)==0: pd.DataFrame().to_parquet(fo); continue
        Q=renorm(ms.load_parts(a,'s1',c,k),pool); Ps={}
        for src in ('s2','s3'):
            own=ms.load_parts(a,src,c,k); own=renorm(own,pool) if own is not None else None
            if k not in('_ALL','_NOSTATE'):
                if (c,src) not in nost: b=ms.load_parts(a,src,c,'_NOSTATE'); nost[(c,src)]=renorm(b,pool) if b is not None else None   # broadcast pool normalised once per country/source
                parts=[p for p in (own,nost[(c,src)]) if p is not None]
            else: parts=[p for p in (own,) if p is not None]
            Ps[src]=pd.concat(parts,ignore_index=True) if parts else None
        X=rescore_shard(Q,Ps,cand)
        if M is not None: X['s']=M.predict(X[FE].to_numpy(np.float32)); X=X[['q_id','p_id','src','s','p_empty','p_native']]
        X.to_parquet(fo); print(f'[{time.time()-T0:7.0f}s] shard {i}/{len(shards)} {sh}: {len(Q)} S1, {len(X)} rows, {time.time()-t:.0f}s',flush=True); del Q,Ps,X,cand; gc.collect()
    pool.close()
if __name__=='__main__': drive(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
