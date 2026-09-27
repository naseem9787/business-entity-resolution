"""Exp1 step 3-4: blocking for ALL region S1 (same four TF-IDF retrievers as 02_blocking_benchmark, maxdf scaled to pool size),
top-30/source by fused cosine, then feats.build on top-15/source.  Output -> pilot/exp1/cands_top30.parquet, feats_top15.parquet, truth_pairs.parquet, blocking_report.json"""
import sys,os,json,time,numpy as np,pandas as pd,scipy.sparse as sp
sys.path.insert(0,'pilot')
from blk import tfidf_word,tfidf_char,prune,topk,rank_within,pair_key
from feats import build,idf_tables
OUT='pilot/exp1/'; KFEAT=15; KKEEP=30; T0=time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
def rowdot(A,B,qi,pj,ch=1_000_000):
    out=np.empty(len(qi),np.float32)
    for s in range(0,len(qi),ch): out[s:s+ch]=np.asarray(A[qi[s:s+ch]].multiply(B[pj[s:s+ch]]).sum(1)).ravel()
    return out
tok=lambda p,x: x.map(lambda s:' '.join(p+t for t in s.split()))
S1=pd.read_parquet(OUT+'bench_s1.parquet'); POOL={s:pd.read_parquet(OUT+f'bench_{s}.parquet') for s in ('s2','s3')}
# truth pairs (from pool owner column) and entity-level totals
tp_=[]
for s,p in POOL.items():
    t=p[p.owner!=''][['owner','id']].rename(columns={'owner':'q_id','id':'p_id'}); t['src']=s; tp_.append(t)
truth=pd.concat(tp_,ignore_index=True); truth.to_parquet(OUT+'truth_pairs.parquet')
ntrue_tot=truth.groupby('q_id').size().reindex(S1.id).fillna(0).astype(int); ntrue_tot.index.name='id'
bucket=lambda n: '1' if n==1 else '2' if n==2 else '3-5' if n<=5 else '6+'
REP={'combos':{}}; CANDS=[]; FEATS=[]
for country in sorted(S1.country.unique()):
    Q=S1[S1.country==country].reset_index(drop=True)
    for src in ('s2','s3'):
        P=POOL[src][POOL[src].country==country].reset_index(drop=True); NP=len(P); nq=len(Q)
        qmap=dict(zip(Q.id,range(nq))); tr=truth[(truth.src==src)&truth.q_id.isin(qmap)]; pm=dict(zip(P.id,range(NP)))
        tq=tr.q_id.map(qmap).values.astype(np.int64); tp=tr.p_id.map(pm).values.astype(np.int64); tk=tq*NP+tp
        log(country,src,'queries',nq,'pool',NP,'true pairs',len(tk))
        Pw,Qw=tfidf_word(P.nc,Q.nc); Pa,Qa=tfidf_word(P.ac,Q.ac)
        Pnc,Qnc=tfidf_char(P.nc.str.replace(' ',''),Q.nc.str.replace(' ',''),(4,4)); Pac,Qac=tfidf_char(P.ac,Q.ac,(4,4),'char_wb')
        Pall,Qall=tfidf_word(tok('n_',P.nc)+' '+tok('a_',P.ac),tok('n_',Q.nc)+' '+tok('a_',Q.ac))
        md=max(50,int(0.0066*NP)); ks=[]
        for Pm,Qm in ((Pw,Qw),(Pa,Qa),(Pnc,Qnc),(Pac,Qac)):
            Pp,Qp=prune(Pm,Qm,md); qi,pj,sc,nc_=topk(Qp,Pp,100); ks.append(pair_key(qi,pj,NP)); del Pp,Qp
        u=np.unique(np.concatenate(ks)); uq=(u//NP).astype(np.int32); up=(u%NP).astype(np.int32); log('union pairs',len(u),'maxdf',md)
        F=pd.DataFrame({'qi':uq,'pj':up,'cos_nw':rowdot(Qw,Pw,uq,up),'cos_aw':rowdot(Qa,Pa,uq,up),'cos_nc':rowdot(Qnc,Pnc,uq,up),'cos_ac':rowdot(Qac,Pac,uq,up),'cos_all':rowdot(Qall,Pall,uq,up)})
        F['fused']=F.cos_nw+2*F.cos_aw+F.cos_nc+2*F.cos_ac; F['label']=np.isin(u,tk).astype(np.int8)
        union_sz=np.bincount(F.qi.values,minlength=nq)
        F=F.sort_values(['qi','fused'],ascending=[True,False],kind='stable').reset_index(drop=True); F['rank']=rank_within(F.qi.values)
        keyall=pair_key(F.qi.values,F.pj.values,NP)
        r={'n_queries':nq,'pool':NP,'true_pairs':int(len(tk)),'maxdf':md,'union_recall_ceiling':float(F.label.sum()/max(len(tk),1)),
           'union_size':dict(mean=float(union_sz.mean()),median=float(np.median(union_sz)),p95=float(np.percentile(union_sz,95)),p99=float(np.percentile(union_sz,99)),max=int(union_sz.max()))}
        bk=np.array([bucket(n) for n in ntrue_tot.reindex(tr.q_id).values]) if len(tr) else np.array([])
        for K in (1,3,5,10,15,20,30):
            sel=np.isin(tk,keyall[F['rank'].values<=K]); nk=np.minimum(union_sz,K)
            r[f'K{K}']=dict(recall=float(sel.mean()),recall_by_bucket={b:float(sel[bk==b].mean()) for b in ('1','2','3-5','6+') if (bk==b).any()},
                             cands=dict(mean=float(nk.mean()),median=float(np.median(nk)),p95=float(np.percentile(nk,95)),p99=float(np.percentile(nk,99)),max=int(nk.max())),reduction_ratio=float(1-nk.mean()/NP))
            log(country,src,f'K={K} recall',round(r[f'K{K}']['recall'],4))
        REP['combos'][f'{country}/{src}']=r
        Fk=F[F['rank']<=KKEEP].copy(); Fk['q_id']=Q.id.values[Fk.qi.values]; Fk['p_id']=P.id.values[Fk.pj.values]; Fk['country']=country; Fk['src']=src; Fk['fold']=Q.fold.values[Fk.qi.values]
        CANDS.append(Fk.drop(columns=['qi','pj']))
        # ---- features on top KFEAT per source
        Fk=Fk[Fk['rank']<=KFEAT].sort_values(['q_id','fused'],ascending=[True,False],kind='stable').reset_index(drop=True)
        Qi=Q.set_index('id'); Pi=P.set_index('id'); idn,ida=idf_tables([Q,P]); X=[]
        for a in range(0,len(Fk),300000): X.append(build(Fk.iloc[a:a+300000],Qi,Pi,idn,ida))
        X=pd.concat(X,ignore_index=True)
        f=X.fused.values; g=pd.Series(f).groupby(Fk.q_id.values)
        X['fused_rank']=g.rank(ascending=False,method='first').values; X['fused_gap_top']=f-g.transform('max').values; X['n_cands']=g.transform('size').values
        X['q_id']=Fk.q_id.values; X['p_id']=Fk.p_id.values; X['label']=Fk.label.values; X['country']=country; X['src']=src; X['fold']=Fk.fold.values
        FEATS.append(X); log(country,src,'features',X.shape); del Pw,Pa,Pnc,Pac,Pall,Qw,Qa,Qnc,Qac,Qall,F,X
CA=pd.concat(CANDS,ignore_index=True); CA.to_parquet(OUT+'cands_top30.parquet')
X=pd.concat(FEATS,ignore_index=True)
g=X.groupby('q_id').fused; X['fused_rank_all']=g.rank(ascending=False,method='first'); X['fused_gap_top_all']=X.fused-g.transform('max')
X.to_parquet(OUT+'feats_top15.parquet'); ntrue_tot.to_frame('ntrue').to_parquet(OUT+'ntrue.parquet')
# entity-level recall ceilings & candidate-count distributions (both sources together)
cnt=X.groupby('q_id').size().reindex(S1.id).fillna(0)
REP['entities']=dict(n=len(S1),cands_top15_per_source_total=dict(mean=float(cnt.mean()),median=float(cnt.median()),p95=float(cnt.quantile(.95)),p99=float(cnt.quantile(.99)),max=int(cnt.max())),
    true_pairs_total=int(len(truth)),pairs_in_top15=int(X.label.sum()),pair_recall_top15=float(X.label.sum()/len(truth)))
json.dump(REP,open(OUT+'blocking_report.json','w'),indent=1,default=float); log('done')
