import sys,json,time,re,numpy as np,pandas as pd,scipy.sparse as sp
sys.path.insert(0,'pilot')
from blk import *
import jellyfish
country,src,nq=sys.argv[1],sys.argv[2],int(sys.argv[3])
T0=time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
s1=pd.read_parquet('pilot/s1_n.parquet'); P=pd.read_parquet(f'pilot/{src}_n.parquet'); gp=pd.read_parquet('pilot/gt_pairs.parquet')
Q=s1[s1.country==country].sample(frac=1,random_state=0).head(nq).reset_index(drop=True); P=P[P.country==country].reset_index(drop=True)
NP=len(P); qmap=dict(zip(Q.id,range(len(Q)))); pmap=dict(zip(P.id,range(NP)))
g=gp[gp.m.str.startswith(src.upper())]; g=g[g.s1.isin(qmap)]
tpm=g.m.map(pmap); miss=int(tpm.isna().sum()); g=g[tpm.notna().values]
tq=g.s1.map(qmap).values.astype(np.int64); tp=g.m.map(pmap).values.astype(np.int64)
tk=tq*NP+tp; nm=np.bincount(tq,minlength=len(Q))            # true matches (this source) per query
log(country,src,'queries',len(Q),'pool',NP,'true pairs',len(tk),'true pairs whose match is in a different-country pool:',miss)
R={}
def stats(cnt):
    c=np.asarray(cnt,float); return dict(mean=c.mean(),median=float(np.median(c)),p90=np.percentile(c,90),p95=np.percentile(c,95),p99=np.percentile(c,99),max=c.max())
def rowdot(A,B,qi,pj,ch=1_000_000):
    out=np.empty(len(qi),np.float32)
    for s in range(0,len(qi),ch): out[s:s+ch]=np.asarray(A[qi[s:s+ch]].multiply(B[pj[s:s+ch]]).sum(1)).ravel()
    return out
def bucket_recall(hit):
    lab=np.select([nm[tq]==1,nm[tq]==2,nm[tq]<=5],['1','2','3-5'],'6+')
    return pd.DataFrame({'l':lab,'h':hit}).groupby('l').h.mean().round(4).to_dict()
# ---------- hash-join blockers (recall = true pairs sharing key; cand = size of key bucket in pool) ----------
def run_join(name,f):
    kq=[f(r) for r in Q.itertuples()]; kp=[f(r) for r in P.itertuples()]
    vc=pd.Series(kp); vc=vc[vc!=''].value_counts(); per_q=np.array([vc.get(k,0) if k else 0 for k in kq],float)
    hit=np.array([bool(kq[a]) and kq[a]==kp[b] for a,b in zip(tq,tp)])
    R[name]=dict(kind='join',recall=float(hit.mean()),recall_by_nmatch=bucket_recall(hit),cand=stats(per_q),pairs=float(per_q.sum()))
    log(name,round(hit.mean(),4),round(R[name]['cand']['mean'],1))
mph=lambda t: jellyfish.metaphone(t) or t
pin=re.compile(r'(?<!\d)(\d{6}|\d{5})(?!\d)')
def pinf(r):
    m=pin.findall(r.addr_raw); return m[-1] if m else ''
run_join('exact_name',lambda r:r.nc)
run_join('exact_addr',lambda r:r.ac)
run_join('name_prefix4',lambda r:r.nc.replace(' ','')[:4])
run_join('name_prefix6',lambda r:r.nc.replace(' ','')[:6])
run_join('sorted_token_sig',lambda r:' '.join(sorted(set(r.nc.split()))))
run_join('phonetic_name',lambda r:' '.join(sorted(mph(t) for t in set(r.nc.split()))))
run_join('postal_pin',pinf); R['postal_pin']['coverage_queries']=float(np.mean([bool(pinf(r)) for r in Q.itertuples()]))
run_join('addr_number_sig',lambda r:' '.join(sorted(t for t in r.ac.split() if any(c.isdigit() for c in t))))
# ---------- sparse blockers ----------
mats={}
def sparse_eval(name,Pm,Qm,maxdf,keep=False):
    Pp,Qp=prune(Pm,Qm,maxdf)
    qi,pj,sc,ncand=topk(Qp,Pp,100); rk=rank_within(qi)
    anyhit=rowdot(Qp,Pp,tq,tp)>0
    kk=pair_key(qi,pj,NP); rc=recall_curve(qi,pj,rk,tk,NP)
    R[name]=dict(kind='sparse',recall_any_shared=float(anyhit.mean()),recall_by_nmatch_any=bucket_recall(anyhit),recall_at_K=rc,cand_any_shared=stats(ncand),
                 recall_at_K_by_nmatch={K:bucket_recall(np.isin(tk,kk[rk<=K])) for K in (10,30)})
    log(name,'any',round(anyhit.mean(),4),'cand mean',round(ncand.mean(),1),'R@K',{k:round(v,4) for k,v in rc.items()})
    if keep: mats[name]=(qi,pj,sc)
tok=lambda p,x: x.map(lambda s:' '.join(p+t for t in s.split()))
Pw,Qw=tfidf_word(P.nc,Q.nc); Pa,Qa=tfidf_word(P.ac,Q.ac)
Pnc,Qnc=tfidf_char(P.nc.str.replace(' ',''),Q.nc.str.replace(' ',''),(4,4)); Pac,Qac=tfidf_char(P.ac,Q.ac,(4,4),'char_wb')
Pall,Qall=tfidf_word(tok('n_',P.nc)+' '+tok('a_',P.ac),tok('n_',Q.nc)+' '+tok('a_',Q.ac))
for md in (200,2000,10000): sparse_eval(f'name_word_tfidf_maxdf{md}',Pw,Qw,md,keep=(md==2000))
for md in (200,2000,10000): sparse_eval(f'addr_word_tfidf_maxdf{md}',Pa,Qa,md,keep=(md==2000))
for md in (2000,10000): sparse_eval(f'name_char4_maxdf{md}',Pnc,Qnc,md,keep=(md==2000))
for md in (2000,10000): sparse_eval(f'addr_char4_maxdf{md}',Pac,Qac,md,keep=(md==2000))
sparse_eval('name+addr_word_tfidf_maxdf2000',Pall,Qall,2000)
# ---------- fused union of the 4 retrievers, ranked by a fused cosine ----------
u=np.unique(np.concatenate([pair_key(*mats[k][:2],NP) for k in mats])); uq=(u//NP).astype(np.int32); up=(u%NP).astype(np.int32); log('union pairs',len(u))
F=pd.DataFrame({'qi':uq,'pj':up,'cos_nw':rowdot(Qw,Pw,uq,up),'cos_aw':rowdot(Qa,Pa,uq,up),'cos_nc':rowdot(Qnc,Pnc,uq,up),'cos_ac':rowdot(Qac,Pac,uq,up),'cos_all':rowdot(Qall,Pall,uq,up)})
F['label']=np.isin(u,tk).astype(np.int8)
for nme,w in {'fuse_sum4':dict(cos_nw=1,cos_aw=1,cos_nc=1,cos_ac=1),'fuse_addr_heavy':dict(cos_nw=1,cos_aw=2,cos_nc=1,cos_ac=2),'fuse_name_only':dict(cos_nw=1,cos_nc=1),'fuse_addr_only':dict(cos_aw=1,cos_ac=1)}.items():
    F['f']=sum(F[c]*v for c,v in w.items()); Fs=F.sort_values(['qi','f'],ascending=[True,False]); rk=rank_within(Fs.qi.values)
    kk=pair_key(Fs.qi.values,Fs.pj.values,NP)
    R[nme]=dict(kind='fused',recall_at_K=recall_curve(Fs.qi.values,Fs.pj.values,rk,tk,NP),mean_cands_at_K={K:float(np.bincount(Fs.qi.values[rk<=K],minlength=len(Q)).mean()) for K in KS},
                recall_at_K_by_nmatch={K:bucket_recall(np.isin(tk,kk[rk<=K])) for K in (5,10,20,30)})
    log(nme,{k:round(v,4) for k,v in R[nme]['recall_at_K'].items()})
R['union_recall_ceiling']=float(F.label.sum()/len(tk)); R['meta']=dict(country=country,src=src,nq=len(Q),NP=NP,true_pairs=int(len(tk)),singletons=int((nm==0).sum()),union_pairs=len(F))
F['q_id']=Q.id.values[F.qi.values]; F['p_id']=P.id.values[F.pj.values]; F['country']=country; F['src']=src
F.drop(columns=['f']).to_parquet(f'pilot/cand_{country}_{src}.parquet'); json.dump(R,open(f'pilot/report_block_{country}_{src}.json','w'),default=float,indent=1); log('done')
