"""Exp2 step 1: multi-channel retrieval for every region S1 query. READS pilot/exp1/bench_* (never writes there); writes pilot/exp2/.
Channels (forward S1->pool): nw, aw, nc(char4), ac(char_wb4)   [the four Exp1 channels]
                             nc3 (name char 3-gram, typo tolerant), an (number-normalised address words)
                             quota (name-only retrieval restricted to EMPTY-address pool records), key (exact / near-exact address key)
             reverse (pool->S1): top-8 S1 per pool record on nw/aw/nc/ac
Every pair keeps: 7 cosines, per-channel ranks (999 = absent), key flags, quota rank, reverse rank, label, cheap length features.
usage: python pilot/20_retrieval.py [country/source ...]   (default all four)"""
import sys,os,json,re,time,gc,collections,numpy as np,pandas as pd
sys.path.insert(0,'pilot')
from blk import tfidf_word,tfidf_char,prune,topk,rank_within,pair_key
BENCH='pilot/exp1/'; OUT='pilot/exp2/'; os.makedirs(OUT,exist_ok=True); KC=60; KQ=10; KREV=8; BUCKET_CAP=30; T0=time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
def rowdot(A,B,qi,pj,ch=1_000_000):
    out=np.empty(len(qi),np.float32)
    for s in range(0,len(qi),ch): out[s:s+ch]=np.asarray(A[qi[s:s+ch]].multiply(B[pj[s:s+ch]]).sum(1)).ravel()
    return out
tok=lambda p,x: x.map(lambda s:' '.join(p+t for t in s.split()))
# ---------------- retrieval-side number normalisation (an ADDITIONAL representation; ac stays untouched) ----------------
ORD={w:i+1 for i,w in enumerate('first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth'.split())}
CARD={w:i+1 for i,w in enumerate('one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
_ORDN=re.compile(r'^(\d+)(st|nd|rd|th)$')
def numnorm(a):
    out=[]
    for t in a.split():
        m=_ORDN.match(t)
        if m: t=m.group(1)
        elif t in ORD: t=str(ORD[t])
        elif t in CARD: t=str(CARD[t])
        out.append(t)
        if t.isdigit() and len(t)>=4: out+= [t[1:],t[:-1]]      # dropped-digit variants
        elif len(t)>=4 and t[:-1].isdigit(): out.append(t[:-1])  # 12a -> 12 style suffixes
    return ' '.join(out)
S1=pd.read_parquet(BENCH+'bench_s1.parquet'); POOL={s:pd.read_parquet(BENCH+f'bench_{s}.parquet') for s in ('s2','s3')}
combos=[a for a in sys.argv[1:]] or [f'{c}/{s}' for c in ('India','US') for s in ('s2','s3')]
for combo in combos:
    country,src=combo.split('/'); t0=time.time()
    Q=S1[S1.country==country].reset_index(drop=True); P=POOL[src][POOL[src].country==country].reset_index(drop=True); nq,NP=len(Q),len(P)
    Q['an2']=Q.ac.map(numnorm); P['an2']=P.ac.map(numnorm)
    qmap=dict(zip(Q.id,range(nq))); pm=dict(zip(P.id,range(NP)))
    tr=P[P.owner!=''][['owner','id']]; tr=tr[tr.owner.isin(qmap)]; tk=tr.owner.map(qmap).values.astype(np.int64)*NP+tr.id.map(pm).values.astype(np.int64)
    log(combo,'queries',nq,'pool',NP,'true pairs',len(tk))
    Pw,Qw=tfidf_word(P.nc,Q.nc); Pa,Qa=tfidf_word(P.ac,Q.ac)
    Pnc,Qnc=tfidf_char(P.nc.str.replace(' ',''),Q.nc.str.replace(' ',''),(4,4)); Pac,Qac=tfidf_char(P.ac,Q.ac,(4,4),'char_wb')
    Pn3,Qn3=tfidf_char(P.nc.str.replace(' ',''),Q.nc.str.replace(' ',''),(3,3)); Pan,Qan=tfidf_word(P.an2,Q.an2)
    Pall,Qall=tfidf_word(tok('n_',P.nc)+' '+tok('a_',P.ac),tok('n_',Q.nc)+' '+tok('a_',Q.ac))
    md=max(50,int(0.0066*NP)); md3=max(100,int(0.013*NP)); CH={}      # channel -> (keys, ranks)
    def add(name,qi,pj,rk): CH[name]=(pair_key(qi,pj,NP),rk.astype(np.int16))
    for name,(Pm,Qm,m) in {'nw':(Pw,Qw,md),'aw':(Pa,Qa,md),'nc':(Pnc,Qnc,md),'ac':(Pac,Qac,md),'nc3':(Pn3,Qn3,md3),'an':(Pan,Qan,md)}.items():
        Pp,Qp=prune(Pm,Qm,m); qi,pj,sc,_=topk(Qp,Pp,KC); add(name,qi,pj,rank_within(qi)); del Pp,Qp,qi,pj,sc; gc.collect(); log(' channel',name,len(CH[name][0]))
    # ---- quota channel: name-only retrieval against EMPTY-address pool records (small pool -> no pruning needed)
    E=np.flatnonzero(P.addr_raw.values==''); pemp=(P.addr_raw.values=='').astype(np.int8)
    if len(E):
        import scipy.sparse as sp
        qi,pj,sc,_=topk(sp.hstack([Qn3,Qw]).tocsr(),sp.hstack([Pn3[E],Pw[E]]).tocsr(),KQ); add('quota',qi,E[pj],rank_within(qi)); del qi,pj,sc
    log(' quota',len(CH.get('quota',([],))[0]),'empty-address pool records',len(E))
    # ---- address key channel: exact token-set key and near-exact key (numbers + 2 rarest alpha tokens); big buckets skipped
    dfp=collections.Counter(t for s in P.ac for t in set(s.split()))
    def nkey(a):
        t=a.split(); nums=sorted(x for x in t if any(c.isdigit() for c in x)); al=sorted((x for x in t if not any(c.isdigit() for c in x)),key=lambda x:(dfp.get(x,0),x))[:2]
        return ' '.join(nums)+'|'+' '.join(sorted(al)) if nums and al else ''
    ekey=lambda a: ' '.join(sorted(set(a.split()))) if a else ''
    kinfo={}
    for kn,f in (('kx',ekey),('kn',nkey)):
        kq=pd.DataFrame({'key':Q.ac.map(f).values,'qi':np.arange(nq)}); kp=pd.DataFrame({'key':P.ac.map(f).values,'pj':np.arange(NP)})
        kq=kq[kq.key!='']; kp=kp[kp.key!='']; vc=kp.key.value_counts(); kp=kp[kp.key.map(vc)<=BUCKET_CAP]; m=kq.merge(kp,on='key'); kinfo[kn]=(pair_key(m.qi.values,m.pj.values,NP),m.key.map(vc).values.astype(np.int16)); log(' ',kn,'pairs',len(m))
    # ---- reverse retrieval: for every pool record its top-KREV S1 entities (S1 side is the 'pool' for pruning)
    mdq=max(50,int(0.0066*nq)); REV={}
    for name,(Pm,Qm) in {'nw':(Pw,Qw),'aw':(Pa,Qa),'nc':(Pnc,Qnc),'ac':(Pac,Qac)}.items():
        Qp,Pp=prune(Qm,Pm,mdq); pj,qi,sc,_=topk(Pp,Qp,KREV); REV[name]=(pair_key(qi,pj,NP),rank_within(pj).astype(np.int16)); del Qp,Pp,pj,qi,sc; gc.collect()
    log(' reverse',{k:len(v[0]) for k,v in REV.items()})
    # ---- union + per-pair features
    allk=[v[0] for v in CH.values()]+[v[0] for v in kinfo.values()]+[v[0] for v in REV.values()]; u=np.unique(np.concatenate(allk)); del allk
    uq=(u//NP).astype(np.int32); up=(u%NP).astype(np.int32); log(' union pairs',len(u),'per query',round(len(u)/nq,1))
    D=pd.DataFrame({'qi':uq,'pj':up}); del uq,up
    for c in ('nw','aw','nc','ac','nc3','an'):
        r=np.full(len(u),999,np.int16); keys,rk=CH[c]; r[np.searchsorted(u,keys)]=rk; D['r_'+c]=r
    r=np.full(len(u),999,np.int16)
    if 'quota' in CH: keys,rk=CH['quota']; r[np.searchsorted(u,keys)]=rk
    D['r_quota']=r
    for kn in ('kx','kn'):
        f=np.zeros(len(u),np.int16); keys,bs=kinfo[kn]; f[np.searchsorted(u,keys)]=bs; D['key_'+kn]=f      # bucket size (0 = no key match)
    r=np.full(len(u),999,np.int16)
    for name,(keys,rk) in REV.items(): pos=np.searchsorted(u,keys); r[pos]=np.minimum(r[pos],rk)
    D['rev_rank']=r
    qi=D.qi.values; pj=D.pj.values
    for name,(A,B) in {'cos_nw':(Qw,Pw),'cos_aw':(Qa,Pa),'cos_nc':(Qnc,Pnc),'cos_ac':(Qac,Pac),'cos_all':(Qall,Pall),'cos_nc3':(Qn3,Pn3),'cos_an':(Qan,Pan)}.items(): D[name]=rowdot(A,B,qi,pj)
    D['label']=np.isin(u,tk).astype(np.int8)
    ln=lambda s:s.str.replace(' ','').str.len().values.astype(np.int16); nt=lambda s:s.str.count(' ').values.astype(np.int16)+1
    qf={'q_nlen':ln(Q.nc),'q_ntok':nt(Q.nc),'q_atok':nt(Q.ac)}; pf={'p_nlen':ln(P.nc),'p_ntok':nt(P.nc),'p_atok':nt(P.ac),'p_empty':pemp}
    for k,v in qf.items(): D[k]=v[qi]
    for k,v in pf.items(): D[k]=v[pj]
    D.to_parquet(OUT+f'union_{country}_{src}.parquet'); pd.DataFrame({'id':Q.id,'fold':Q.fold}).to_parquet(OUT+f'qinfo_{country}.parquet'); pd.DataFrame({'id':P.id,'owner':P.owner,'addr_empty':pemp}).to_parquet(OUT+f'pinfo_{country}_{src}.parquet')
    json.dump(dict(nq=nq,NP=NP,true_pairs=int(len(tk)),union_pairs=int(len(u)),union_per_query=float(len(u)/nq),union_recall_ceiling=float(D.label.sum()/len(tk)),
                   channel_sizes={k:int(len(v[0])) for k,v in CH.items()},key_pairs={k:int(len(v[0])) for k,v in kinfo.items()},reverse_pairs={k:int(len(v[0])) for k,v in REV.items()},seconds=time.time()-t0),open(OUT+f'retrieval_meta_{country}_{src}.json','w'),indent=1)
    log(combo,'saved; union ceiling',round(D.label.sum()/len(tk),4),'| seconds',round(time.time()-t0)); del D,Pw,Qw,Pa,Qa,Pnc,Qnc,Pac,Qac,Pn3,Qn3,Pan,Qan,Pall,Qall,CH,REV,kinfo,u; gc.collect()
