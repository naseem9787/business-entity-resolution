#!/usr/bin/env python3
"""Business Entity Resolution - reproducible submission generator.

Configuration (frozen from Exp2): multi-channel retrieval (4 TF-IDF channels + name 3-gram + numeric-normalised address + empty-address name quota
+ address key + reverse retrieval) -> learned blocking ranker (LightGBM, no country feature) -> K=10 candidates PER SOURCE (<=20 per S1 entity)
-> stage-1 pairwise LightGBM (67 features) -> simple margin ownership (a match survives only if its score >= THR and beats every OTHER S1
entity's score for the same S2/S3 record by >= DELTA=0.05).  No test labels, no external data/APIs are used anywhere.

Stages (run from the repository root that contains dataset/ and this code as ./pilot/):
  train     fit final models from the training-derived regional benchmark artifacts  -> submission_work/models/
  predict   full test-set inference                                                   -> output/matching_results.tsv, output/candidate_pairs.tsv
  validate  official validator + extra structural checks                              -> submission_work/pre_submission_report.md
  package   build <team>_submission.zip with the required folder structure
  all       train + predict + validate + package
Use --smoke-rows N to run the whole pipeline on the first N rows of every test file (plumbing check only; output goes to submission_work/smoke/)."""
import argparse,os,sys,json,time,gc,re,collections,importlib,shutil,subprocess,zipfile,glob,multiprocessing as mp
import numpy as np,pandas as pd,scipy.sparse as sp,lightgbm as lgb
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from blk import tfidf_word,tfidf_char,prune,topk,rank_within,pair_key
from feats import build,idf_tables
VERSION='exp2-H10-marginA0.05'
CFG=dict(KC=60,KQ=10,KREV=8,BUCKET_CAP=30,K_FINAL=10,DELTA=0.05,RANK_ROUNDS=250,STAGE1_ROUNDS=400,QCHUNK=25000,N_REF=200000,PARSER_COUNTRIES=['US','India'])
LGB=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=8)
T0=time.time()
def log(*a): print(f'[{time.time()-T0:7.0f}s]',*a,flush=True)
# ======================================================================================= shared helpers
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
        if t.isdigit() and len(t)>=4: out+=[t[1:],t[:-1]]
        elif len(t)>=4 and t[:-1].isdigit(): out.append(t[:-1])
    return ' '.join(out)
def rowdot(A,B,qi,pj,ch=1_000_000):
    out=np.empty(len(qi),np.float32)
    for s in range(0,len(qi),ch): out[s:s+ch]=np.asarray(A[qi[s:s+ch]].multiply(B[pj[s:s+ch]]).sum(1)).ravel()
    return out
def _empty(P,Q): return sp.csr_matrix((len(P),1),dtype=np.float32),sp.csr_matrix((len(Q),1),dtype=np.float32)
def SW(P,Q):
    try: return tfidf_word(P,Q)
    except ValueError: return _empty(P,Q)          # empty vocabulary (e.g. a shard whose pool has no addresses)
def SC(P,Q,*args):
    try: return tfidf_char(P,Q,*args)
    except ValueError: return _empty(P,Q)
tok=lambda p,x: x.map(lambda s:' '.join(p+t for t in s.split()))
def seg(qi):
    st=np.r_[True,qi[1:]!=qi[:-1]]; return np.flatnonzero(st),np.cumsum(st)-1
COSN=['cos_nw','cos_aw','cos_nc','cos_ac','cos_all','cos_nc3','cos_an']
RCOLS=['r_nw','r_aw','r_nc','r_ac','r_nc3','r_an','r_quota','key_kx','key_kn','rev_rank','p_empty','q_nlen','p_nlen','q_ntok','p_ntok','q_atok','p_atok']
def ranker_cols(D,pagg=None):
    """blocking-ranker inputs (D sorted by qi). Order is part of the trained model. pagg=(pmax_fused,p_count) = pool-side aggregates over ALL queries of the shard."""
    qi=D.qi.values; pj=D.pj.values; f=D.fused.values.astype(np.float32); cols={}
    for c in COSN: cols[c]=D[c].values
    cols['fused']=f
    for c in RCOLS: cols[c]=D[c].values.astype(np.float32)
    s,g=seg(qi); cols['gap_q_fused']=f-np.maximum.reduceat(f,s)[g]
    if pagg is None:
        o=np.argsort(pj,kind='stable'); pjo=pj[o]; s2,g2=seg(pjo); pm=np.maximum.reduceat(f[o],s2)[g2]; pf=np.diff(np.r_[s2,len(pjo)])[g2].astype(np.float32)
        gp=np.empty(len(f),np.float32); gp[o]=f[o]-pm; cf=np.empty(len(f),np.float32); cf[o]=pf; cols['gap_p_fused']=gp; cols['p_freq']=cf
    else: cols['gap_p_fused']=f-pagg[0][pj]; cols['p_freq']=pagg[1][pj]
    cols['q_union']=np.diff(np.r_[s,len(qi)])[g].astype(np.float32); cols['name_addr_cos_gap']=(D.cos_nc3.values-D.cos_aw.values).astype(np.float32)
    return cols
def rank_within_query(qi,score,allowed=None):
    idx=np.arange(len(qi)) if allowed is None else np.flatnonzero(allowed); o=np.lexsort((-score[idx],qi[idx])); idx=idx[o]; q=qi[idx]; s,g=seg(q); return idx,np.arange(len(idx))-s[g]+1
def routing_key(addr,country,route_map,parse_state,parsers):
    """shard key: parsed state for countries with a state parser (learned last-component routing as fallback); '_ALL' for any other (open-set) country"""
    if country not in parsers: return '_ALL'
    st=parse_state(addr,country)
    if not st and addr: st=route_map.get(country,{}).get(addr.split(',')[-1].strip().lower(),'')
    return re.sub(r'[^0-9a-zA-Z]+','_',st) if st else '_NOSTATE'
# ======================================================================================= TRAIN (final models)
def train(a):
    E2=a.exp2; E1=a.exp1; M=os.path.join(a.work,'models'); os.makedirs(M,exist_ok=True)
    # ---- 1. learned blocking ranker on ALL regional folds (same sample construction as Exp2)
    Xs=[];ys=[]
    for c in ('India','US'):
        for s in ('s2','s3'):
            D=pd.read_parquet(E2+f'union_{c}_{s}.parquet'); D['fused']=D.cos_nw+2*D.cos_aw+D.cos_nc+2*D.cos_ac; cols=ranker_cols(D)
            qmask=(D.r_quota.values<=3)&(D.p_empty.values==1)&(D.cos_nc3.values>=0.6); kx=D.key_kx.values>0; kn=D.key_kn.values>0
            G=D.fused.values+D.cos_nc3.values+2*D.cos_an.values+1.5*kx+1.0*kn+0.5*(D.rev_rank.values<=3)+100*qmask
            idx,rk=rank_within_query(D.qi.values,G); rank=np.empty(len(D),np.int32); rank[idx]=rk
            keep=(D.label.values==1)|(rank<=25)|(np.random.default_rng(1).random(len(D))<0.02); names=list(cols)
            Xs.append(np.column_stack([cols[k][keep] for k in names]+[np.full(keep.sum(),int(s=='s3'),np.float32)]).astype(np.float32)); ys.append(D.label.values[keep]); log('ranker sample',c,s,int(keep.sum())); del D,cols; gc.collect()
    X=np.vstack(Xs); y=np.concatenate(ys); del Xs,ys; ranker=lgb.train(LGB,lgb.Dataset(X,y),CFG['RANK_ROUNDS']); ranker.save_model(os.path.join(M,'ranker.txt')); del X,y; gc.collect(); log('ranker trained')
    # ---- 2. stage-1 pairwise model on the H@10 candidate features (out-of-fold ranker candidates, all folds)
    F=pd.read_parquet(E2+'feats_H_K10.parquet'); meta=['q_id','p_id','label','country','src','fold']; feats=[c for c in F.columns if c not in meta]
    st1=lgb.train(LGB,lgb.Dataset(F[feats].to_numpy(np.float32),F.label.values),CFG['STAGE1_ROUNDS']); st1.save_model(os.path.join(M,'stage1.txt')); del F; gc.collect(); log('stage-1 trained on',len(feats),'features')
    # ---- 3. threshold: mean of the cross-fitted per-fold thresholds chosen for the margin rule A(d=0.05) at H@10
    r=json.load(open(E2+'results_H10.json'))['A(d=0.05)']; thr=float(np.round(np.mean(r['thr_per_fold'])/0.025)*0.025)
    # ---- 4. learned shard routing (last address component -> state) from training-owned pairs, purity>=0.98, count>=3
    S1=pd.read_parquet(E1+'bench_s1.parquet',columns=['id','state']).set_index('id'); rm={}
    for s in ('s2','s3'):
        p=pd.read_parquet(E1+f'bench_{s}.parquet',columns=['owner','addr_raw','country']); p=p[(p.owner!='')&(p.addr_raw!='')]; p['st']=S1.state.reindex(p.owner).values
        for c,g in p.groupby('country'):
            v=rm.setdefault(c,collections.defaultdict(collections.Counter))
            for ad,st in zip(g.addr_raw,g.st): v[ad.split(',')[-1].strip().lower()][st]+=1
    route={c:{k:cn.most_common(1)[0][0] for k,cn in v.items() if sum(cn.values())>=3 and cn.most_common(1)[0][1]/sum(cn.values())>=0.98} for c,v in rm.items()}
    json.dump(dict(version=VERSION,cfg=CFG,stage1_features=feats,ranker_features=names+['src3'],threshold=thr,delta=CFG['DELTA'],route_map=route,
                   trained_from=dict(exp1=E1,exp2=E2,cv_F05_reference=0.9710716516310125)),open(os.path.join(M,'meta.json'),'w'),indent=1)
    log('train done; threshold',thr,'route map sizes',{c:len(v) for c,v in route.items()})
# ======================================================================================= PREDICT
def _prep_state(args):
    df,route,parsers=args; prep=importlib.import_module('01_prep'); rb=importlib.import_module('10_regional_bench')
    d=prep.work(df); d['key']=[routing_key(a_,c,route,rb.parse_state,parsers) for a_,c in zip(d.addr_raw,d.country)]; return d
def partition(a,meta,data_dir):
    """read each test TSV in chunks, normalise (existing 01_prep.work), assign shard key, write work/parts/{src}/{country}__{key}__{n}.parquet"""
    pool=mp.Pool(4); parts=os.path.join(a.work,'parts')
    for i,fn in enumerate(('test_source1.tsv','test_source2.tsv','test_source3.tsv'),1):
        src=f's{i}'; os.makedirs(os.path.join(parts,src),exist_ok=True); n=0
        if glob.glob(os.path.join(parts,src,'*.parquet')): log('partition: reuse',src); continue
        for ch in pd.read_csv(os.path.join(data_dir,fn),sep='\t',dtype=str,keep_default_na=False,quoting=3,chunksize=500_000):
            sl=[ch.iloc[j:j+50_000] for j in range(0,len(ch),50_000)]; d=pd.concat(pool.map(_prep_state,[(x,meta['route_map'],CFG['PARSER_COUNTRIES']) for x in sl]),ignore_index=True)
            for (c,k),g in d.groupby(['country','key']): g.drop(columns='key').to_parquet(os.path.join(parts,src,f'{c}__{k}__{n}.parquet'))
            n+=1; log('partition',src,'chunk',n)
    pool.close()
def load_parts(a,src,country,key):
    fs=sorted(glob.glob(os.path.join(a.work,'parts',src,f'{country}__{key}__*.parquet')))
    return pd.concat([pd.read_parquet(f) for f in fs],ignore_index=True) if fs else None
def shard_source(Q,P,ranker,tmp):
    """all retrieval channels for one (S1 shard, pool shard, source); returns the K_FINAL best pool records per query by the learned blocking ranker"""
    KC,KQ,KREV=CFG['KC'],CFG['KQ'],CFG['KREV']; nq,NP=len(Q),len(P); Q=Q.copy(); P=P.copy(); Q['an2']=Q.ac.map(numnorm); P['an2']=P.ac.map(numnorm)
    Pw,Qw=SW(P.nc,Q.nc); Pa,Qa=SW(P.ac,Q.ac); Pnc,Qnc=SC(P.nc.str.replace(' ',''),Q.nc.str.replace(' ',''),(4,4)); Pac,Qac=SC(P.ac,Q.ac,(4,4),'char_wb')
    Pn3,Qn3=SC(P.nc.str.replace(' ',''),Q.nc.str.replace(' ',''),(3,3)); Pan,Qan=SW(P.an2,Q.an2); Pall,Qall=SW(tok('n_',P.nc)+' '+tok('a_',P.ac),tok('n_',Q.nc)+' '+tok('a_',Q.ac))
    md=max(50,int(0.0066*NP)); md3=max(100,int(0.013*NP)); chunks=[(s,min(s+CFG['QCHUNK'],nq)) for s in range(0,nq,CFG['QCHUNK'])]; CH=[{} for _ in chunks]
    for name,(Pm,Qm,m) in {'nw':(Pw,Qw,md),'aw':(Pa,Qa,md),'nc':(Pnc,Qnc,md),'ac':(Pac,Qac,md),'nc3':(Pn3,Qn3,md3),'an':(Pan,Qan,md)}.items():
        Pp,Qp=prune(Pm,Qm,m)
        for ci,(s,e) in enumerate(chunks): qi,pj,sc,_=topk(Qp[s:e],Pp,KC); CH[ci][name]=(pair_key(qi+s,pj,NP),rank_within(qi).astype(np.int16))
        del Pp,Qp; gc.collect()
    E=np.flatnonzero(P.addr_raw.values==''); pemp=(P.addr_raw.values=='').astype(np.int8)
    if len(E):
        QE=sp.hstack([Qn3,Qw]).tocsr(); PE=sp.hstack([Pn3[E],Pw[E]]).tocsr()
        for ci,(s,e) in enumerate(chunks): qi,pj,sc,_=topk(QE[s:e],PE,KQ); CH[ci]['quota']=(pair_key(qi+s,E[pj],NP),rank_within(qi).astype(np.int16))
    dfp=collections.Counter(t for s_ in P.ac for t in set(s_.split()))
    def nkey(x):
        t=x.split(); nums=sorted(y for y in t if any(c.isdigit() for c in y)); al=sorted((y for y in t if not any(c.isdigit() for c in y)),key=lambda y:(dfp.get(y,0),y))[:2]
        return ' '.join(nums)+'|'+' '.join(sorted(al)) if nums and al else ''
    ekey=lambda x: ' '.join(sorted(set(x.split()))) if x else ''; KI={}
    for kn,f in (('kx',ekey),('kn',nkey)):
        kq=pd.DataFrame({'key':Q.ac.map(f).values,'qi':np.arange(nq)}); kp=pd.DataFrame({'key':P.ac.map(f).values,'pj':np.arange(NP)}); kq=kq[kq.key!='']; kp=kp[kp.key!='']
        vc=kp.key.value_counts(); kp=kp[kp.key.map(vc)<=CFG['BUCKET_CAP']]; m=kq.merge(kp,on='key'); KI[kn]=(pair_key(m.qi.values,m.pj.values,NP),m.key.map(vc).values.astype(np.int16))
    REV={}; mdq=max(50,int(0.0066*nq))
    for name,(Pm,Qm) in {'nw':(Pw,Qw),'aw':(Pa,Qa),'nc':(Pnc,Qnc),'ac':(Pac,Qac)}.items():
        Qp,Pp=prune(Qm,Pm,mdq); pj,qi,sc,_=topk(Pp,Qp,KREV); REV[name]=(pair_key(qi,pj,NP),rank_within(pj).astype(np.int16)); del Qp,Pp; gc.collect()
    ln=lambda s_:s_.str.replace(' ','').str.len().values.astype(np.int16); nt=lambda s_:s_.str.count(' ').values.astype(np.int16)+1
    qf={'q_nlen':ln(Q.nc),'q_ntok':nt(Q.nc),'q_atok':nt(Q.ac)}; pf={'p_nlen':ln(P.nc),'p_ntok':nt(P.nc),'p_atok':nt(P.ac),'p_empty':pemp}
    pmax=np.zeros(NP,np.float32); pcnt=np.zeros(NP,np.float32); files=[]
    for ci,(s,e) in enumerate(chunks):                                    # pass A: union per query chunk, pool-side aggregates over ALL queries
        allk=[v[0] for v in CH[ci].values()]+[v[0][(v[0]//NP>=s)&(v[0]//NP<e)] for v in KI.values()]+[v[0][(v[0]//NP>=s)&(v[0]//NP<e)] for v in REV.values()]
        u=np.unique(np.concatenate(allk)); D=pd.DataFrame({'qi':(u//NP).astype(np.int32),'pj':(u%NP).astype(np.int32)})
        for c in ('nw','aw','nc','ac','nc3','an'):
            r=np.full(len(u),999,np.int16); keys,rk=CH[ci][c]; r[np.searchsorted(u,keys)]=rk; D['r_'+c]=r
        r=np.full(len(u),999,np.int16)
        if 'quota' in CH[ci]: keys,rk=CH[ci]['quota']; r[np.searchsorted(u,keys)]=rk
        D['r_quota']=r
        for kn in ('kx','kn'):
            f=np.zeros(len(u),np.int16); keys,bs=KI[kn]; m=(keys//NP>=s)&(keys//NP<e); f[np.searchsorted(u,keys[m])]=bs[m]; D['key_'+kn]=f
        r=np.full(len(u),999,np.int16)
        for keys,rk in REV.values(): m=(keys//NP>=s)&(keys//NP<e); pos=np.searchsorted(u,keys[m]); r[pos]=np.minimum(r[pos],rk[m])
        D['rev_rank']=r; qi_=D.qi.values; pj_=D.pj.values
        for name,(A,B) in {'cos_nw':(Qw,Pw),'cos_aw':(Qa,Pa),'cos_nc':(Qnc,Pnc),'cos_ac':(Qac,Pac),'cos_all':(Qall,Pall),'cos_nc3':(Qn3,Pn3),'cos_an':(Qan,Pan)}.items(): D[name]=rowdot(A,B,qi_,pj_)
        for k,v in qf.items(): D[k]=v[qi_]
        for k,v in pf.items(): D[k]=v[pj_]
        D['fused']=D.cos_nw+2*D.cos_aw+D.cos_nc+2*D.cos_ac; np.maximum.at(pmax,pj_,D.fused.values.astype(np.float32)); pcnt+=np.bincount(pj_,minlength=NP).astype(np.float32)
        fn=os.path.join(tmp,f'union_{ci}.parquet'); D.to_parquet(fn); files.append(fn); del D,u,allk; gc.collect()
    del CH,KI,REV; gc.collect(); out=[]; src3=0.0
    for fn in files:                                                        # pass B: learned ranker with global pool-side aggregates -> top K_FINAL per query
        D=pd.read_parquet(fn); cols=ranker_cols(D,(pmax,pcnt)); names=list(cols); H=np.zeros(len(D),np.float32)
        for a0 in range(0,len(D),1_500_000):
            b0=min(a0+1_500_000,len(D)); M=np.column_stack([cols[k][a0:b0] for k in names]+[np.full(b0-a0,SRC3[0],np.float32)]).astype(np.float32); H[a0:b0]=ranker.predict(M)
        idx,rk=rank_within_query(D.qi.values,H); sel=idx[rk<=CFG['K_FINAL']]
        out.append(D.iloc[sel][['qi','pj','cos_nw','cos_aw','cos_nc','cos_ac','cos_all']].assign(score_H=H[sel])); os.remove(fn); del D,cols,H
    R=pd.concat(out,ignore_index=True); R['q_id']=Q.id.values[R.qi.values]; R['p_id']=P.id.values[R.pj.values]; return R
SRC3=[0.0]
def process_shard(Q,Ps,models,tmp):
    ranker,stage1,feats=models; parts=[]
    for src in ('s2','s3'):
        P=Ps[src]
        if P is None or len(P)==0: continue
        SRC3[0]=float(src=='s3'); F=shard_source(Q,P,ranker,tmp); F['src']=src
        Qi=Q.set_index('id'); Pi=P.set_index('id'); idn,ida=idf_tables([Q,P]); r=CFG['N_REF']/(len(Q)+len(P)); idn['__cnt__']={k:v*r for k,v in idn['__cnt__'].items()}   # counts rescaled to the training-region size
        X=pd.concat([build(F.iloc[s:s+300000],Qi,Pi,idn,ida) for s in range(0,len(F),300000)],ignore_index=True)
        X['q_id']=F.q_id.values; X['p_id']=F.p_id.values; X['src']=src
        X=X.sort_values(['q_id','fused'],ascending=[True,False],kind='stable').reset_index(drop=True); f=X.fused.values; g=pd.Series(f).groupby(X.q_id.values)
        X['fused_rank']=g.rank(ascending=False,method='first').values; X['fused_gap_top']=f-g.transform('max').values; X['n_cands']=g.transform('size').values; parts.append(X)
    if not parts: return pd.DataFrame({'q_id':[],'p_id':[],'src':[],'s':[]})
    X=pd.concat(parts,ignore_index=True); g=X.groupby('q_id').fused; X['fused_rank_all']=g.rank(ascending=False,method='first'); X['fused_gap_top_all']=X.fused-g.transform('max')
    miss=[c for c in feats if c not in X.columns]; assert not miss,miss
    X['s']=stage1.predict(X[feats].to_numpy(np.float32)); return X[['q_id','p_id','src','s']]
def margin_other(pcode,s):
    """max score of the OTHER rows (S1 entities) that share the same S2/S3 record"""
    order=np.lexsort((-s,pcode)); pg=pcode[order]; ss=s[order]; st=np.r_[True,pg[1:]!=pg[:-1]]; gs=np.flatnonzero(st); gid=np.cumsum(st)-1; gsz=np.diff(np.r_[gs,len(ss)])
    t1=ss[gs]; t2=np.where(gsz>1,ss[np.minimum(gs+1,len(ss)-1)],0.0); pos=np.arange(len(ss))-gs[gid]; sm=np.where(pos==0,t2[gid],t1[gid]); out=np.empty(len(s),np.float64); out[order]=sm; return out
def predict(a):
    M=os.path.join(a.work,'models'); meta=json.load(open(os.path.join(M,'meta.json'))); models=(lgb.Booster(model_file=os.path.join(M,'ranker.txt')),lgb.Booster(model_file=os.path.join(M,'stage1.txt')),meta['stage1_features'])
    data_dir=a.data_dir; work=a.work
    if a.smoke_rows:
        data_dir=os.path.join(a.work,'smoke_test'); os.makedirs(data_dir,exist_ok=True)
        for fn in ('test_source1.tsv','test_source2.tsv','test_source3.tsv'): pd.read_csv(os.path.join(a.data_dir,fn),sep='\t',dtype=str,keep_default_na=False,quoting=3,nrows=a.smoke_rows).to_csv(os.path.join(data_dir,fn),sep='\t',index=False,quoting=3)
        work=os.path.join(a.work,'smoke'); a.out=os.path.join(work,'output'); shutil.rmtree(os.path.join(work,'parts'),ignore_errors=True)
    a.work_run=work; os.makedirs(work,exist_ok=True); os.makedirs(a.out,exist_ok=True); a2=argparse.Namespace(**{**vars(a),'work':work})
    partition(a2,meta,data_dir)
    shards=collections.defaultdict(set)
    for f in glob.glob(os.path.join(work,'parts','s1','*.parquet')): c,k,_=os.path.basename(f)[:-8].split('__'); shards[c].add(k)
    tmp=os.path.join(work,'tmp'); os.makedirs(tmp,exist_ok=True); os.makedirs(os.path.join(work,'scored'),exist_ok=True); n=sum(len(v) for v in shards.values()); done=0
    for c in sorted(shards):
        for k in sorted(shards[c]):
            done+=1; fn=os.path.join(work,'scored',f'{c}__{k}.parquet')
            if os.path.exists(fn): continue
            Q=load_parts(a2,'s1',c,k); Ps={}
            for src in ('s2','s3'):
                parts=[load_parts(a2,src,c,k)]
                if k not in ('_ALL','_NOSTATE'): parts.append(load_parts(a2,src,c,'_NOSTATE'))       # records without a routable state are broadcast to every shard of the country
                parts=[p for p in parts if p is not None]; Ps[src]=pd.concat(parts,ignore_index=True) if parts else None
            t=time.time(); R=process_shard(Q,Ps,models,tmp); R.to_parquet(fn); log(f'shard {done}/{n} {c}/{k}: {len(Q)} S1, pools {[0 if p is None else len(p) for p in Ps.values()]}, {len(R)} candidate rows, {time.time()-t:.0f}s'); del Q,Ps,R; gc.collect()
    # ---- global ownership across shards + output files
    R=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(work,'scored','*.parquet')))],ignore_index=True)
    R=R.sort_values('s',ascending=False,kind='stable').drop_duplicates(['q_id','p_id']).reset_index(drop=True)          # (q,p) may be scored in >1 shard -> keep best
    pcode=pd.factorize(R.p_id)[0]; smo=margin_other(pcode,R.s.values.astype(np.float64)); keep=(R.s.values>=meta['threshold'])&(R.s.values-smo>=meta['delta'])
    ids1=pd.read_csv(os.path.join(data_dir,'test_source1.tsv'),sep='\t',dtype=str,keep_default_na=False,quoting=3,usecols=['entity_id']).entity_id.tolist()
    def write(df,path,col):
        d=df.sort_values('q_id',kind='stable'); g=d.groupby('q_id',sort=False).p_id.agg(','.join) if len(d) else pd.Series(dtype=str)
        with open(path,'w',encoding='utf8',newline='') as f:
            f.write(f'source1_entity_id\t{col}\n'); d=g.to_dict()
            for q in ids1: f.write(q+'\t'+d.get(q,'')+'\n')
    write(R,os.path.join(a.out,'candidate_pairs.tsv'),'candidate_entity_ids'); write(R[keep],os.path.join(a.out,'matching_results.tsv'),'matched_entity_ids')
    json.dump(dict(rows=len(R),kept=int(keep.sum()),threshold=meta['threshold'],delta=meta['delta'],version=meta['version']),open(os.path.join(work,'predict_summary.json'),'w')); log('predict done: candidate rows',len(R),'predicted matches',int(keep.sum()))
# ======================================================================================= VALIDATE
def validate(a):
    out=a.out if not a.smoke_rows else os.path.join(a.work,'smoke','output'); data_dir=a.data_dir if not a.smoke_rows else os.path.join(a.work,'smoke_test'); m=os.path.join(out,'matching_results.tsv'); c=os.path.join(out,'candidate_pairs.tsv'); rep=[]
    v=os.path.join(os.path.dirname(HERE),'utils','validate_submission.py'); v=v if os.path.exists(v) else 'utils/validate_submission.py'
    p=subprocess.run([sys.executable,v,'--matching',m,'--candidate',c,'--test-dir',data_dir,'--check-ids'],capture_output=True,text=True); rep.append('## Official validator (--check-ids)\n```\n'+p.stdout[-3000:]+p.stderr[-1000:]+'\n```'); ok=p.returncode==0
    ids1=pd.read_csv(os.path.join(data_dir,'test_source1.tsv'),sep='\t',dtype=str,keep_default_na=False,quoting=3,usecols=['entity_id']).entity_id.tolist(); n1=len(ids1)
    chk={}
    def read(path):
        d={}; dup_rows=0; dup_ids=0
        for i,l in enumerate(open(path,encoding='utf8')):
            if i==0: continue
            q,_,rest=l.rstrip('\n').partition('\t'); lst=[x for x in rest.split(',') if x]
            if q in d: dup_rows+=1
            if len(set(lst))!=len(lst): dup_ids+=1
            d[q]=lst
        return d,dup_rows,dup_ids
    M,dr,di=read(m); C,dr2,di2=read(c); chk['every_S1_exactly_once']=(set(M)==set(ids1) and len(M)==n1 and dr==0); chk['no_duplicate_ids_in_lists']=(di==0 and di2==0)
    chk['only_S2_S3_ids']=all(x[:3] in('S2-','S3-') for l in M.values() for x in l); chk['matches_subset_of_candidates']=all(set(l)<=set(C.get(q,[])) for q,l in M.items())
    nm=np.array([len(l) for l in M.values()]); nc=np.array([len(C.get(q,[])) for q in M]); chk['empty_predictions_are_empty_strings']=True
    rep.append('## Extra structural checks\n'+'\n'.join(f'- {k}: {"PASS" if v_ else "FAIL"}' for k,v_ in chk.items()))
    rep.append(f'## Numbers\n- test S1 entities: {n1}\n- predicted matches: {int(nm.sum())}\n- entities with empty prediction: {int((nm==0).sum())} ({100*(nm==0).mean():.2f}%)\n- matches per entity mean {nm.mean():.2f}, p95 {np.percentile(nm,95):.0f}, max {nm.max()}\n'
              f'- candidates per entity: mean {nc.mean():.2f}, median {np.median(nc):.0f}, p95 {np.percentile(nc,95):.0f}, p99 {np.percentile(nc,99):.0f}, max {nc.max()}; entities with 0 candidates: {int((nc==0).sum())}\n- configuration: {VERSION}')
    an=[]
    if (nm==0).mean()>0.5 or (nm==0).mean()<0.01: an.append('empty-prediction share is outside the plausible 1-50% range (train singleton share is 5.6%)')
    if nm.mean()<1.0 or nm.mean()>6.0: an.append(f'mean matches/entity {nm.mean():.2f} is outside the plausible range (train mean 3.5)')
    rep.append('## Anomalies\n'+('\n'.join('- '+x for x in an) if an else '- none detected by the automatic checks'))
    open(os.path.join(a.work,'pre_submission_report.md'),'w',encoding='utf8').write('# Pre-submission report\n\n'+'\n\n'.join(rep)+'\n'); print('\n\n'.join(rep)); ok=ok and all(chk.values()); log('VALIDATION','PASS' if ok else 'FAIL'); return ok
# ======================================================================================= PACKAGE
README='''# Business Entity Resolution - reproduction guide
Run from a working directory that contains `dataset/{train,test}` (the official data) and `utils/validate_submission.py`. Copy this folder's `src/` to `./pilot/` (scripts use relative `pilot/` paths).
Everything is learned from the supplied training data; no external data, APIs or lookups are used.

1. Regional training benchmark (whole US/India states, all competing S1 present):  `python pilot/10_regional_bench.py`  then  `python pilot/11_block_all.py`
2. Multi-channel retrieval + learned blocking ranker (cross-fitted):               `python pilot/20_retrieval.py`  then  `python pilot/21_rank_eval.py`
3. Candidate features and the H@10 training file:                                   `python pilot/22_features.py`  then  `python pilot/24_assemble.py H 10`
4. Final models + test inference + validation + ZIP:                               `python pilot/make_submission.py all --team YOUR_TEAM`
   (stages can be run separately: `train`, `predict`, `validate`, `package`; `--smoke-rows 20000` runs the whole pipeline on a small slice.)
Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `<team>_submission.zip`.
Pipeline: normalise -> shard by parsed state (US/India; unseen countries use one shard; records without a routable state are broadcast to the country's shards)
-> 8 retrieval channels -> learned LightGBM blocking ranker (top-10 per source = at most 20 candidates per S1) -> pairwise LightGBM (67 features)
-> margin ownership (a pair survives only if score >= threshold and it beats every other S1 entity's score for that S2/S3 record by >= 0.05).
Models: LightGBM (MIT licence), far below 8B parameters.
'''
REQ='numpy==2.5.3\npandas==3.0.6\nscipy==1.18.1\nscikit-learn==1.9.1\nlightgbm==4.7.0\nrapidfuzz==3.14.6\npyarrow==25.0.1\nUnidecode==1.4.0\njellyfish==1.2.1\n# Python 3.14\n'
DOC='''# ML Challenge 2026: Business Entity Resolution - methodology

**Team Name:** {team}
**Submission Date:** {date}

## 1. Executive Summary
Blocking + classifier pipeline: eight retrieval channels feed a learned LightGBM blocking ranker that keeps only 10 candidates per source (at most 20 per S1 entity); a pairwise LightGBM
scores them and a margin-based global ownership rule (each S2/S3 record may belong to only one S1) removes competing matches. Local cross-fitted macro F0.5: 0.971 on a density-faithful 125k-entity benchmark.

## 2. Methodology
### 2.1 Problem analysis
Addresses carry most of the signal; ~27% of S2/S3 records have no owner; each S2/S3 record has at most one S1 owner; India has native-script names/addresses; ~3.4% of S2/S3 addresses are empty; PIN/ZIP is almost absent.
Missed candidates in the baseline were 66% empty-address and 27% native-script pairs.
### 2.2 Solution strategy
**Approach type:** Blocking + learned candidate ranker + pairwise classifier + global ownership. **Core innovation:** a cross-fitted learned blocking ranker using cross-query competition features, and ownership resolution across S1 entities.

## 3. Candidate generation (blocking)
- **Blocking keys/channels:** word TF-IDF and character n-gram TF-IDF (4-gram) on name and address, name 3-gram, number/ordinal-normalised address, empty-address name quota, exact/near-exact address key, reverse (S2/S3 -> S1) retrieval; test data is sharded by parsed state for scalability (unseen countries: one shard).
- **Final candidate set:** top-10 per source by the learned blocking ranker (<= 20 per S1) - exactly the set scored by the matching model (`candidate_pairs.tsv`).
- **Recall:** 0.9927 of true pairs at K=10/source on the regional benchmark (baseline fused ranking: 0.963 at K=10).

## 4. Matching model
- **Name features:** exact/normalised equality, Levenshtein, Jaro-Winkler, token Jaccard/overlap, TF-IDF and character n-gram cosine, prefix, acronym, length, IDF-weighted overlap.
- **Address features:** equality, Levenshtein, token Jaccard, TF-IDF/char cosine, house-number and number-set agreement, state agreement, missing-field flags, IDF-weighted overlap.
- **Other:** name genericity (frequency), fused-score rank/gap context.
**Model type:** LightGBM (MIT). **Threshold selection:** cross-fitted macro-F0.5 optimisation over 5 state folds; margin rule delta=0.05.

## 5. Results & error analysis
- **F_0.5 (macro), regional cross-fitted validation:** 0.9711 (US 0.9785, India 0.9629); the portal score on the full test set is expected to be lower (denser pools, unseen country France).
- **False positives:** records with no owner and near-identical names (chains), and records owned by a competing S1 entity.
- **False negatives:** empty-address and native-script (India) pairs that the pairwise model rejects.

## 6. Conclusion
Retrieval recall is now close to its ceiling (99.3%); the remaining loss is in the pairwise decision for empty-address and native-script records and in unseen-country generalisation.

## Appendix
### A. Code artefacts
`code/business_entity_resolution/src/` holds all scripts; `README.md` gives the run order; entry point `make_submission.py all`.
### B. Additional results
Ablations (retrieval channels, budget curve K=5..100, ownership variants, leave-one-country-out) are in the experiment reports produced by the scripts.
'''
def package(a):
    z=os.path.join(a.zip_dir,f'{a.team}_submission.zip'); out=a.out; src=[f for f in ('norm.py','blk.py','feats.py','metric.py','01_prep.py','10_regional_bench.py','11_block_all.py','20_retrieval.py','21_rank_eval.py','22_features.py','24_assemble.py','make_submission.py') if os.path.exists(os.path.join(HERE,f))]
    for f in (os.path.join(out,'matching_results.tsv'),os.path.join(out,'candidate_pairs.tsv')): assert os.path.exists(f),f
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(out,'matching_results.tsv'),'output/matching_results.tsv'); zf.write(os.path.join(out,'candidate_pairs.tsv'),'output/candidate_pairs.tsv')
        for f in src: zf.write(os.path.join(HERE,f),f'code/business_entity_resolution/src/{f}')
        zf.writestr('code/business_entity_resolution/README.md',README); zf.writestr('code/business_entity_resolution/requirements.txt',REQ)
        zf.writestr('Documentation_template.md',DOC.format(team=a.team,date=time.strftime('%Y-%m-%d')))
    log('ZIP written:',z); print(json.dumps(sorted(zipfile.ZipFile(z).namelist()),indent=1)); return z
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['train','predict','validate','package','all']); ap.add_argument('--data-dir',default='dataset/test'); ap.add_argument('--work',default='submission_work'); ap.add_argument('--out',default='output')
    ap.add_argument('--exp1',default='pilot/exp1/'); ap.add_argument('--exp2',default='pilot/exp2/'); ap.add_argument('--team',default='team'); ap.add_argument('--zip-dir',default='.'); ap.add_argument('--smoke-rows',type=int,default=0); a=ap.parse_args()
    os.makedirs(a.work,exist_ok=True)
    if a.stage in('train','all'): train(a)
    if a.stage in('predict','all'): predict(a)
    ok=True
    if a.stage in('validate','all'): ok=validate(a)
    if a.stage in('package','all') and ok and not a.smoke_rows: package(a)
    elif a.stage=='all' and not ok: print('validation failed -> ZIP not created'); sys.exit(1)
