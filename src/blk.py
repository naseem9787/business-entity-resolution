import numpy as np, pandas as pd, scipy.sparse as sp, re, gc
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
KS=[5,10,20,30,50,100]
def tfidf_word(pool,qry,binary=False):
    v=TfidfVectorizer(token_pattern=r'\S+',lowercase=False,sublinear_tf=True,dtype=np.float32,binary=binary)
    P=v.fit_transform(pool); return P, v.transform(qry)
def tfidf_char(pool,qry,ng=(3,3),an='char'):
    v=TfidfVectorizer(analyzer=an,ngram_range=ng,lowercase=False,sublinear_tf=True,dtype=np.float32)
    P=v.fit_transform(pool); return P, v.transform(qry)
def prune(P,Q,maxdf):
    """zero-out columns whose pool document frequency > maxdf"""
    df=np.diff(P.tocsc().indptr); keep=(df<=maxdf).astype(np.float32)
    D=sp.diags(keep); return (P@D).tocsr(),(Q@D).tocsr()
def topk(Q,P,K=100,chunk=4000,min_score=0.0):
    """exact sparse cosine top-K per query row (rows already L2-normalised). returns qi,pj,score (sorted by score desc within q) and #nonzero candidates per q"""
    PT=P.T.tocsr(); qi_l=[];pj_l=[];sc_l=[]; ncand=np.zeros(Q.shape[0],np.int32)
    for s in range(0,Q.shape[0],chunk):
        S=(Q[s:s+chunk]@PT).tocsr(); S.sum_duplicates()
        ip,ix,dt=S.indptr,S.indices,S.data
        for r in range(S.shape[0]):
            a,b=ip[r],ip[r+1]; n=b-a; ncand[s+r]=n
            if n==0: continue
            d=dt[a:b]; j=ix[a:b]
            if n>K: sel=np.argpartition(-d,K)[:K]; d=d[sel]; j=j[sel]
            o=np.argsort(-d,kind='stable'); qi_l.append(np.full(len(o),s+r,np.int32)); pj_l.append(j[o]); sc_l.append(d[o])
        del S; gc.collect()
    if not qi_l: return np.zeros(0,np.int32),np.zeros(0,np.int32),np.zeros(0,np.float32),ncand
    return np.concatenate(qi_l),np.concatenate(pj_l),np.concatenate(sc_l),ncand
def rank_within(qi):
    """rank (1-based) of each row within its query, assuming rows are grouped by qi in ranked order"""
    start=np.r_[0,np.flatnonzero(np.diff(qi))+1]; cnt=np.diff(np.r_[start,len(qi)])
    return np.arange(len(qi))-np.repeat(start,cnt)+1
def pair_key(qi,pj,NP): return qi.astype(np.int64)*NP+pj
def recall_curve(qi,pj,rank,true_keys,NP,Ks=KS):
    k=pair_key(qi,pj,NP); hit=np.isin(k,true_keys)
    return {K:float(np.isin(true_keys,k[rank<=K]).mean()) for K in Ks}
