import numpy as np, pandas as pd, collections, math
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
US_ST=set('al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy'.split())
IN_ST=set('up mp ap tn wb hp jk mh rj gj dl hr ka ts kl pb br od as jh cg uk ga'.split())
STATES=US_ST|IN_ST
def cp(a,b,scorer):
    return process.cpdist(a,b,scorer=scorer,workers=4,dtype=np.float32)
def jac(a,b):
    return np.array([len(x&y)/len(x|y) if (x or y) else 0. for x,y in zip(a,b)],np.float32)
def build(F,Q,P,idf_q=None,idf_p=None):
    """F: candidate pairs (q_id,p_id,cos_*,label,...). Q,P: normalised record frames indexed by id. returns feature frame aligned with F"""
    q=Q.loc[F.q_id.values]; p=P.loc[F.p_id.values]
    out={}
    def col(n,v): out[n]=np.asarray(v,np.float32)
    nc_q,nc_p=q.nc.tolist(),p.nc.tolist(); nt_q,nt_p=q.nt.tolist(),p.nt.tolist(); nn_q,nn_p=q.nn.tolist(),p.nn.tolist()
    ac_q,ac_p=q.ac.tolist(),p.ac.tolist(); an_q,an_p=q.an.tolist(),p.an.tolist()
    ns_q=[x.replace(' ','') for x in nc_q]; ns_p=[x.replace(' ','') for x in nc_p]
    # ---------------- NAME ----------------
    col('n_eq_nn',[a==b for a,b in zip(nn_q,nn_p)]); col('n_eq_nc',[a==b for a,b in zip(nc_q,nc_p)]); col('n_eq_ns',[a==b for a,b in zip(ns_q,ns_p)])
    col('n_lev_nc',cp(nc_q,nc_p,Levenshtein.normalized_similarity)); col('n_lev_nt',cp(nt_q,nt_p,Levenshtein.normalized_similarity)); col('n_lev_nn',cp(nn_q,nn_p,Levenshtein.normalized_similarity))
    col('n_lev_ns',cp(ns_q,ns_p,Levenshtein.normalized_similarity))
    col('n_jw_nc',cp(nc_q,nc_p,JaroWinkler.similarity)); col('n_jw_nn',cp(nn_q,nn_p,JaroWinkler.similarity))
    col('n_tsr',cp(nc_q,nc_p,fuzz.token_set_ratio)/100); col('n_tsort',cp(nc_q,nc_p,fuzz.token_sort_ratio)/100); col('n_partial',cp(ns_q,ns_p,fuzz.partial_ratio)/100)
    tq=[set(x.split()) for x in nc_q]; tp=[set(x.split()) for x in nc_p]
    col('n_jac',jac(tq,tp)); col('n_ov',[len(x&y) for x,y in zip(tq,tp)])
    col('n_ovq',[len(x&y)/max(len(x),1) for x,y in zip(tq,tp)]); col('n_ovp',[len(x&y)/max(len(y),1) for x,y in zip(tq,tp)])
    if idf_q is not None:
        col('n_widf',[sum(idf_q.get(t,12) for t in x&y)/max(sum(idf_q.get(t,12) for t in x),1e-6) for x,y in zip(tq,tp)])
        col('n_widf_p',[sum(idf_q.get(t,12) for t in x&y)/max(sum(idf_q.get(t,12) for t in y),1e-6) for x,y in zip(tq,tp)])
        col('n_maxidf_unshared_q',[max([idf_q.get(t,12) for t in x-y],default=0) for x,y in zip(tq,tp)])
    def cpl(a,b):
        n=min(len(a),len(b)); i=0
        while i<n and a[i]==b[i]: i+=1
        return i/max(n,1)
    col('n_prefix',[cpl(a,b) for a,b in zip(ns_q,ns_p)])
    ini=lambda s:''.join(w[0] for w in s.split())
    col('n_acr',[(len(a)>=2 and (ini(b_full)==a or ini(a_full)==b)) for a,b,a_full,b_full in zip(ns_q,ns_p,nc_q,nc_p)])
    col('n_contains',[(a in b or b in a) if a and b else 0 for a,b in zip(ns_q,ns_p)])
    col('n_len_q',[len(x) for x in nc_q]); col('n_len_p',[len(x) for x in nc_p]); col('n_lendiff',[abs(len(a)-len(b)) for a,b in zip(nc_q,nc_p)]); col('n_lenratio',[min(len(a),len(b))/max(len(a),len(b),1) for a,b in zip(nc_q,nc_p)])
    col('n_ntok_q',[len(x) for x in tq]); col('n_ntok_p',[len(x) for x in tp])
    col('n_script_diff',[a.isascii()!=b.isascii() for a,b in zip(q.name_raw,p.name_raw)])
    col('n_cos_w',F.cos_nw.values); col('n_cos_c',F.cos_nc.values)
    # ---------------- ADDRESS ----------------
    col('a_eq',[a==b and a!='' for a,b in zip(ac_q,ac_p)])
    col('a_lev',cp(ac_q,ac_p,Levenshtein.normalized_similarity)); col('a_jw',cp(ac_q,ac_p,JaroWinkler.similarity))
    col('a_tsr',cp(ac_q,ac_p,fuzz.token_set_ratio)/100); col('a_tsort',cp(ac_q,ac_p,fuzz.token_sort_ratio)/100); col('a_partial',cp(ac_q,ac_p,fuzz.partial_ratio)/100)
    aq=[x.split() for x in ac_q]; ap=[x.split() for x in ac_p]; sq=[set(x) for x in aq]; sp_=[set(x) for x in ap]
    isn=lambda t:any(c.isdigit() for c in t)
    nq=[set(t for t in x if isn(t)) for x in sq]; npp=[set(t for t in x if isn(t)) for x in sp_]
    alq=[x-n for x,n in zip(sq,nq)]; alp=[x-n for x,n in zip(sp_,npp)]
    col('a_jac',jac(sq,sp_)); col('a_jac_alpha',jac(alq,alp)); col('a_jac_num',jac(nq,npp)); col('a_num_ov',[len(x&y) for x,y in zip(nq,npp)])
    col('a_num_missing',[(len(x)==0)|(len(y)==0) for x,y in zip(nq,npp)])
    col('a_num_conflict',[(len(x)>0 and len(y)>0 and len(x&y)==0) for x,y in zip(nq,npp)])
    def first_num(x):
        for t in x:
            if isn(t): return t
        return ''
    fq=[first_num(x) for x in aq]; fp=[first_num(x) for x in ap]
    col('a_firstnum_eq',[a==b and a!='' for a,b in zip(fq,fp)])
    col('a_house_eq',[len(x)>0 and len(y)>0 and x[0]==y[0] and isn(x[0]) for x,y in zip(aq,ap)])
    col('a_ovq',[len(x&y)/max(len(x),1) for x,y in zip(sq,sp_)]); col('a_ovp',[len(x&y)/max(len(y),1) for x,y in zip(sq,sp_)])
    if idf_p is not None:
        col('a_widf',[sum(idf_p.get(t,12) for t in x&y)/max(sum(idf_p.get(t,12) for t in x),1e-6) for x,y in zip(sq,sp_)])
        col('a_widf_p',[sum(idf_p.get(t,12) for t in x&y)/max(sum(idf_p.get(t,12) for t in y),1e-6) for x,y in zip(sq,sp_)])
    stq=[x&STATES for x in sq]; stp=[x&STATES for x in sp_]
    col('a_state_eq',[len(x&y)>0 for x,y in zip(stq,stp)]); col('a_state_conflict',[(len(x)>0 and len(y)>0 and len(x&y)==0) for x,y in zip(stq,stp)])
    col('a_miss_q',[len(x)==0 for x in aq]); col('a_miss_p',[len(x)==0 for x in ap]); col('a_len_q',[len(x) for x in aq]); col('a_len_p',[len(x) for x in ap])
    col('a_cos_w',F.cos_aw.values); col('a_cos_c',F.cos_ac.values)
    # ---------------- OTHER / CONTEXT ----------------
    col('cos_all',F.cos_all.values); col('src3',(F.src.values=='s3'))
    col('name_freq_q',np.log1p(q.nc.map(idf_q['__cnt__'].get).fillna(1).values) if idf_q is not None else 0)
    col('name_freq_p',np.log1p(p.nc.map(idf_q['__cnt__'].get).fillna(1).values) if idf_q is not None else 0)
    X=pd.DataFrame(out)
    f=(F.cos_nw+2*F.cos_aw+F.cos_nc+2*F.cos_ac).values; X['fused']=f
    g=pd.Series(f).groupby(F.q_id.values)
    X['fused_rank']=g.rank(ascending=False,method='first').values; X['fused_gap_top']=f-g.transform('max').values; X['n_cands']=g.transform('size').values
    return X
def idf_tables(frames):
    """document frequency tables over tokens of core names / canonical addresses from the given normalised frames (used for rarity/genericity features)"""
    cn=collections.Counter(); ca=collections.Counter(); nn=collections.Counter(); N=0
    for d in frames:
        N+=len(d)
        for s in d.nc: cn.update(set(s.split()))
        for s in d.ac: ca.update(set(s.split()))
        nn.update(d.nc)
    idf_n={t:math.log(N/(1+c)) for t,c in cn.items()}; idf_n['__cnt__']=dict(nn)
    idf_a={t:math.log(N/(1+c)) for t,c in ca.items()}
    return idf_n,idf_a
