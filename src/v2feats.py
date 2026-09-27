"""v2 evidence features, computed per SHARD (= the set of S1 queries and pool records scored together), identically at training and inference.
X : candidate feature frame (must contain q_id,p_id,src,n_lev_nc,a_cos_w,fused,a_miss_p) for ALL rows of the shard (both sources)
Q : normalised S1 frame of the shard (id,country,nc,ac,an,nn)      P : normalised pool frame of the shard, both sources, with column src
Groups: x_name_* (name ambiguity / competition), x_addr_* (address rarity), x_sib_* (address-confirmed sibling agreement), x_tr_* (learned native->Latin map)."""
import json,numpy as np,pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
_TM=None
def _tm(path):
    global _TM
    if _TM is None: T=json.load(open(path,encoding='utf8')); _TM=({t:v[0] for t,v in T['name'].items()},{t:v[0] for t,v in T['addr'].items()})
    return _TM
def _other_best(g,v):
    o=np.lexsort((-v,g)); gs=g[o]; vs=v[o]; st=np.r_[True,gs[1:]!=gs[:-1]]; idx=np.flatnonzero(st); gid=np.cumsum(st)-1; sz=np.diff(np.r_[idx,len(vs)])
    t1=vs[idx]; t2=np.where(sz>1,vs[np.minimum(idx+1,len(vs)-1)],0.0); pos=np.arange(len(vs))-idx[gid]; res=np.where(pos==0,t2[gid],t1[gid]); out=np.empty(len(v)); out[o]=res; return out
isnat=lambda t:any(ord(c)>127 for c in t)
def add(X,Q,P,translit_path,legal):
    n=len(X); NEW={}; Qi=Q.set_index('id'); Pi=P.set_index('id')
    qnk=Q.country+'|'+Q.nc; qtk=Q.country+'|'+Q.nc.map(lambda s:' '.join(sorted(set(s.split())))); qak=Q.country+'|'+Q.ac
    cq_n=pd.Series(qnk.map(qnk.value_counts()).values,index=Q.id); cq_t=pd.Series(qtk.map(qtk.value_counts()).values,index=Q.id); cq_a=pd.Series(qak.map(qak.value_counts()).values,index=Q.id)
    pnk=P.src+'|'+P.nc; pak=P.src+'|'+P.ac; cp_n=pd.Series(pnk.map(pnk.value_counts()).values,index=P.id); cp_a=pd.Series(np.where(P.ac.values!='',pak.map(pak.value_counts()).values,0),index=P.id)
    qs1n=cq_n.reindex(X.q_id).values; qs1t=cq_t.reindex(X.q_id).values; qs1a=cq_a.reindex(X.q_id).values; ppn=cp_n.reindex(X.p_id).values; ppa=cp_a.reindex(X.p_id).values
    NEW['x_name_q_s1_dup']=np.log1p(qs1n); NEW['x_name_q_s1_tokset_dup']=np.log1p(qs1t); NEW['x_name_p_pool_dup']=np.log1p(ppn); NEW['x_name_dup_diff']=np.log1p(ppn)-np.log1p(qs1n)
    NEW['x_addr_q_s1_freq']=np.log1p(qs1a); NEW['x_addr_p_pool_freq']=np.log1p(ppa); NEW['x_addr_freq_diff']=np.log1p(ppa)-np.log1p(qs1a); NEW['x_addr_rare_unique']=((qs1a<=1)&(ppa<=1)&(X.a_miss_p.values==0)).astype(np.float32)
    gp=pd.factorize(X.p_id)[0]; gq=pd.factorize(X.q_id)[0]
    for m,vals in (('name',X.n_lev_nc.values.astype(float)),('addr',X.a_cos_w.values.astype(float)),('fused',X.fused.values.astype(float))):
        ob=_other_best(gp,vals); NEW[f'x_name_p_other_best_{m}']=ob; NEW[f'x_name_p_margin_{m}']=vals-ob
    NEW['x_name_p_rows']=np.bincount(gp)[gp].astype(np.float32); ob=_other_best(gq,X.n_lev_nc.values.astype(float)); NEW['x_name_q_other_best_name']=ob; NEW['x_name_q_margin_name']=X.n_lev_nc.values-ob
    conf=((X.a_cos_w.values>=0.8)&(X.a_miss_p.values==0)); pn=Pi.nn.reindex(X.p_id).values
    T=pd.DataFrame({'q_id':X.q_id.values,'p_id':X.p_id.values,'pn':pn,'row':np.arange(n)}); S=T[conf].rename(columns={'p_id':'sp','pn':'sn'})[['q_id','sp','sn']]
    sib_max=np.full(n,np.nan,np.float32); sib_n=np.zeros(n,np.float32); sib_eq=np.zeros(n,np.float32)
    for c in range(8):
        tm=(gq%8)==c; M=T[tm].merge(S,on='q_id'); M=M[M.p_id!=M.sp]
        if len(M):
            sim=process.cpdist(M.pn.tolist(),M.sn.tolist(),scorer=Levenshtein.normalized_similarity,workers=4,dtype=np.float32); g=pd.DataFrame({'row':M.row.values,'sim':sim}).groupby('row').sim.agg(['max','size'])
            sib_max[g.index.values]=g['max'].values; sib_n[g.index.values]=g['size'].values; sib_eq[g.index.values]=(g['max'].values>=0.999)
    NEW['x_sib_name_lev_max']=sib_max; NEW['x_sib_n']=sib_n; NEW['x_sib_name_eq']=sib_eq
    nmap,amap=_tm(translit_path)
    def mapped(s,mp,drop_legal):
        toks=s.split(); nn=sum(isnat(t) for t in toks); mp_=[mp.get(t,t) for t in toks]; hit=sum(1 for t in toks if isnat(t) and t in mp)
        if drop_legal: mp_=[t for t in mp_ if t not in legal] or mp_
        return ' '.join(mp_),(hit/nn if nn else 1.0)
    up=pd.unique(X.p_id.values); dn={p:mapped(Pi.at[p,'nn'],nmap,True) for p in up}; da={p:mapped(Pi.at[p,'an'],amap,False) for p in up}
    mn=[dn[x][0] for x in X.p_id.values]; qnc=Qi.nc.reindex(X.q_id).values; qan=Qi.an.reindex(X.q_id).values
    NEW['x_tr_name_cov']=np.array([dn[x][1] for x in X.p_id.values],np.float32); NEW['x_tr_addr_cov']=np.array([da[x][1] for x in X.p_id.values],np.float32)
    NEW['x_tr_n_lev_map']=process.cpdist(mn,list(qnc),scorer=Levenshtein.normalized_similarity,workers=4,dtype=np.float32)
    jac=lambda a,b:(len(a&b)/len(a|b) if (a or b) else 0.0)
    NEW['x_tr_n_jac_map']=np.array([jac(set(a.split()),set(b.split())) for a,b in zip(mn,qnc)],np.float32)
    NEW['x_tr_a_jac_map']=np.array([jac(set(da[p][0].split()),set(b.split())) for p,b in zip(X.p_id.values,qan)],np.float32)
    NEW['x_tr_native_any']=np.array([(dn[p][1]<1.0) or isnat(Pi.at[p,'nn']) for p in X.p_id.values],np.float32)
    for k,v in NEW.items(): X[k]=np.asarray(v,np.float32)
    return X
GROUPS={'competition':lambda c:c.startswith('x_name_p_') or c.startswith('x_name_q_other') or c.startswith('x_name_q_margin'),
        'frequency':lambda c:c in('x_name_q_s1_dup','x_name_q_s1_tokset_dup','x_name_p_pool_dup','x_name_dup_diff') or c.startswith('x_addr_'),
        'sibling':lambda c:c.startswith('x_sib_'),'translit':lambda c:c.startswith('x_tr_')}
