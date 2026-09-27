"""Exp3: new matcher evidence for the H@10 candidate rows (read-only on exp1/exp2; writes pilot/exp3/).  All features are label-free.
Groups (column prefix):
  x_name_*  name-ambiguity evidence (chain size among S1 / pool, p-side & q-side name competition)      -> empty-address specialist
  x_addr_*  address rarity (how many S1 entities / pool records share this exact canonical address)     -> address evidence
  x_sib_*   sibling agreement: name similarity between p and the address-CONFIRMED other candidates of q (a_cos_w>=0.8) -> empty-address & native-script
  x_tr_*    learned native->Latin dictionary features (translit_map.json, learned from training pairs only)  -> native script
Output: newfeat_H10_fix.parquet (keys + new columns) and feats_H_K10_all.parquet (Exp2 feature file + new columns, same row order)."""
import sys,json,time,numpy as np,pandas as pd
sys.path.insert(0,'pilot/exp3'); sys.path.insert(0,'pilot'); from norm3 import clean,LEGAL
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
E1='pilot/exp1/'; E2='pilot/exp2/'; OUT='pilot/exp3/'; T0=time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
F=pd.read_parquet(OUT+'feats_H_K10_fix.parquet'); n=len(F)
S1=pd.read_parquet(OUT+'bench3_s1.parquet',columns=['id','country','nc','ac','an','nn']).set_index('id')
P=pd.concat([pd.read_parquet(OUT+f'bench3_{s}.parquet',columns=['id','country','nc','ac','an','nn']).assign(src=s) for s in ('s2','s3')]).set_index('id')
q=S1.loc[F.q_id.values]; p=P.loc[F.p_id.values]; NEW={}
# ---------------- x_name / x_addr: frequency evidence
S1['nk']=S1.country+'|'+S1.nc; S1['tk']=S1.country+'|'+S1.nc.map(lambda s:' '.join(sorted(set(s.split())))); S1['ak']=S1.country+'|'+S1.ac
P['nk']=P.src+'|'+P.country+'|'+P.nc; P['ak']=P.src+'|'+P.country+'|'+P.ac
cs1n=S1.nk.map(S1.nk.value_counts()); cs1t=S1.tk.map(S1.tk.value_counts()); cs1a=S1.ak.map(S1.ak.value_counts()); cpn=P.nk.map(P.nk.value_counts()); cpa=P.ak.map(P.ak.value_counts()).where(P.ac!='',0)
qs1n=cs1n.loc[F.q_id.values].values; qs1t=cs1t.loc[F.q_id.values].values; qs1a=cs1a.loc[F.q_id.values].values; ppn=cpn.loc[F.p_id.values].values; ppa=cpa.loc[F.p_id.values].values
NEW['x_name_q_s1_dup']=np.log1p(qs1n); NEW['x_name_q_s1_tokset_dup']=np.log1p(qs1t); NEW['x_name_p_pool_dup']=np.log1p(ppn); NEW['x_name_dup_diff']=np.log1p(ppn)-np.log1p(qs1n)
NEW['x_addr_q_s1_freq']=np.log1p(qs1a); NEW['x_addr_p_pool_freq']=np.log1p(ppa); NEW['x_addr_freq_diff']=np.log1p(ppa)-np.log1p(qs1a)
NEW['x_addr_rare_unique']=((qs1a<=1)&(ppa<=1)&(F.a_miss_p.values==0)).astype(np.float32)
log('frequency features')
# ---------------- x_name: candidate-level name competition (other S1 entities competing for the same p, other pool records competing for the same q)
def other_best(g,v):
    o=np.lexsort((-v,g)); gs=g[o]; vs=v[o]; st=np.r_[True,gs[1:]!=gs[:-1]]; idx=np.flatnonzero(st); gid=np.cumsum(st)-1; sz=np.diff(np.r_[idx,len(vs)])
    t1=vs[idx]; t2=np.where(sz>1,vs[np.minimum(idx+1,len(vs)-1)],0.0); pos=np.arange(len(vs))-idx[gid]; res=np.where(pos==0,t2[gid],t1[gid]); out=np.empty(len(v)); out[o]=res; return out
gp=pd.factorize(F.p_id)[0]; gq=pd.factorize(F.q_id)[0]
for m,vals in (('name',F.n_lev_nc.values.astype(float)),('addr',F.a_cos_w.values.astype(float)),('fused',F.fused.values.astype(float))):
    ob=other_best(gp,vals); NEW[f'x_name_p_other_best_{m}']=ob; NEW[f'x_name_p_margin_{m}']=vals-ob
NEW['x_name_p_rows']=np.bincount(gp)[gp].astype(np.float32); ob=other_best(gq,F.n_lev_nc.values.astype(float)); NEW['x_name_q_other_best_name']=ob; NEW['x_name_q_margin_name']=F.n_lev_nc.values-ob
log('competition features')
# ---------------- x_sib: address-confirmed sibling agreement (chunked by query)
conf=((F.a_cos_w.values>=0.8)&(F.a_miss_p.values==0)); pn=P.nn.loc[F.p_id.values].values
sib_max=np.full(n,np.nan,np.float32); sib_n=np.zeros(n,np.float32); sib_eq=np.zeros(n,np.float32); ch=gq%8
for c in range(8):
    tm=ch==c; sm=tm&conf
    T=pd.DataFrame({'q_id':F.q_id.values[tm],'p_id':F.p_id.values[tm],'pn':pn[tm],'row':np.flatnonzero(tm)}); Sb=pd.DataFrame({'q_id':F.q_id.values[sm],'sp':F.p_id.values[sm],'sn':pn[sm]})
    M=T.merge(Sb,on='q_id'); M=M[M.p_id!=M.sp]
    if len(M):
        sim=process.cpdist(M.pn.tolist(),M.sn.tolist(),scorer=Levenshtein.normalized_similarity,workers=4,dtype=np.float32); g=pd.DataFrame({'row':M.row.values,'sim':sim}).groupby('row').sim.agg(['max','size']); sib_max[g.index.values]=g['max'].values; sib_n[g.index.values]=g['size'].values
        sib_eq[g.index.values]=(g['max'].values>=0.999)
    log('siblings chunk',c,len(M))
NEW['x_sib_name_lev_max']=sib_max; NEW['x_sib_n']=sib_n; NEW['x_sib_name_eq']=sib_eq
# ---------------- x_tr: learned native->Latin dictionary
TM=json.load(open(OUT+'translit_map.json',encoding='utf8')); nmap={t:v[0] for t,v in TM['name'].items()}; amap={t:v[0] for t,v in TM['addr'].items()}; isnat=lambda t:any(ord(c)>127 for c in t)
def mapped(s,mp,drop_legal):
    toks=s.split(); nn=sum(isnat(t) for t in toks); mp_=[mp.get(t,t) for t in toks]; hit=sum(1 for t in toks if isnat(t) and t in mp)
    if drop_legal: mp_=[t for t in mp_ if t not in LEGAL] or mp_
    return ' '.join(mp_),(hit/nn if nn else 1.0)
uP=P.loc[pd.unique(F.p_id.values)]; dn={};da={}
for pid,nn_,an_ in zip(uP.index,uP.nn,uP.an): dn[pid]=mapped(nn_,nmap,True); da[pid]=mapped(an_,amap,False)
mn=np.array([dn[x][0] for x in F.p_id.values],dtype=object); NEW['x_tr_name_cov']=np.array([dn[x][1] for x in F.p_id.values],np.float32); NEW['x_tr_addr_cov']=np.array([da[x][1] for x in F.p_id.values],np.float32)
NEW['x_tr_n_lev_map']=process.cpdist(mn.tolist(),q.nc.tolist(),scorer=Levenshtein.normalized_similarity,workers=4,dtype=np.float32)
jac=lambda a,b:(len(a&b)/len(a|b) if (a or b) else 0.0)
NEW['x_tr_n_jac_map']=np.array([jac(set(a.split()),set(b.split())) for a,b in zip(mn,q.nc.values)],np.float32)
ma=[da[x][0] for x in F.p_id.values]; NEW['x_tr_a_jac_map']=np.array([jac(set(a.split()),set(b.split())) for a,b in zip(ma,q.an.values)],np.float32)
NEW['x_tr_native_any']=(np.array([dn[x][1]<1.0 or isnat(P.at[x,'nn']) for x in F.p_id.values])).astype(np.float32)
log('translit features; name coverage on candidate rows (native rows only):',float(np.mean([dn[x][1] for x in F.p_id.values if dn[x][1]<1.0]) if any(dn[x][1]<1.0 for x in F.p_id.values[:200000]) else 1.0))
N=pd.DataFrame({k:np.asarray(v,np.float32) for k,v in NEW.items()}); N.insert(0,'p_id',F.p_id.values); N.insert(0,'q_id',F.q_id.values); N.to_parquet(OUT+'newfeat_H10_fix.parquet')
for c in N.columns[2:]: F[c]=N[c].values
F.to_parquet(OUT+'feats_H_K10_fixall.parquet'); log('saved; new columns',len(N.columns)-2,list(N.columns[2:]))
