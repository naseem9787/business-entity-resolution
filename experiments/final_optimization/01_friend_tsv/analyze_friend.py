"""Final-day task 1: audit friend's matching_results.tsv against OUR submitted output (read-only). No test labels exist -> every 'quality' statement here is a
label-free proxy (ownership violations are CERTAIN errors; model scores are only proxies). Outputs friend_audit.json + pairs_union.parquet in this folder."""
import os,sys,json,re,time,glob,unicodedata,numpy as np,pandas as pd
T0=time.time(); OUT=os.path.dirname(os.path.abspath(__file__))+'/'
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
FRIEND=sys.argv[1] if len(sys.argv)>1 else 'C:/Users/nasee/Downloads/matching_results.tsv'; OURS='output/matching_results.tsv'; TD='dataset/test/'
def code(ids):
    s=pd.Series(ids,dtype=str); src=s.str[1].astype(np.int64); return s.str[3:].astype(np.int64)*4+src
def read_sub(fn):
    d=pd.read_csv(fn,sep='\t',dtype=str,keep_default_na=False,quoting=3); q=code(d.source1_entity_id).values
    e=d.assign(m=d.iloc[:,1].str.split(',')).explode('m'); e=e[e.m!='']; return q,pd.DataFrame({'q':code(e.source1_entity_id).values,'p':code(e.m).values})
qF,F=read_sub(FRIEND); qO,O=read_sub(OURS); log('friend pairs',len(F),'ours pairs',len(O))
R={}
# ---------- test metadata
s1=pd.read_csv(TD+'test_source1.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3); s1['q']=code(s1.entity_id).values
ALLQ=np.sort(s1.q.values); R['test_S1']=len(s1); R['friend_rows']=len(qF); R['friend_missing_S1']=int(len(np.setdiff1d(ALLQ,qF))); R['friend_extra_S1']=int(len(np.setdiff1d(qF,ALLQ))); R['friend_dup_S1_rows']=int(len(qF)-len(np.unique(qF)))
qc=dict(zip(s1.q,s1.country))
def squash(x):
    x=unicodedata.normalize('NFKD',x.lower()); x=''.join(c for c in x if not unicodedata.combining(c)); return re.sub(r'[\W_]+','',x)
LEG=re.compile(r'\b(inc|incorporated|llc|ltd|limited|private|pvt|corp|corporation|co|company|llp|lp|plc|sarl|sas|sa|sci|eurl|sasu|the|and|et|of|services|center|centre|group|groupe)\b')
def core(x):
    x=unicodedata.normalize('NFKD',x.lower()); x=''.join(c for c in x if not unicodedata.combining(c)); x=re.sub(r'[^\w\s]',' ',x); return re.sub(r'\s+','',LEG.sub(' ',x))
NUM=re.compile(r'(?<![\w-])(\d{1,6})(?![\w-])')
def hnum(a):
    m=NUM.findall(a); return m[0] if m else ''
s1i=pd.DataFrame({'q':s1.q.values,'country':s1.country.values,'q_core':[core(x) for x in s1.business_name],'q_hn':[hnum(x) for x in s1.business_address]}).set_index('q'); log('S1 meta')
P=[]
for i,fn in ((2,'test_source2.tsv'),(3,'test_source3.tsv')):
    d=pd.read_csv(TD+fn,sep='\t',dtype=str,keep_default_na=False,quoting=3)
    P.append(pd.DataFrame({'p':code(d.entity_id).values,'p_country':d.country.values,'p_empty':(d.business_address=='').values,'p_native':d.business_name.str.contains('[^\x00-\u024f]',regex=True).values,
                           'p_core':[core(x) for x in d.business_name],'p_hn':[hnum(x) for x in d.business_address]})); del d; log('pool meta',fn)
P=pd.concat(P,ignore_index=True).set_index('p')
# ---------- our candidates + scores (dedupe exactly as production) + margin
C=[]
for f in sorted(glob.glob('submission_work/scored/*.parquet')):
    d=pd.read_parquet(f); C.append(pd.DataFrame({'q':code(d.q_id).values,'p':code(d.p_id).values,'s':d.s.values.astype(np.float32)}))
C=pd.concat(C,ignore_index=True).sort_values('s',ascending=False,kind='stable').drop_duplicates(['q','p']).reset_index(drop=True); log('our candidates',len(C))
C['top_other']=C.groupby('p').s.transform(lambda x: x.nlargest(2).iloc[-1] if len(x)>1 else 0.0) if False else 0.0
o=np.lexsort((-C.s.values,C.p.values)); pv=C.p.values[o]; sv=C.s.values[o]; st=np.r_[True,pv[1:]!=pv[:-1]]; gs=np.flatnonzero(st); gid=np.cumsum(st)-1; gz=np.diff(np.r_[gs,len(sv)])
t1=sv[gs]; t2=np.where(gz>1,sv[np.minimum(gs+1,len(sv)-1)],0); pos=np.arange(len(sv))-gs[gid]; smo=np.where(pos==0,t2[gid],t1[gid]); m=np.empty(len(C),np.float32); m[o]=smo; C['smo']=m; C['margin']=C.s-C.smo
# ---------- A. basic stats
def basic(D,qarr,name):
    n=np.bincount(np.searchsorted(ALLQ,D.q.values),minlength=len(ALLQ)); cty=pd.Series([qc[x] for x in D.q.values])
    return dict(pairs=len(D),linked_S1=int((n>0).sum()),empty_S1=int((n==0).sum()),empty_rate=float((n==0).mean()),mean_matches=float(n.mean()),mean_matches_linked=float(n[n>0].mean()),
                by_country=cty.value_counts().to_dict(),by_source={'S2':int((D.p.values%4==2).sum()),'S3':int((D.p.values%4==3).sum())},
                p_assigned_to_multiple_S1=int(D.p.duplicated(keep=False).sum()),distinct_p_multi=int(D.p[D.p.duplicated()].nunique()),hist={int(k):int(v) for k,v in zip(*np.unique(np.minimum(n,12),return_counts=True))})
R['A_friend']=basic(F,qF,'friend'); R['A_ours']=basic(O,qO,'ours'); log('A',R['A_friend']['pairs'],R['A_ours']['pairs'])
for c in ('US','India','France'):
    nq=int((s1.country==c).sum()); R.setdefault('A_country',{})[c]={'S1':nq,'friend_matches':R['A_friend']['by_country'].get(c,0),'ours_matches':R['A_ours']['by_country'].get(c,0)}
# ---------- B. agreement
kF=F.q.values*(1<<34)+F.p.values; kO=O.q.values*(1<<34)+O.p.values; both=np.intersect1d(kF,kO); R['B']=dict(agree_pairs=int(len(both)),friend_only_pairs=int(len(kF)-len(both)),ours_only_pairs=int(len(kO)-len(both)),jaccard=float(len(both)/len(np.union1d(kF,kO))))
gF=F.groupby('q').p.apply(frozenset); gO=O.groupby('q').p.apply(frozenset); cat={}
for q in ALLQ:
    a=gF.get(q,frozenset()); b=gO.get(q,frozenset())
    k='both_empty' if not a and not b else 'same_set' if a==b else 'friend_empty' if not a else 'ours_empty' if not b else 'friend_subset_of_ours' if a<b else 'ours_subset_of_friend' if b<a else 'partial_overlap' if a&b else 'disjoint'
    cat[k]=cat.get(k,0)+1
R['B']['entity_categories']=cat; log('B',cat)
# ---------- C/D. friend-only and ours-only pairs vs our candidate set / scores / the other side's owner
Ck=C.q.values*(1<<34)+C.p.values; ordk=np.argsort(Ck); Cks=Ck[ordk]
def lookup(k):
    i=np.searchsorted(Cks,k); i=np.minimum(i,len(Cks)-1); hit=Cks[i]==k; return hit,np.where(hit,ordk[i],-1)
fo=F[~np.isin(kF,both)].copy(); hit,idx=lookup(fo.q.values*(1<<34)+fo.p.values); fo['in_our_cands']=hit; fo['our_s']=np.where(hit,C.s.values[np.maximum(idx,0)],np.nan); fo['our_margin']=np.where(hit,C.margin.values[np.maximum(idx,0)],np.nan)
ourown=pd.Series(O.q.values,index=O.p.values); ourown=ourown[~ourown.index.duplicated()]; fo['ours_assign_p_to']=ourown.reindex(fo.p.values).values
fo['ours_q_empty']=~pd.Series(fo.q.values).isin(set(O.q.values)).values
frown=pd.Series(F.q.values,index=F.p.values); frown=frown[~frown.index.duplicated()]
oo=O[~np.isin(kO,both)].copy(); hit2,idx2=lookup(oo.q.values*(1<<34)+oo.p.values); oo['our_s']=C.s.values[np.maximum(idx2,0)]; oo['our_margin']=C.margin.values[np.maximum(idx2,0)]
oo['friend_assign_p_to']=frown.reindex(oo.p.values).values; oo['friend_q_empty']=~pd.Series(oo.q.values).isin(set(F.q.values)).values
def desc(D,prefix):
    r={'n':len(D)}
    if 'in_our_cands' in D: r['in_our_final_candidates']=float(D.in_our_cands.mean()); r['our_score_quantiles_when_candidate']=D.our_s.quantile([.1,.25,.5,.75,.9]).round(3).to_dict()
    else: r['our_score_quantiles']=D.our_s.quantile([.1,.25,.5,.75,.9]).round(3).to_dict()
    oth='ours_assign_p_to' if 'ours_assign_p_to' in D else 'friend_assign_p_to'; own=D[oth]
    r['p_assigned_by_other_model_to_different_S1']=float((own.notna()&(own!=D.q)).mean()); r['p_unassigned_by_other_model']=float(own.isna().mean())
    r['q_empty_in_other_model']=float(D[[c for c in D.columns if c.endswith('_q_empty')][0]].mean()); return r
R['C_friend_only']=desc(fo,'fo'); R['D_ours_only']=desc(oo,'oo'); log('C',R['C_friend_only']); log('D',R['D_ours_only'])
# ---------- E. slice the three pair groups by evidence
ag=O[np.isin(kO,both)].copy(); hit3,idx3=lookup(ag.q.values*(1<<34)+ag.p.values); ag['our_s']=C.s.values[np.maximum(idx3,0)]
U=pd.concat([ag.assign(grp='agree'),fo.assign(grp='friend_only'),oo.assign(grp='ours_only')],ignore_index=True)[['q','p','grp','our_s']]
U=U.join(s1i,on='q').join(P,on='p'); U['src']=np.where(U.p%4==2,'S2','S3')
U['name_exact_core']=(U.q_core==U.p_core)&(U.q_core!=''); U['hn_state']=np.select([(U.q_hn=='')|(U.p_hn==''),U.q_hn==U.p_hn],['missing','equal'],'conflict')
U.drop(columns=['q_core','p_core','q_hn','p_hn']).to_parquet(OUT+'pairs_union.parquet'); R['E']={}
for dim in ('country','src','p_empty','p_native','name_exact_core','hn_state'):
    R['E'][dim]=pd.crosstab(U[dim],U.grp).to_dict();
print(pd.concat({dim:pd.crosstab(U[dim],U.grp,normalize='columns').round(3) for dim in ('country','src','p_empty','p_native','name_exact_core','hn_state')}).to_string())
# ---------- certain-error counts: records given to >1 S1 (at most one owner exists)
cross=pd.DataFrame({'p':fo.p.values,'fq':fo.q.values,'oq':fo.ours_assign_p_to.values}); cross=cross[cross.oq.notna()&(cross.oq!=cross.fq)]
R['conflict_same_record_different_S1']=dict(n=len(cross),note='friend and ours give the same S2/S3 record to DIFFERENT S1 -> at least one of the two is a false merge')
json.dump(R,open(OUT+'friend_audit.json','w'),indent=1,default=lambda x:float(x) if isinstance(x,(np.floating,)) else int(x) if isinstance(x,(np.integer,)) else str(x))
fo.to_parquet(OUT+'friend_only.parquet'); oo.to_parquet(OUT+'ours_only.parquet'); log('saved')
print(json.dumps({k:R[k] for k in ('A_friend','A_ours','A_country','B','C_friend_only','D_ours_only','conflict_same_record_different_S1','friend_missing_S1','friend_dup_S1_rows')},indent=1,default=str))
