"""Exp3: learn a native-script -> Latin TOKEN dictionary from aligned TRAINING pairs only (no external transliteration data).
For every training pair (S1 owner, S2/S3 record) where the pool record has a non-ASCII (native-script) token, count co-occurrences of the native token with the tokens of the owner's
Latin name (and, separately, address). A native token t maps to the Latin token u with the highest LIFT (purity / global frequency of u) subject to minimum support and purity.
The benchmark's own entities (Exp1 region S1) are EXCLUDED so evaluation on the benchmark has no leakage. Output: pilot/exp3/translit_map.json + coverage stats."""
import sys,json,time,collections,numpy as np,pandas as pd
sys.path.insert(0,'pilot/exp3'); from norm3 import clean
D='dataset/train/'; E1='pilot/exp1/'; OUT='pilot/final_optimization/02_ablation/'; T0=time.time(); CH=500_000
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
rd=lambda f,**kw: pd.read_csv(D+f,sep='\t',dtype=str,keep_default_na=False,quoting=3,**kw)
R=set(pd.read_parquet(E1+'bench_s1.parquet',columns=['id']).id)|set(pd.read_csv('pilot/final_optimization/00_production_replay/replay_data/test_source1.tsv',sep='	',dtype=str,usecols=['entity_id']).entity_id)
s1c=pd.concat([c[c.country=='India'][['entity_id']] for c in rd('train_source1.tsv',chunksize=CH,usecols=['entity_id','country'])]); IND=set(s1c.entity_id)-R; log('India S1 (excluding benchmark region):',len(IND))
pairs=[]
for ch in rd('train_ground_truth.tsv',chunksize=CH):
    ch=ch[(ch.matched_entity_ids!='')&ch.source1_entity_id.isin(IND)]
    if len(ch): pairs.append(ch.assign(m=ch.matched_entity_ids.str.split(',')).explode('m')[['source1_entity_id','m']].rename(columns={'m':'entity_id'}))
pairs=pd.concat(pairs,ignore_index=True); log('India owned pairs',len(pairs))
nat=[]
for fn in ('train_source2.tsv','train_source3.tsv'):
    for ch in rd(fn,chunksize=CH):
        ch=ch[(ch.country=='India')&(ch.business_name.str.contains(r'[^\x00-\x7f]')|ch.business_address.str.contains(r'[^\x00-\x7f]'))]
        if len(ch): nat.append(ch.merge(pairs,on='entity_id'))
    log(fn,'native records so far',sum(len(x) for x in nat))
nat=pd.concat(nat,ignore_index=True); need=set(nat.source1_entity_id)
s1=pd.concat([c[c.entity_id.isin(need)][['entity_id','business_name','business_address']] for c in rd('train_source1.tsv',chunksize=CH)]).set_index('entity_id'); log('owners loaded',len(s1))
isnat=lambda t: any(ord(c)>127 for c in t)
cn=collections.Counter(); cu=collections.Counter(); cnu=collections.defaultdict(collections.Counter); an=collections.Counter(); au=collections.Counter(); anu=collections.defaultdict(collections.Counter); N=0; NA=0
for name,addr,o in zip(nat.business_name,nat.business_address,nat.source1_entity_id):
    ln=set(clean(s1.at[o,'business_name']).split()); la=set(clean(s1.at[o,'business_address']).split()); tn={t for t in clean(name).split() if isnat(t)}; ta={t for t in clean(addr).split() if isnat(t)}
    if tn:
        N+=1; cu.update(ln)
        for t in tn: cn[t]+=1; cnu[t].update(ln)
    if ta:
        NA+=1; au.update(la)
        for t in ta: an[t]+=1; anu[t].update(la)
def pick(c,cu_,cnu_,Ntot,minc,minp):
    m={}
    for t,n in c.items():
        if n<minc: continue
        best=max(cnu_[t].items(),key=lambda kv:(kv[1]/n)/(cu_[kv[0]]/Ntot)); u,k=best; p=k/n
        if p>=minp: m[t]=[u,round(p,3),n]
    return m
nm=pick(cn,cu,cnu,N,3,0.6); am=pick(an,au,anu,NA,5,0.6); log('name map',len(nm),'of',len(cn),'native name tokens | addr map',len(am),'of',len(an))
json.dump(dict(name=nm,addr=am,pairs_name=N,pairs_addr=NA),open(OUT+'translit_map.json','w',encoding='utf8'),ensure_ascii=False)
# ---- coverage on the benchmark's native records (held out from learning)
tot=hit=totT=hitT=0
for s in ('s2','s3'):
    p=pd.read_parquet(E1+f'bench_{s}.parquet',columns=['country','name_raw']); p=p[p.country=='India']
    for name in p.name_raw:
        for t in clean(name).split():
            if isnat(t): totT+=1; hitT+=t in nm
print('held-out coverage: native NAME tokens mapped',round(hitT/max(totT,1),3),'of',totT)
print('examples',list(nm.items())[:12]); print('top mapped by support',sorted(nm.items(),key=lambda kv:-kv[1][2])[:12])
