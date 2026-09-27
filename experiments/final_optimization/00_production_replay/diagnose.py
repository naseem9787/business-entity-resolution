"""Why does production replay lose 5.7% of true pairs at retrieval and produce so many foreign-owned false merges? Routing/broadcast diagnosis."""
import os,sys,glob,json,numpy as np,pandas as pd
R='pilot/final_optimization/00_production_replay/'; W=R+'work_prod/'
key={}
for s in ('s1','s2','s3'):
    for f in glob.glob(W+f'parts/{s}/*.parquet'):
        c,k,_=os.path.basename(f)[:-8].split('__'); ids=pd.read_parquet(f,columns=['id']).id.values; key.update(dict.fromkeys(ids,f'{c}/{k}'))
print('routed records',len(key))
gt=pd.read_parquet(R+'replay_gt.parquet').reset_index(drop=True); C=pd.concat([pd.read_parquet(f,columns=['q_id','p_id','s']) for f in glob.glob(W+'scored/*.parquet')]).drop_duplicates(['q_id','p_id'])
ck=set(zip(C.q_id,C.p_id)); gt['in_cands']=[(a,b) in ck for a,b in zip(gt.q_id,gt.p_id)]; gt['qk']=gt.q_id.map(key); gt['pk']=gt.p_id.map(key)
gt['p_route']=np.where(gt.pk.str.endswith('_NOSTATE'),'broadcast(NOSTATE)',np.where(gt.pk==gt.qk,'same_shard','OTHER_shard(unreachable)'))
print('\nTRUE pairs: routing x in_candidates'); print(pd.crosstab(gt.p_route,gt.in_cands,margins=True))
print('\nby country:'); gt['cty']=gt.qk.str.split('/').str[0]; print(pd.crosstab([gt.cty,gt.p_route],gt.in_cands,normalize='index').round(3))
# pool composition per shard
comp={}
for s in ('s2','s3'):
    for f in glob.glob(W+f'parts/{s}/*.parquet'):
        c,k,_=os.path.basename(f)[:-8].split('__'); comp[(s,c,k)]=comp.get((s,c,k),0)+len(pd.read_parquet(f,columns=['id']))
for c in ('US','India'):
    b=comp.get(('s2',c,'_NOSTATE'),0); st=sum(v for (s,cc,k),v in comp.items() if s=='s2' and cc==c and not k.startswith('_')); print(f'{c} S2: routed-to-states {st}, broadcast NOSTATE {b} ({b/(b+st):.1%}) -> every {c} shard receives all {b} broadcast records')
# broadcast records: how many empty-address?
