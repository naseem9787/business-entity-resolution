"""Replay-artifact correction: FP rate on records whose true owner IS present (another replay S1) vs ABSENT (owner outside replay).
In the real test every owner is present, so the 'owner present' rate is the production-faithful one."""
import glob,numpy as np,pandas as pd,json
R='pilot/final_optimization/00_production_replay/'; W=R+'work_prod/'
s1=pd.read_csv(R+'replay_data/test_source1.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3,usecols=['entity_id','country']); Q=set(s1.entity_id); cty=dict(zip(s1.entity_id,s1.country))
own=pd.read_parquet(R+'all_owner.parquet'); owner=dict(zip(own.p_id,own.owner)); gt=set(zip(*pd.read_parquet(R+'replay_gt.parquet')[['q_id','p_id']].values.T))
C=pd.concat([pd.read_parquet(f) for f in glob.glob(W+'scored/*.parquet')]).sort_values('s',ascending=False).drop_duplicates(['q_id','p_id']).reset_index(drop=True)
C['o']=C.p_id.map(owner).fillna(''); C['lab']=[(a,b) in gt for a,b in zip(C.q_id,C.p_id)]; C['cty']=C.q_id.map(cty)
o=np.lexsort((-C.s.values,pd.factorize(C.p_id)[0])); pg=pd.factorize(C.p_id)[0][o]; ss=C.s.values[o]; st=np.r_[True,pg[1:]!=pg[:-1]]; gs=np.flatnonzero(st); gid=np.cumsum(st)-1; gz=np.diff(np.r_[gs,len(ss)])
t2=np.where(gz>1,ss[np.minimum(gs+1,len(ss)-1)],0); pos=np.arange(len(ss))-gs[gid]; smo=np.where(pos==0,t2[gid],ss[gs][gid]); m=np.empty(len(C)); m[o]=smo; C['pred']=(C.s>=0.65)&(C.s-m>=0.05)
C['grp']=np.where(C.lab,'true',np.where(C.o=='','unowned',np.where(C.o.isin(Q),'owner_present','owner_absent')))
neg=C[~C.lab]; t=neg.groupby(['cty','grp']).pred.agg(['size','sum','mean']); print(t)
# does the TRUE owner (present) out-score the wrong claimant? (owner_present negatives)
op=C[C.grp=='owner_present'].copy(); tru=C[C.lab].set_index('p_id').s; op['owner_s']=op.p_id.map(tru)
print('\nowner_present negatives: owner has p in its candidates %.3f | owner score > claimant score %.3f'%(op.owner_s.notna().mean(),(op.owner_s>op.s).mean()))
r=t['mean'].unstack(); print('\nFP rate ratio absent/present:',(r['owner_absent']/r['owner_present']).round(1).to_dict())
json.dump({'fp_rates':{f'{a}|{b}':float(v) for (a,b),v in t['mean'].items()},'n':{f'{a}|{b}':int(v) for (a,b),v in t['size'].items()}},open(R+'owner_presence.json','w'),indent=1)
C[['q_id','p_id','s','lab','grp','cty','pred']].to_parquet(R+'replay_scored_labelled.parquet')
