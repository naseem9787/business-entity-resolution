import numpy as np,pandas as pd,glob,json
exec(open('pilot/final_optimization/10_final_candidates/v2_hybrid/shift_analysis.py',encoding='utf8').read().split('RES={}')[0])
s1r=pd.read_csv(RP+'replay_data/test_source1.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3,usecols=['entity_id','country']); cr=dict(zip(code_q(s1r.entity_id),s1r.country))
gt=pd.read_parquet(RP+'replay_gt.parquet'); gtk=set(zip(code_q(gt.q_id),code_p(gt.p_id))); own=pd.read_parquet(RP+'all_owner.parquet'); Qs=set(code_q(s1r.entity_id))
D=load(None,glob.glob(RP+'work_prod/scored/*.parquet'),pd.read_parquet('pilot/final_optimization/02_ablation/hybrid_H5_oof.parquet')); D['lab']=[(a,b) in gtk for a,b in zip(D.q.values,D.p.values)]
om=dict(zip(code_p(own.p_id),code_q(own.owner))); o=D.p.map(om); D['foreign']=o.notna()&~o.isin(Qs)&~D.lab
R=rules(D); out={}
for T in ('v2_T0.7','v2_T0.8','v2_T0.9'):
    add=(R[T]&~R['subA']).values; r={}
    for nm,m in {'prod<.2':D.sp.values<.2,'prod .2-.65':(D.sp.values>=.2)&(D.sp.values<.65),'prod .65-.85':(D.sp.values>=.65)&(D.sp.values<.85),'prod>=.85 (margin-rejected)':D.sp.values>=.85,'prod margin<.05':D.mp.values<.05}.items():
        k=add&m; r[nm]=dict(n=int(k.sum()),precision=round(float(D.lab.values[k].mean()),4) if k.any() else None,precision_excl_foreign=round(float(D.lab.values[k&~D.foreign.values].mean()),4) if (k&~D.foreign.values).any() else None)
    out[T]=r; print(T,r)
json.dump(out,open(V2+'additions_precision.json','w'),indent=1)
