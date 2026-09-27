"""Hybrid = production candidates re-scored by v2 matcher. Same protocol as train_eval_v2.py (header reused)."""
import numpy as np,pandas as pd,lightgbm as lgb,glob,json,os
exec(open('pilot/final_optimization/02_ablation/train_eval_v2.py',encoding='utf8').read().split('RES=[]')[0])
H=pd.concat([pd.read_parquet(f) for f in glob.glob(RP+'hybrid_feats/*.parquet')],ignore_index=True); H=H[H.q_id.isin(qmap)].reset_index(drop=True)
H['label']=np.array([(a,b) in gts for a,b in zip(H.q_id,H.p_id)]).astype(np.int8); H['state']=qstate[H.q_id.map(qmap).values]
B=pd.read_parquet(OUT+'bench_v2_feats.parquet'); META=['q_id','p_id','label','country','src','fold','p_empty','p_native','shard','state','s']
base=[c for c in B.columns if c not in META and not c.startswith('x_')]; xall=[c for c in B.columns if c.startswith('x_')]; RES=[]
for name,fs in {'H0 hybrid tokenizer-only (base feats)':base,'H1 hybrid +translit':base+[c for c in xall if v2feats.GROUPS['translit'](c)],'H3 hybrid full v2':base+xall}.items():
    m=lgb.train(LGB,lgb.Dataset(B[fs].to_numpy(np.float32),B.label.values),400); D=H[['q_id','p_id','p_empty','p_native']].copy(); D['s']=m.predict(H[fs].to_numpy(np.float32)); RES.append(evaluate(D,name+' [bench-trained]'))
s=np.zeros(len(H)); XB=B[base+xall].to_numpy(np.float32)
for st in np.unique(H.state):
    tr=H.state.values!=st; m=lgb.train(LGB,lgb.Dataset(np.vstack([H.loc[tr,base+xall].to_numpy(np.float32),XB]),np.concatenate([H.label.values[tr],B.label.values])),400); s[~tr]=m.predict(H.loc[~tr,base+xall].to_numpy(np.float32)); log('H5 fold',st)
D=H[['q_id','p_id','p_empty','p_native']].copy(); D['s']=s; RES.append(evaluate(D,'H5 hybrid full v2, bench+replay LOSO')); D.to_parquet(OUT+'hybrid_H5_oof.parquet')
json.dump(RES,open(OUT+'hybrid_results.json','w'),indent=1,default=float); log('saved')
