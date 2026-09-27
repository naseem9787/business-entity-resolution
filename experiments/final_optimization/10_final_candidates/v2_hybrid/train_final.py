"""Final v2 hybrid stage-1: trained on benchmark v2 features + ALL production-replay hybrid features (production density). Rule selected on replay LOSO OOF (half A)."""
import glob,json,numpy as np,pandas as pd,lightgbm as lgb
RP='pilot/final_optimization/00_production_replay/'; A='pilot/final_optimization/02_ablation/'; D='pilot/final_optimization/10_final_candidates/v2_hybrid/'
LGB=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=8)
B=pd.read_parquet(A+'bench_v2_feats.parquet'); META=['q_id','p_id','label','country','src','fold','p_empty','p_native','shard','state','s']
fs=[c for c in B.columns if c not in META and not c.startswith('x_')]+[c for c in B.columns if c.startswith('x_')]
gt=pd.read_parquet(RP+'replay_gt.parquet'); gts=set(zip(gt.q_id,gt.p_id))
H=pd.concat([pd.read_parquet(f) for f in glob.glob(RP+'hybrid_feats/*.parquet')],ignore_index=True); H['label']=np.array([(a,b) in gts for a,b in zip(H.q_id,H.p_id)]).astype(np.int8)
m=lgb.train(LGB,lgb.Dataset(np.vstack([H[fs].to_numpy(np.float32),B[fs].to_numpy(np.float32)]),np.concatenate([H.label.values,B.label.values])),400); m.save_model(D+'stage1.txt')
rule=[r for r in json.load(open(A+'hybrid_results.json')) if r['model'].startswith('H5')][0]['rule']
json.dump(dict(features=fs,rule=rule,train_rows=int(len(H)+len(B)),note='hybrid: production candidates re-scored with norm3 + v2 features'),open(D+'meta.json','w'),indent=1); print('saved',len(fs),'features, rule',rule)
