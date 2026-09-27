"""Exp2 step 5: assemble a per-config candidate/feature file in EXACTLY the Exp1 schema (feats_top15.parquet analogue) from the pair-feature store.
usage: python pilot/24_assemble.py <A|G|H> <K>   -> pilot/exp2/feats_{cfg}_K{K}{SUF}.parquet  (feed to 12_ownership.py via EXP_OUT/EXP_FEATS)"""
import sys,gc,numpy as np,pandas as pd
import os
cfg,K=sys.argv[1],int(sys.argv[2]); OUT='pilot/exp2/'; parts=[]; KEEP=[c for c in os.environ.get('KEEP','').split(',') if c]; SUF='_'+'_'.join(KEEP) if KEEP else ''
for c in ('India','US'):
    for s in ('s2','s3'):
        X=pd.read_parquet(OUT+f'pairfeat_{c}_{s}.parquet'); X=X[X[f'rank_{cfg}']<=K].copy()
        # context features, computed exactly as in Exp1 (per query x source list, on the fused cosine)
        f=X.fused.values; X=X.sort_values(['q_id','fused'],ascending=[True,False],kind='stable').reset_index(drop=True); f=X.fused.values; g=pd.Series(f).groupby(X.q_id.values)
        X['fused_rank']=g.rank(ascending=False,method='first').values; X['fused_gap_top']=f-g.transform('max').values; X['n_cands']=g.transform('size').values
        parts.append(X.drop(columns=[c for c in ['qi','score_H','rank_A','rank_G','rank_H'] if c not in KEEP])); del X; gc.collect()
X=pd.concat(parts,ignore_index=True); del parts
g=X.groupby('q_id').fused; X['fused_rank_all']=g.rank(ascending=False,method='first'); X['fused_gap_top_all']=X.fused-g.transform('max')
X.to_parquet(OUT+f'feats_{cfg}_K{K}{SUF}.parquet'); print(cfg,K,X.shape,'label sum',int(X.label.sum()),'per entity',round(len(X)/124965,2))
