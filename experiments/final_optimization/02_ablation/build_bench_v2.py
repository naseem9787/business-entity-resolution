"""Benchmark training data for the v2 stage-1: Exp3 norm3 features on the Exp2 H@10 candidates (feats_H_K10_fix.parquet) + v2feats computed per
production-style shard (country x state of the S1; pool = records of that state + records without a state, i.e. the broadcast analogue)."""
import sys,time,numpy as np,pandas as pd
sys.path.insert(0,'pilot/exp3'); sys.path.insert(0,'pilot'); import norm3,v2feats
T0=time.time(); OUT='pilot/final_optimization/02_ablation/'
X=pd.read_parquet('pilot/exp3/feats_H_K10_fix.parquet')
S1=pd.read_parquet('pilot/exp3/bench3_s1.parquet',columns=['id','country','nc','ac','an','nn','state'])
P=pd.concat([pd.read_parquet(f'pilot/exp3/bench3_{s}.parquet',columns=['id','country','nc','ac','an','nn','owner','state']).assign(src=s) for s in ('s2','s3')],ignore_index=True)
ost=S1.set_index('id').state; P['st']=np.where(P.owner!='',ost.reindex(P.owner).values,P.state.values); P['st']=P.st.fillna('')
X['q_state']=S1.set_index('id').state.reindex(X.q_id).values; parts=[]
for (c,st),Xs in X.groupby(['country','q_state']):
    Q=S1[(S1.country==c)&(S1.state==st)]; Ps=P[(P.country==c)&((P.st==st)|(P.st==''))]; Ps=pd.concat([Ps,P[P.id.isin(set(Xs.p_id))&~P.id.isin(set(Ps.id))]])
    parts.append(v2feats.add(Xs.reset_index(drop=True).copy(),Q,Ps,'pilot/final_optimization/02_ablation/translit_map.json',norm3.LEGAL)); print(f'[{time.time()-T0:5.0f}s]',c,st,len(Xs),flush=True)
X=pd.concat(parts,ignore_index=True).drop(columns=['q_state']); X.to_parquet(OUT+'bench_v2_feats.parquet'); print('saved',X.shape)
