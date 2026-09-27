"""Calibration table for the cached production TEST scores (unlabelled), per country, streamed per shard (each S1 lives in exactly one shard)."""
import glob,os,json,numpy as np,pandas as pd
acc={}
for f in glob.glob('submission_work/scored/*.parquet'):
    c=os.path.basename(f).split('__')[0]; d=pd.read_parquet(f,columns=['q_id','s']); d=d.sort_values(['q_id','s'],ascending=[True,False]); g=d.groupby('q_id').s
    t1=g.nth(0).values; t2=g.nth(1).reindex(range(0)) if False else d.groupby('q_id').s.apply(lambda x:x.iloc[1] if len(x)>1 else 0.0).values
    a=acc.setdefault(c,{'top1':[],'top2':[],'n':0,'amb':0,**{t:[] for t in (.5,.6,.65,.7,.75,.8,.9)}})
    a['top1'].append(t1); a['top2'].append(t2); a['n']+=len(d); a['amb']+=int(d.s.between(.2,.97).sum())
    for t in (.5,.6,.65,.7,.75,.8,.9): a[t].append((d.s>=t).groupby(d.q_id).sum().values)
out={}
for c,a in acc.items():
    t1=np.concatenate(a['top1']); t2=np.concatenate(a['top2'])
    out[c]={'top1_mean':float(t1.mean()),'top1_q':np.quantile(t1,[.1,.25,.5,.75,.9]).round(3).tolist(),'frac_pairs_in_.2-.97':a['amb']/a['n'],'top2_mean':float(t2.mean()),'top1-top2_median':float(np.median(t1-t2)),**{f'per_S1_>={t}':float(np.concatenate(a[t]).mean()) for t in (.5,.6,.65,.7,.75,.8,.9)}}
json.dump(out,open('pilot/final_optimization/00_production_replay/test_calibration.json','w'),indent=1); print(json.dumps(out,indent=1))
