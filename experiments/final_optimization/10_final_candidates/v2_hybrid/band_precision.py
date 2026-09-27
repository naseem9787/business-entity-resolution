import numpy as np,pandas as pd,glob,json
exec(open('pilot/final_optimization/10_final_candidates/v2_hybrid/additions_precision.py',encoding='utf8').read().split('R=rules(D); out={}')[0])
R=rules(D); out={}
for lo,hi,a,b in ((.7,.8,'v2_T0.7','v2_T0.8'),(.8,.9,'v2_T0.8','v2_T0.9')):
    band=(R[a]&~R[b]).values; nf=~D.foreign.values; cont=D.mp.values<.05; low=D.sp.values<.2
    r=dict(n=int(band.sum()),precision=float(D.lab.values[band].mean()),precision_owner_present=float(D.lab.values[band&nf].mean()),
           precision_contested=float(D.lab.values[band&nf&cont].mean()) if (band&nf&cont).any() else None,precision_prodlow=float(D.lab.values[band&nf&low].mean()) if (band&nf&low).any() else None,
           share_contested=float(cont[band].mean()),share_prodlow=float(low[band].mean()))
    out[f'{lo}-{hi}']=r; print('REPLAY band',lo,hi,{k:round(v,4) for k,v in r.items()})
for c in ('US','India','France'):
    T=load(glob.glob(V2+f'test_scored/{c}__*.parquet'),glob.glob(f'submission_work/scored/{c}__*.parquet')); RT=rules(T)
    for lo,hi,a,b in ((.7,.8,'v2_T0.7','v2_T0.8'),(.8,.9,'v2_T0.8','v2_T0.9')):
        band=(RT[a]&~RT[b]).values; cont=T.mp.values<.05; low=T.sp.values<.2
        out[f'TEST {c} {lo}-{hi}']=dict(n=int(band.sum()),share_contested=float(cont[band].mean()),share_prodlow=float(low[band].mean()))
        print('TEST',c,'band',lo,hi,out[f'TEST {c} {lo}-{hi}'])
    del T,RT
json.dump(out,open(V2+'band_precision.json','w'),indent=1)
