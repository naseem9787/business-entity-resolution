"""(1) extend margin grid beyond 0.3, (2) score the final rule on the held-out half ONLY, (3) India routing A/B (broadcast vs none) on India replay S1."""
import sys,json,numpy as np,pandas as pd,glob
exec(open('pilot/final_optimization/00_production_replay/eval_corrected.py',encoding='utf8').read().split('OUT={}')[0])
keepq,D,ntrue=setup(0.2,True); s=D.s.values; mg=s-D.smo.values; res={}
for t in (.8,.85,.9): 
    for m in (.2,.3,.4,.5,.6):
        f,_,_=fq(D,(s>=t)&(mg>=m),ntrue); res[f'{t}|{m}']=(round(f[keepq&half].mean(),4),round(f[keepq&~half].mean(),4))
print('extended grid (A,B):',res)
def rule(D,t0,m0,e=.1,n=-.15,h=.08,tc=(.02,.05)):
    s=D.s.values; mg=s-D.smo.values; th=np.full(len(D),t0); ma=np.full(len(D),m0)
    th=np.where(D.p_empty.values,np.maximum(th,t0+e),th); th=np.where(D.ncomp.values>=2,np.maximum(th,t0+h),th); th=np.where(D.p_native.values&~D.p_empty.values,np.minimum(th,t0+n),th)
    tcm=D.s2nd_q.values>=0.9; th=np.where(tcm,np.maximum(th,t0+tc[0]),th); ma=np.where(tcm,ma+tc[1],ma); return (s>=th)&(mg>=ma)
for name,p in (('production',(s>=.65)&(mg>=.05)),('global .85/.3',(s>=.85)&(mg>=.3)),('final rule',rule(D,.85,.3)),('final rule, no native relax',rule(D,.85,.3,n=0))):
    f,tp,npd=fq(D,p,ntrue); print(f'{name:28s} held-out B F {f[keepq&~half].mean():.4f} | US {f[keepq&~half&(qc=="US")].mean():.4f} India {f[keepq&~half&(qc=="India")].mean():.4f} | singleton {(npd[keepq&(ntrue==0)]==0).mean():.3f}')
# India routing A/B (pessimistic, no owner correction; both runs identical except the India broadcast pool)
for name,W in (('broadcast (production)','work_prod'),('no broadcast','work_india_nobroadcast')):
    X=pd.concat([pd.read_parquet(f) for f in glob.glob(R+W+'/scored/India__*.parquet')]).sort_values('s',ascending=False).drop_duplicates(['q_id','p_id']).reset_index(drop=True)
    X['qi']=X.q_id.map(qmap).values; gts=set(zip(gt.q_id,gt.p_id)); X['lab']=[(a,b) in gts for a,b in zip(X.q_id,X.p_id)]; X['smo']=smo_of(X); sel=qc=='India'
    for t,m in ((.65,.05),(.85,.3)):
        pred=(X.s.values>=t)&(X.s.values-X.smo.values>=m); f,tp,npd=fq(X,pred,ntrue_all)
        print(f'India {name:24s} rule {t}/{m}: F {f[sel].mean():.4f} P {tp[sel].sum()/npd[sel].sum():.4f} R {tp[sel].sum()/ntrue_all[sel].sum():.4f} cand-recall {X.lab.sum()/ntrue_all[sel].sum():.4f} FM {int(npd[sel].sum()-tp[sel].sum())} top1-mean {X.groupby("q_id").s.max().mean():.3f}')
