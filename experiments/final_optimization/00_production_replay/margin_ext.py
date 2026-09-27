import numpy as np
exec(open('pilot/final_optimization/00_production_replay/eval_corrected.py',encoding='utf8').read().split('OUT={}')[0])
exec(open('pilot/final_optimization/00_production_replay/final_check.py',encoding='utf8').read().split('def rule')[1].join(['def rule','']) if False else '')
def rule(D,t0,m0,e=.1,n=-.15,h=.08,tc=(.02,.05)):
    s=D.s.values; mg=s-D.smo.values; th=np.full(len(D),t0); ma=np.full(len(D),m0)
    th=np.where(D.p_empty.values,np.maximum(th,t0+e),th); th=np.where(D.ncomp.values>=2,np.maximum(th,t0+h),th); th=np.where(D.p_native.values&~D.p_empty.values,np.minimum(th,t0+n),th)
    tcm=D.s2nd_q.values>=0.9; th=np.where(tcm,np.maximum(th,t0+tc[0]),th); ma=np.where(tcm,ma+tc[1],ma); return (s>=th)&(mg>=ma)
for w in (0.2,0.0,0.5):
    keepq,D,ntrue=setup(w,True); s=D.s.values; mg=s-D.smo.values
    for t in (.75,.8,.85,.9):
        row=[]
        for m in (.05,.3,.5,.7,.9):
            f,_,_=fq(D,(s>=t)&(mg>=m),ntrue); g,_,_=fq(D,rule(D,t,m),ntrue); row.append(f'm={m}: glob {f[keepq&~half].mean():.4f} rule {g[keepq&~half].mean():.4f}')
        print(f'w={w} t={t} |',' | '.join(row))
    if w==0.2:
        keepq,Dp,ntp=setup(0.2,False); g,_,_=fq(Dp,rule(Dp,.85,.5),ntp); g2,_,_=fq(Dp,rule(Dp,.85,.3),ntp); p,_,_=fq(Dp,(Dp.s.values>=.65)&(Dp.s.values-Dp.smo.values>=.05),ntp)
        print('PESSIMISTIC (no owner correction) w=0.2 held-out B: production %.4f | rule .85/.3 %.4f | rule .85/.5 %.4f'%(p[keepq&~half].mean(),g2[keepq&~half].mean(),g[keepq&~half].mean()))
