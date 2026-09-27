import numpy as np
exec(open('pilot/final_optimization/00_production_replay/margin_ext.py',encoding='utf8').read().split('for w in')[0])
for w in (0.2,0.0,0.5):
    for corr in (True,False):
        keepq,D,ntrue=setup(w,corr); out=[]
        for name,p in (('production .65/.05',(D.s.values>=.65)&(D.s.values-D.smo.values>=.05)),('thr-only rule .85/.05',rule(D,.85,.05,h=0,tc=(0,0))),('rule .85/.30',rule(D,.85,.3,h=0,tc=(0,0)))):
            f,tp,npd=fq(D,p,ntrue); out.append(f'{name}: B {f[keepq&~half].mean():.4f} US {f[keepq&~half&(qc=="US")].mean():.4f} IN {f[keepq&~half&(qc=="India")].mean():.4f} single {(npd[keepq&(ntrue==0)]==0).mean():.3f} m/S1 {npd[keepq].mean():.2f}')
        print(f'w={w} {"corrected" if corr else "pessimist"} |',' | '.join(out))
