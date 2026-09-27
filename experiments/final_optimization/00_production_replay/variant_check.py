import numpy as np
exec(open('pilot/final_optimization/00_production_replay/margin_ext.py',encoding='utf8').read().split('for w in')[0])
for w in (0.2,0.0,0.5):
    for corr in (True,False):
        keepq,D,ntrue=setup(w,corr); r={}
        for name,p in (('production',(D.s.values>=.65)&(D.s.values-D.smo.values>=.05)),('full rule',rule(D,.85,.3)),('no high_comp',rule(D,.85,.3,h=0)),('no high_comp,no native relax',rule(D,.85,.3,h=0,n=0)),('empty only',rule(D,.85,.3,h=0,n=0,tc=(0,0)))):
            f,_,npd=fq(D,p,ntrue); r[name]=f'{f[keepq&~half].mean():.4f} (US {f[keepq&~half&(qc=="US")].mean():.4f} IN {f[keepq&~half&(qc=="India")].mean():.4f})'
        print(f'w={w} {"corrected" if corr else "pessimist"}:',r)
print('replay ncomp distribution:',np.bincount(np.minimum(D.ncomp.values,5))/len(D))
