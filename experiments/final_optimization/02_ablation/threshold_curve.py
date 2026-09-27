import glob,numpy as np,pandas as pd
exec(open('pilot/final_optimization/02_ablation/train_eval_v2.py',encoding='utf8').read().split('RES=[]')[0])
H=pd.read_parquet(OUT+'hybrid_H5_oof.parquet'); D=prep(H)
T=pd.concat([pd.read_parquet(f) for f in glob.glob('pilot/final_optimization/10_final_candidates/v2_hybrid/test_scored/*.parquet')],ignore_index=True).sort_values('s',ascending=False).drop_duplicates(['q_id','p_id']).reset_index(drop=True)
T['smo']=ms.margin_other(pd.factorize(T.p_id)[0],T.s.values.astype(np.float64))
for tb in (.70,.75,.80,.85,.90):
    g=(tb,tb,max(tb-.2,.3)); f,tp,npd=fq(D,rule(D,*g)); sel=keepq&~half
    thr=np.select([T.p_empty.values==1,T.p_native.values==1],[g[1],g[2]],g[0]); nt=int(((T.s.values>=thr)&(T.s.values-T.smo.values>=.05)).sum())
    print(f'T_BASE {tb:.2f}: replay B F0.5 {f[sel].mean():.4f} US {f[sel&(qc=="US")].mean():.4f} IN {f[sel&(qc=="India")].mean():.4f} P {tp[sel].sum()/npd[sel].sum():.4f} | TEST matches {nt} (subA 5313551, ratio {nt/5313551:.3f})',flush=True)
