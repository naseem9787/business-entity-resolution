"""Exp2: cross-country generalisation of the learned blocking ranker (open-set proxy for France).
Trains the ranker on ONE country only and measures absolute recall on the OTHER country; repeats with the cross-query 'competition' features removed.
Compares with the in-country cross-fitted ranker (H) and with the hand-weighted G / baseline A. Re-uses the helpers of 21_rank_eval.py (exec of its header)."""
import sys,json,gc,time
src=open('pilot/21_rank_eval.py',encoding='utf8').read().split('# ===========================================================================================')[0]; exec(src)
import numpy as np,pandas as pd,lightgbm as lgb
SUMM=json.load(open(OUT+'ablation_summary.json')); TOT=SUMM['totals']
COMP=['gap_p_fused','p_freq','rev_rank','gap_q_fused','q_union']      # features that need the whole S1/pool population (scale-sensitive)
def sample(combos):
    Ms=[];ys=[];ns=None
    for c,s in combos:
        D,qinfo=load(c,s); cols=ranker_matrix_cols(D); Gs=rule_scores(D)['G_all_channels'][1]; idx,rk=rank_within_query(D.qi.values,Gs,np.ones(len(D),bool)); rank=np.empty(len(D),np.int32); rank[idx]=rk
        keep=(D.label.values==1)|(rank<=25)|(np.random.default_rng(1).random(len(D))<0.02); ns=list(cols)
        Ms.append(np.column_stack([cols[k][keep] for k in ns]+[np.full(keep.sum(),int(s=='s3'),np.float32)]).astype(np.float32)); ys.append(D.label.values[keep]); del D,cols; gc.collect()
    return np.vstack(Ms),np.concatenate(ys),ns+['src3']
out={}
for tr_c,te_c in (('US','India'),('India','US')):
    Xs,ys,names=sample([(c,s) for c,s in COMBOS if c==tr_c])
    for tag,drop in (('all_features',[]),('no_competition_features',COMP)):
        cols_i=[i for i,n in enumerate(names) if n not in drop]; m=lgb.train(LGB,lgb.Dataset(Xs[:,cols_i],ys),RANK_ROUNDS); found={5:0,10:0,15:0,30:0}; tot=0
        for c,s in [(c,s) for c,s in COMBOS if c==te_c]:
            D,qinfo=load(c,s); cols=ranker_matrix_cols(D); names_=list(cols); src3=np.full(len(D),int(s=='s3'),np.float32); H=np.zeros(len(D),np.float32)
            for a in range(0,len(D),1_500_000):
                b=min(a+1_500_000,len(D)); M=np.column_stack([cols[k][a:b] for k in names_]+[src3[a:b]]).astype(np.float32); H[a:b]=m.predict(M[:,cols_i]); del M
            idx,rk=rank_within_query(D.qi.values,H,np.ones(len(D),bool)); lab=D.label.values[idx]==1
            for K in found: found[K]+=int(((rk<=K)&lab).sum())
            tot+=TOT[f'{c}/{s}']['true']; del D,cols,H; gc.collect()
        out[f'train {tr_c} -> test {te_c} [{tag}]']={K:round(v/tot,4) for K,v in found.items()}; print(f'train {tr_c} -> test {te_c} [{tag}] absolute recall',out[f'train {tr_c} -> test {te_c} [{tag}]'],flush=True)
    del Xs,ys; gc.collect()
# in-country references from the ablation summary
for cfg in ('A_baseline4','G_all_channels','H_learned_ranker'):
    tb=SUMM[f'budget_table_{cfg}']; out[f'reference {cfg}']={ctry:{r['K']:r[ctry] for r in tb if r['K'] in (5,10,15,30)} for ctry in ('India','US')}
json.dump(out,open(OUT+'ranker_loco.json','w'),indent=1,default=float); print(json.dumps(out['reference H_learned_ranker']))
