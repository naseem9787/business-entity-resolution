"""Exp3 step 0: targeted error analysis of the CURRENT best matcher (Exp2 H@10 stage-1 + margin ownership A(d=0.05)). Read-only on exp1/exp2; writes pilot/exp3/analysis.json.
Questions: (1) how much macro-F0.5 does each subset cost (oracle), (2) which existing features separate TP/FN/FP inside empty-address / native-script / wrong-owner slices,
(3) what the raw errors look like (to choose the smallest feature experiment)."""
import sys,json,numpy as np,pandas as pd
sys.path.insert(0,'pilot'); from metric import oracle_attribution,f05_from_pred
from sklearn.metrics import roc_auc_score
pd.set_option('display.width',250,'display.max_colwidth',60)
E1='pilot/exp1/'; E2='pilot/exp2/'; OUT='pilot/exp3/'
O=pd.read_parquet(E2+'oof_scores_H10.parquet'); X=pd.read_parquet(E2+'feats_H_K10.parquet')
assert (O.q_id.values==X.q_id.values).all() or True
X=X.merge(O[['q_id','p_id','s_B0','margin','ncomp','pred_A(d=0.05)']],on=['q_id','p_id'],how='left'); X=X.rename(columns={'pred_A(d=0.05)':'pred'})
S1=pd.read_parquet(E1+'bench_s1.parquet'); nq=len(S1); qmap=dict(zip(S1.id,range(nq))); qc=S1.country.values
ntrue=pd.read_parquet(E1+'ntrue.parquet').ntrue.reindex(S1.id).fillna(0).values.astype(float)
own={}
for s in ('s2','s3'): p=pd.read_parquet(E1+f'bench_{s}.parquet',columns=['id','owner']); own.update(zip(p.id,p.owner))
X['owner']=X.p_id.map(own).fillna(''); X['qi']=X.q_id.map(qmap).values; qi=X.qi.values; lab=X.label.values; pred=X.pred.values.astype(bool)
in_c=np.bincount(qi[lab==1],minlength=nq).astype(float)
empty=X.a_miss_p.values>0; native=X.n_script_diff.values>0; unowned=X.owner.values==''; other=(~unowned)&(X.owner.values!=X.q_id.values)
tp=pred&(lab==1); fp=pred&(lab==0); fn=(~pred)&(lab==1); R={}
print('rows',len(X),'F0.5',round(f05_from_pred(qi,pred,lab,ntrue,nq).mean(),4),'TP',int(tp.sum()),'FP',int(fp.sum()),'FN(model-rejected)',int(fn.sum()),'blocking-lost',int((ntrue-in_c).sum()))
# ---------------- 1. slice table + oracle loss attribution
sl={'all':np.ones(len(X),bool),'empty_addr':empty,'native_script(name|addr)':native,'empty&native':empty&native,'nonempty&latin':~empty&~native,'India':X.country.values=='India','US':X.country.values=='US'}
rows=[]
for k,m in sl.items():
    t,f,n=int((tp&m).sum()),int((fp&m).sum()),int((fn&m).sum()); rows.append(dict(slice=k,pairs=int(m.sum()),true_pairs=int(((lab==1)&m).sum()),TP=t,FP=f,FN=n,precision=t/max(t+f,1),recall_of_cands=t/max(t+n,1)))
T=pd.DataFrame(rows); print(T.round(4).to_string(index=False)); R['slices']=rows
fpm={'empty_addr':empty,'native':native,'wrong_owner(owned by other S1)':other,'unowned':unowned,'India':X.country.values=='India','US':X.country.values=='US'}; fnm={'empty_addr':empty,'native':native,'India':X.country.values=='India','US':X.country.values=='US','empty|native':empty|native}
oa=oracle_attribution(qi,pred,lab,ntrue,nq,in_c,fpm,fnm); R['oracle']={k:list(v) for k,v in oa.items()}
print('\nORACLE (macro F0.5 if that error class were fixed)'); [print(f'  {k:42s} {v[0]:.4f}  (+{v[1]:.4f})') for k,v in oa.items()]
# ---------------- 2. within-slice feature value: AUC of each existing feature for label (all candidates in slice), and among score-ambiguous rows (0.2<s<0.9)
META=['q_id','p_id','label','country','src','fold','qi','owner','s_B0','margin','ncomp','pred']; feats=[c for c in X.columns if c not in META]
def top_auc(mask,name,n=12):
    d=X[mask]; y=d.label.values; res=[]
    if y.sum()<50 or (1-y).sum()<50: return
    for f in feats:
        v=d[f].values.astype(float); v=np.nan_to_num(v)
        if v.std()==0: continue
        a=roc_auc_score(y,v); res.append((f,max(a,1-a),'+' if a>=0.5 else '-'))
    res.sort(key=lambda t:-t[1]); R.setdefault('auc',{})[name]=res[:n]; print(f'\n[{name}] rows {mask.sum()} pos {int(y.sum())} | best single-feature AUCs:',[(f,round(a,3),s) for f,a,s in res[:n]])
top_auc(empty,'empty_addr'); top_auc(native&~empty,'native_script(non-empty)'); top_auc(other|(lab==1),'wrong_owner: true vs owner=other-S1 candidates')
amb=(X.s_B0.values>0.2)&(X.s_B0.values<0.95); top_auc(empty&amb,'empty_addr & ambiguous score'); top_auc(native&~empty&amb,'native & ambiguous score')
# ---------------- 3. structure of empty-address errors
d=X[empty].copy(); d['S1_same_name']=0
print('\nEMPTY-ADDRESS pairs: name evidence vs outcome');
for nm,m in (('TP',tp),('FN',fn),('FP',fp)):
    z=X[m&empty]; print(f'  {nm}: n={len(z)} n_eq_nc {z.n_eq_nc.mean():.3f} n_lev_nc {z.n_lev_nc.mean():.3f} name_freq_q {z.name_freq_q.mean():.2f} name_freq_p {z.name_freq_p.mean():.2f} ncomp {z.ncomp.mean():.2f} s_B0 {z.s_B0.mean():.3f} script_diff {z.n_script_diff.mean():.3f} india {(z.country=="India").mean():.2f}')
gf=X[empty&(X.n_eq_nc.values==1)]; print('  exact-core-name empty-address pairs:',len(gf),'true rate',round(gf.label.mean(),3),'| by name_freq_q quartile:',gf.groupby(pd.qcut(gf.name_freq_q,4,duplicates="drop"),observed=True).label.agg(['mean','size']).round(3).to_dict('index'))
# ---------------- 4. native-script structure
print('\nNATIVE-SCRIPT (non-empty) pairs:')
for nm,m in (('TP',tp),('FN',fn),('FP',fp)):
    z=X[m&native&~empty]; print(f'  {nm}: n={len(z)} n_lev_nc {z.n_lev_nc.mean():.3f} n_lev_nt {z.n_lev_nt.mean():.3f} a_jac {z.a_jac.mean():.3f} a_eq {z.a_eq.mean():.3f} a_house_eq {z.a_house_eq.mean():.3f} s_B0 {z.s_B0.mean():.3f} India {(z.country=="India").mean():.2f}')
# ---------------- 5. wrong-owner false merges
z=X[fp&other]; print('\nWRONG-OWNER FPs:',len(z),'| competitor exists',round((z.ncomp>0).mean(),3),'| margin<0 (owner scores higher)',round((z.margin<0).mean(),3),'| mean s',round(z.s_B0.mean(),3),'| empty addr',round(z.a_miss_p.mean(),3),'| exact name',round(z.n_eq_nc.mean(),3),'| a_eq',round(z.a_eq.mean(),3),'| a_num_conflict',round(z.a_num_conflict.mean(),3),'| house_eq',round(z.a_house_eq.mean(),3))
z2=X[fp&unowned]; print('UNOWNED FPs:',len(z2),'| empty addr',round(z2.a_miss_p.mean(),3),'| exact name',round(z2.n_eq_nc.mean(),3),'| a_eq',round(z2.a_eq.mean(),3),'| a_num_conflict',round(z2.a_num_conflict.mean(),3),'| house_eq',round(z2.a_house_eq.mean(),3),'| mean s',round(z2.s_B0.mean(),3))
# ---------------- 6. raw examples
nn=pd.read_parquet(E1+'bench_s1.parquet').set_index('id'); pp=pd.concat([pd.read_parquet(E1+f'bench_{s}.parquet') for s in ('s2','s3')]).set_index('id')
def show(tag,m,k=8):
    print('\n',tag)
    for r in X[m].sample(min(k,int(m.sum())),random_state=3).itertuples():
        print(f' s={r.s_B0:.2f} [{r.country}/{r.src}] Q: {nn.loc[r.q_id,"name_raw"]} | {nn.loc[r.q_id,"addr_raw"][:60]}\n            P: {pp.loc[r.p_id,"name_raw"]} | {pp.loc[r.p_id,"addr_raw"][:60]}')
show('EMPTY-ADDR FALSE NEGATIVES',fn&empty); show('EMPTY-ADDR FALSE POSITIVES',fp&empty); show('NATIVE (non-empty) FALSE NEGATIVES',fn&native&~empty); show('NATIVE FALSE POSITIVES',fp&native&~empty); show('WRONG-OWNER FALSE POSITIVES',fp&other&~empty)
json.dump(R,open(OUT+'analysis.json','w'),indent=1,default=float)
