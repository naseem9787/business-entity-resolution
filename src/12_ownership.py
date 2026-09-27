"""Exp1 step 5-7: B0 (cross-fitted pairwise LightGBM) + ownership variants A / B / D / E + diagnostics.
Everything (scores AND thresholds) is cross-fitted over the 5 state folds. LightGBM params identical to 04_match.py."""
import sys,os,json,time,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0,'pilot')
from metric import f05_per_query,f05_from_pred,query_counts,paired_bootstrap,oracle_attribution
BENCH='pilot/exp1/'; OUT=os.environ.get('EXP_OUT','pilot/exp1/'); FEATFILE=os.environ.get('EXP_FEATS','feats_top15.parquet'); TAG=os.environ.get('EXP_TAG',''); T0=time.time(); NB=1000
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
LGB=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=8); ROUNDS=400
G=np.round(np.arange(0.30,0.9501,0.025),3)
# ------------------------------------------------------------------ load
X=pd.read_parquet(OUT+FEATFILE).sort_values('fold',kind='stable').reset_index(drop=True)
S1=pd.read_parquet(BENCH+'bench_s1.parquet',columns=['id','country','fold','nc','state']); nq=len(S1); qmap=dict(zip(S1.id,range(nq)))
ntrue=pd.read_parquet(BENCH+'ntrue.parquet').ntrue.reindex(S1.id).fillna(0).values.astype(float)
qfold=S1.fold.values.astype(int); qcountry=S1.country.values
owner={}
for s in ('s2','s3'): p=pd.read_parquet(BENCH+f'bench_{s}.parquet',columns=['id','owner']); owner.update(zip(p.id,p.owner))
qi=X.q_id.map(qmap).values.astype(np.int64); label=X.label.values.astype(np.int8); rf=X.fold.values.astype(int); N=len(X)
pcode=pd.factorize(X.p_id)[0].astype(np.int64); ncq=pd.factorize(S1.nc)[0]
in_cands=np.bincount(qi[label==1],minlength=nq).astype(float)
META=['q_id','p_id','label','country','src','fold']; FEATS=[c for c in X.columns if c not in META]
XF=X[FEATS].to_numpy(np.float32); fi={c:i for i,c in enumerate(FEATS)}
row_country=X.country.values; row_src=X.src.values; miss_addr=X.a_miss_p.values>0; native=X.n_script_diff.values>0
row_owner=X.p_id.map(owner).fillna('').values; is_unowned=row_owner==''; owned_other=(~is_unowned)&(row_owner!=X.q_id.values)
X=X[META].copy(); import gc; gc.collect()
log('rows',N,'queries',nq,'features',len(FEATS),'true pairs',int(ntrue.sum()),'in top15',int(label.sum()),'cand recall',round(label.sum()/ntrue.sum(),4))
def fit_lgb(Xm,y,rounds=ROUNDS): return lgb.train(LGB,lgb.Dataset(Xm,y,free_raw_data=True),rounds)
# ------------------------------------------------------------------ competitor statistics (scope = a set of rows)
class Scope:
    def __init__(s,idx,sc):
        s.idx=idx; s.s=sc[idx].astype(np.float64); s.n=len(idx); pc=pcode[idx]; s.pg=pd.factorize(pc)[0]
        order=np.lexsort((-s.s,s.pg)); pg=s.pg[order]; ss=s.s[order]; start=np.r_[True,pg[1:]!=pg[:-1]]; gs=np.flatnonzero(start); gsz=np.diff(np.r_[gs,len(ss)]); gid=np.cumsum(start)-1
        g_=lambda k: np.where(gsz>k,ss[np.minimum(gs+k,len(ss)-1)],0.0); t1,t2,t3=g_(0),g_(1),g_(2)
        pos=np.arange(len(ss))-gs[gid]; i1=order[gs]; i2=np.where(gsz>1,order[np.minimum(gs+1,len(ss)-1)],-1)
        smo=np.where(pos==0,t2[gid],t1[gid]); s2o=np.where(pos==0,t3[gid],np.where(pos==1,t3[gid],t2[gid])); top=np.where(pos==0,i2[gid],i1[gid]); top=np.where(smo>0,top,np.where(gsz[gid]>1,top,-1))
        def back(v): o=np.empty_like(v); o[order]=v; return o
        s.smo=back(smo); s.s2o=back(s2o); s.top=back(top).astype(np.int64); s.rank=back(pos+1).astype(float); s.ncomp=back((gsz[gid]-1).astype(float))
        gt5=np.bincount(s.pg,weights=(s.s>0.5)).astype(float); s.n_gt5=gt5[s.pg]-(s.s>0.5); sm=np.bincount(s.pg,weights=s.s); s.sum_others=sm[s.pg]-s.s
        s.margin=s.s-s.smo; s.qi=qi[idx]; qs=pd.Series(s.s); g=qs.groupby(s.qi)
        s.q_top1=g.transform('max').values; s.q_gt5=g.transform(lambda x:(x>0.5).sum()).values.astype(float); s.p_rank_in_q=g.rank(ascending=False,method='first').values
    def post(s,iso):
        c=np.clip(iso.predict(s.s),1e-4,1-1e-4); odds=c/(1-c); so=np.bincount(s.pg,weights=odds); return odds/(1+so[s.pg]),c
    def block(s,kind,iso,sel=None):
        sel=np.ones(s.n,bool) if sel is None else sel
        post,_=s.post(iso)
        B=np.column_stack([s.s,s.smo,s.s2o,s.margin,s.ncomp,s.n_gt5,s.rank,post,s.sum_others])[sel].astype(np.float32)
        if kind=='B': return B
        gi=s.idx[np.maximum(s.top,0)][sel]; has=(s.top>=0)[sel]
        cf=lambda name: np.where(has,XF[gi,fi[name]],np.nan)
        chain=np.where(has,(ncq[qi[gi]]==ncq[s.qi[sel]]).astype(float),np.nan)
        E=np.column_stack([s.q_top1[sel],s.q_gt5[sel],s.p_rank_in_q[sel],cf('n_lev_nc'),cf('a_lev'),cf('n_eq_nc'),chain]).astype(np.float32)
        return np.hstack([XF[s.idx[sel]],B,E])
def variant(kind,delta,fitS,applyS,fit_sel,apply_sel):
    """returns scores in [0,1] for applyS rows selected by apply_sel"""
    if kind=='A': return np.where(applyS.margin[apply_sel]>=delta,applyS.s[apply_sel],0.0)
    yfit=label[fitS.idx][fit_sel]; iso=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(fitS.s[fit_sel],yfit)
    if kind=='D': return applyS.post(iso)[0][apply_sel]
    m=fit_lgb(fitS.block(kind,iso,fit_sel),yfit); return m.predict(applyS.block(kind,iso,apply_sel))
# ------------------------------------------------------------------ B0 : stage-1 cross-fit
s0=np.zeros(N)
for k in range(5):
    tr=rf!=k; te=rf==k; m=fit_lgb(XF[tr],label[tr]); s0[te]=m.predict(XF[te]); log('B0 fold',k,'trained',tr.sum(),'->',te.sum()); del m
imp=None
# ------------------------------------------------------------------ evaluation helpers
def crossfit(score,name):
    Fq=np.empty((len(G),nq),np.float32)
    for i,t in enumerate(G): Fq[i]=f05_from_pred(qi,score>=t,label,ntrue,nq)
    ins=Fq.mean(1); fcf=np.zeros(nq); thr=[]
    for k in range(5):
        sel=qfold!=k; ti=int(Fq[:,sel].mean(1).argmax()); thr.append(float(G[ti])); fcf[qfold==k]=Fq[ti,qfold==k]
    pred=score>=np.array(thr)[rf]
    return dict(name=name,F05_crossfit=float(fcf.mean()),F05_insample_best=float(ins.max()),thr_insample=float(G[ins.argmax()]),thr_per_fold=thr,fq=fcf,pred=pred)
def metrics(pred,ref_pred=None):
    tp,npred=query_counts(qi,pred,label,nq); f=f05_per_query(tp,npred,ntrue); single=ntrue==0
    P=np.where(npred>0,tp/np.maximum(npred,1),np.nan); R=np.where(ntrue>0,tp/np.maximum(ntrue,1),np.nan)
    m=dict(F05=float(f.mean()),precision_micro=float(tp.sum()/max(npred.sum(),1)),recall_micro=float(tp.sum()/ntrue.sum()),precision_macro=float(np.nanmean(P)),recall_macro=float(np.nanmean(R)),
           singleton_acc=float((npred[single]==0).mean()),false_merges=int(npred.sum()-tp.sum()),missed_total=int(ntrue.sum()-tp.sum()),missed_blocking_lost=int((ntrue-in_cands).sum()),
           n_pred=int(npred.sum()))
    rej=(label==1)&~pred; m['missed_model_rejected']=int(rej.sum())
    if ref_pred is not None: m['missed_of_which_removed_by_variant_vs_B0']=int((rej&ref_pred).sum())
    m['by_country']={c:dict(F05=float(f[qcountry==c].mean()),precision_micro=float((pred&(label==1)&(row_country==c)).sum()/max((pred&(row_country==c)).sum(),1)),recall_micro=float((pred&(label==1)&(row_country==c)).sum()/max(ntrue[qcountry==c].sum(),1)),false_merges=int((pred&(label==0)&(row_country==c)).sum()),singleton_acc=float((npred[single&(qcountry==c)]==0).mean())) for c in np.unique(qcountry)}
    m['by_source']={s:dict(precision_micro=float((pred&(label==1)&(row_src==s)).sum()/max((pred&(row_src==s)).sum(),1)),recall_of_candidates=float((pred&(label==1)&(row_src==s)).sum()/max(((label==1)&(row_src==s)).sum(),1)),false_merges=int((pred&(label==0)&(row_src==s)).sum())) for s in ('s2','s3')}
    bk=np.select([ntrue==0,ntrue==1,ntrue==2,ntrue<=5],['0','1','2','3-5'],'6+'); m['F05_by_match_bucket']={b:float(f[bk==b].mean()) for b in ('0','1','2','3-5','6+')}
    m['empty_addr_pairs']=dict(pred_true_pos=int((pred&(label==1)&miss_addr).sum()),false_merges=int((pred&(label==0)&miss_addr).sum()),rejected_true=int((rej&miss_addr).sum()),precision=float((pred&(label==1)&miss_addr).sum()/max((pred&miss_addr).sum(),1)))
    return m
R={'meta':dict(rows=N,queries=nq,features=len(FEATS),true_pairs=int(ntrue.sum()),pairs_in_top15=int(label.sum()),folds=np.bincount(qfold).tolist(),grid=G.tolist())}
b0=crossfit(s0,'B0'); m0=metrics(b0['pred']); R['B0']={k:v for k,v in b0.items() if k not in('fq','pred')}; R['B0']['metrics']=m0
log('B0 cross-fitted macro F0.5',round(b0['F05_crossfit'],4),'in-sample best',round(b0['F05_insample_best'],4),'thr',b0['thr_per_fold'],'| P',round(m0['precision_micro'],4),'R',round(m0['recall_micro'],4),'singleton',round(m0['singleton_acc'],4))
allS=Scope(np.arange(N),s0); log('competitor stats built; rows with >=1 competitor:',round(float((allS.ncomp>0).mean()),3))
VAR={'B0':b0}; VSET=os.environ.get('EXP_VARIANTS','A,D,B,E').split(','); LOCO_ON=os.environ.get('EXP_LOCO','1')=='1'
# ---- A
for d in ((0.0,0.05,0.1,0.2) if 'A' in VSET else ()):
    sc=variant('A',d,None,allS,None,np.ones(N,bool)); VAR[f'A(d={d})']=crossfit(sc,f'A(d={d})'); log('A',d,round(VAR[f'A(d={d})']['F05_crossfit'],4))
# ---- D / B / E (fold-wise fit on 4 folds, apply on held-out fold; scope = all rows so competitors from any fold are visible)
oof={}
for kind in [k for k in ('D','B','E') if k in VSET]:
    sc=np.zeros(N)
    for k in range(5):
        tr=rf!=k; te=rf==k; sc[te]=variant(kind,None,allS,allS,tr,te); log(kind,'fold',k,'done')
    oof[kind]=sc; VAR[kind]=crossfit(sc,kind); log(kind,'cross-fitted F0.5',round(VAR[kind]['F05_crossfit'],4))
if len(VAR)==1:   # B0-only mode (used for budget curves): save and stop
    R['mode']='B0_only'; json.dump(R,open(OUT+f'results{TAG}.json','w'),indent=1,default=float)
    pd.DataFrame({'q_id':X.q_id,'p_id':X.p_id,'src':X.src,'country':X.country,'fold':X.fold,'label':label,'s_B0':s0,'pred_B0':b0['pred']}).to_parquet(OUT+f'oof_scores{TAG}.parquet'); log('B0-only saved'); sys.exit(0)
# ------------------------------------------------------------------ report per variant (metrics + paired bootstrap vs B0)
for n,v in VAR.items():
    if n=='B0': continue
    met=metrics(v['pred'],b0['pred']); d,lo,hi=paired_bootstrap(b0['fq'],v['fq'],NB); R[n]={k:x for k,x in v.items() if k not in('fq','pred')}; R[n]['metrics']=met; R[n]['gain_vs_B0']=dict(delta=d,ci95=[lo,hi])
    log(f'{n:10s} F0.5 {v["F05_crossfit"]:.4f}  gain {d:+.4f} CI[{lo:+.4f},{hi:+.4f}]  P {met["precision_micro"]:.4f} R {met["recall_micro"]:.4f} singleton {met["singleton_acc"]:.4f}')
own_vars=[n for n in VAR if n!='B0']; best=max(own_vars,key=lambda n:VAR[n]['F05_crossfit']); R['best_variant']=best
cand=[n for n in own_vars if n.startswith('A') or n=='D']; simple=max(cand,key=lambda n:VAR[n]['F05_crossfit']) if cand else best
R['best_simple_variant']=simple; R['E_minus_best_simple']=(VAR['E']['F05_crossfit']-VAR[simple]['F05_crossfit']) if ('E' in VAR and cand) else None
pick=best if not (best in('E','B') and cand and VAR[best]['F05_crossfit']-VAR[simple]['F05_crossfit']<0.0015) else simple; R['recommended_variant_by_simplicity_rule']=pick
g=R[pick]['gain_vs_B0']; mp=R[pick]['metrics']
R['acceptance']=dict(variant=pick,gain=g['delta'],ci_lower_gt_0=g['ci95'][0]>0,gain_ge_0_002=g['delta']>=0.002,singleton_not_worse=mp['singleton_acc']>=m0['singleton_acc']-1e-9,micro_recall_drop=m0['recall_micro']-mp['recall_micro'],
    recall_drop_le_0_003=(m0['recall_micro']-mp['recall_micro'])<=0.003)
R['acceptance']['KEEP_OWNERSHIP']=bool(R['acceptance']['gain_ge_0_002'] and R['acceptance']['ci_lower_gt_0'] and R['acceptance']['singleton_not_worse'] and R['acceptance']['recall_drop_le_0_003'])
log('ACCEPTANCE',R['acceptance'])
# ------------------------------------------------------------------ oracle attribution (B0 and best) with splits
fpm={'unowned':is_unowned,'owned_by_other_S1':owned_other,'empty_S2S3_addr':miss_addr,'India':row_country=='India','US':row_country=='US'}
fnm={'empty_S2S3_addr':miss_addr,'native_script_vs_latin':native,'India':row_country=='India','US':row_country=='US'}
R['oracle_attribution']={n:oracle_attribution(qi,VAR[n]['pred'],label,ntrue,nq,in_cands,fpm,fnm) for n in ('B0',pick)}
for n,r in R['oracle_attribution'].items():
    log('ORACLE',n); [log(f'   {k:45s} F0.5 {v[0]:.4f} (+{v[1]:.4f})') for k,v in r.items()]
# ------------------------------------------------------------------ false-merge taxonomy for B0
pr=b0['pred']; fpi=np.flatnonzero(pr&(label==0)); pk=X.p_id.values[fpi]; ow=row_owner[fpi]; sfp=s0[fpi]
key_s=pd.Series(s0,index=pd.MultiIndex.from_arrays([X.q_id.values,X.p_id.values]))
key_s=key_s[~key_s.index.duplicated()]
s_own=key_s.reindex(pd.MultiIndex.from_arrays([ow,pk])).values
cat=np.where(ow=='','unowned (no S1 owns it)',np.where(np.isnan(s_own),'owned by other region-S1, p NOT in owner top-15',np.where(s_own>sfp,'owned by other S1, owner scores it HIGHER (ownership-resolvable)','owned by other S1, owner scores it lower/equal')))
tax=pd.Series(cat).value_counts(); R['false_merge_taxonomy_B0']=dict(counts=tax.to_dict(),total_false_merges=int(len(fpi)),share={k:float(v/len(fpi)) for k,v in tax.items()})
resolvable=float((cat=='owned by other S1, owner scores it HIGHER (ownership-resolvable)').mean()); R['false_merge_taxonomy_B0']['resolvable_by_owner_score_share']=resolvable
rem=lambda n: float((pr&~VAR[n]['pred']&(label==0)).sum()/max(len(fpi),1)); R['false_merge_taxonomy_B0']['fraction_of_B0_false_merges_removed']={n:rem(n) for n in own_vars}
R['false_merge_taxonomy_B0']['true_pairs_lost_by_variant']={n:int((pr&~VAR[n]['pred']&(label==1)).sum()) for n in own_vars}
R['false_merge_taxonomy_B0']['fp_with_any_competitor_share']=float((allS.ncomp[fpi]>0).mean()); R['pivot_signal_lt_25pct_resolvable']=bool(rem(pick)<0.25)
log('TAXONOMY',R['false_merge_taxonomy_B0']['counts'],'| removed by',{n:round(rem(n),3) for n in own_vars})
# ------------------------------------------------------------------ density effect vs pilot
R['density_effect']=dict(pilot_F05_at_9pct_density=0.9705,B0_here_crossfit=b0['F05_crossfit'],delta=b0['F05_crossfit']-0.9705)
# ------------------------------------------------------------------ leave-one-country-out (B0 and best variant)
R['LOCO']={}
def loco(tr_c,te_c):
    tri=np.flatnonzero(row_country==tr_c); tei=np.flatnonzero(row_country==te_c)
    m=fit_lgb(XF[tri],label[tri]); s_te=m.predict(XF[tei]); del m
    q_te=np.unique(qi[tei]); qsel=np.zeros(nq,bool); qsel[q_te]=True; qtr=np.zeros(nq,bool); qtr[np.unique(qi[tri])]=True
    def Fb(mask_rows,score_rows,rows,qsel_):
        best=(-1,0);
        for t in G:
            pm=np.zeros(N,bool); pm[rows]=score_rows>=t; f=f05_from_pred(qi,pm,label,ntrue,nq)[qsel_].mean()
            if f>best[0]: best=(f,t)
        return best
    thr_src=b0['thr_insample'] if False else Fb(None,s0[tri],tri,qtr)[1]      # threshold chosen on the training country's OOF scores
    out={}
    Sloco=s0.copy(); Sloco[tei]=s_te
    pm=np.zeros(N,bool); pm[tei]=s_te>=thr_src; out['B0']=dict(F05_at_train_country_thr=float(f05_from_pred(qi,pm,label,ntrue,nq)[qsel].mean()),thr=float(thr_src),F05_oracle_thr=float(Fb(None,s_te,tei,qsel)[0]),
        F05_in_country_crossfit=float(b0['fq'][qsel].mean()))
    # best variant
    kind=pick if pick not in('D','B','E') else pick
    if pick.startswith('A'):
        d=float(pick.split('=')[1][:-1]); fitS=None; ap=Scope(tei,Sloco); sc=variant('A',d,None,ap,None,np.ones(len(tei),bool)); trS=Scope(tri,s0); sc_tr=variant('A',d,None,trS,None,np.ones(len(tri),bool)); kk='A'
    else:
        trS=Scope(tri,s0); ap=Scope(tei,Sloco); sc=variant(pick,None,trS,ap,np.ones(len(tri),bool),np.ones(len(tei),bool)); sc_tr=None
    if sc_tr is None: thr_v=float(np.mean(R[pick]['thr_per_fold']))
    else: thr_v=Fb(None,sc_tr,tri,qtr)[1]
    pm=np.zeros(N,bool); pm[tei]=sc>=thr_v; out[pick]=dict(F05_at_thr=float(f05_from_pred(qi,pm,label,ntrue,nq)[qsel].mean()),thr=thr_v,F05_oracle_thr=float(Fb(None,sc,tei,qsel)[0]),F05_in_country_crossfit=float(VAR[pick]['fq'][qsel].mean()))
    return out
for a,b in ((('US','India'),('India','US')) if LOCO_ON else ()):
    R['LOCO'][f'train {a} -> test {b}']=loco(a,b); log('LOCO',a,'->',b,R['LOCO'][f'train {a} -> test {b}'])
# ------------------------------------------------------------------ artifacts
o=pd.DataFrame({'q_id':X.q_id,'p_id':X.p_id,'src':X.src,'country':X.country,'fold':X.fold,'label':label,'s_B0':s0,'smo':allS.smo,'margin':allS.margin,'ncomp':allS.ncomp,**{f's_{k}':v for k,v in oof.items()},'pred_B0':b0['pred'],f'pred_{pick}':VAR[pick]['pred']})
o.to_parquet(OUT+f'oof_scores{TAG}.parquet'); json.dump(R,open(OUT+f'results{TAG}.json','w'),indent=1,default=float); log('saved results.json + oof_scores.parquet')
