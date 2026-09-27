"""Why does v2 link +8.3% more than subA on TEST but only ~+1% on the REPLAY? Uses cached scores only.
For each dataset x country: join v2 scores and production scores on the identical candidate pairs, apply subA rule (production scores, .85/.95/.70, m .05)
and v2 rules (T/T/T-.2, m .05) with global-per-record margins, then compare slices, score distributions, top1/top2, counts. Replay rows carry labels."""
import sys,glob,json,numpy as np,pandas as pd
sys.path.insert(0,'pilot'); import make_submission as ms
RP='pilot/final_optimization/00_production_replay/'; V2='pilot/final_optimization/10_final_candidates/v2_hybrid/'; OUT=V2+'shift_analysis.json'
def code_q(s): return s.str[3:].astype(np.int64)
def code_p(s): return s.str[3:].astype(np.int64)*4+s.str[1].astype(np.int64)
def load(v2files,prodfiles,v2df=None):
    V=v2df if v2df is not None else pd.concat([pd.read_parquet(f,columns=['q_id','p_id','s','p_empty','p_native']) for f in v2files],ignore_index=True)
    V=pd.DataFrame({'q':code_q(V.q_id).values,'p':code_p(V.p_id).values,'sv':V.s.values.astype(np.float32),'emp':V.p_empty.values.astype(np.int8),'nat':V.p_native.values.astype(np.int8)})
    P=pd.concat([pd.read_parquet(f,columns=['q_id','p_id','s']) for f in prodfiles],ignore_index=True); P=pd.DataFrame({'q':code_q(P.q_id).values,'p':code_p(P.p_id).values,'sp':P.s.values.astype(np.float32)})
    V=V.sort_values('sv',ascending=False,kind='stable').drop_duplicates(['q','p']); P=P.sort_values('sp',ascending=False,kind='stable').drop_duplicates(['q','p'])
    D=V.merge(P,on=['q','p'],how='inner').reset_index(drop=True); pc=pd.factorize(D.p)[0]
    D['mv']=D.sv-ms.margin_other(pc,D.sv.values.astype(np.float64)); D['mp']=D.sp-ms.margin_other(pc,D.sp.values.astype(np.float64)); D['src']=np.where(D.p%4==2,'S2','S3'); return D
def rules(D):
    thrA=np.select([D.emp==1,D.nat==1],[.95,.70],.85); R={'subA':(D.sp>=thrA)&(D.mp>=.05)}
    for T in (.7,.8,.9): thr=np.select([D.emp==1,D.nat==1],[T,T-.2],T); R[f'v2_T{T}']=(D.sv>=thr)&(D.mv>=.05)
    return R
def summarize(D,R,nS1,lab=None):
    g=D.groupby('q').sv; top=D.sort_values(['q','sv'],ascending=[True,False]); t1=top.groupby('q').sv.nth(0).values; t2=top.groupby('q').sv.nth(1).values
    o={'S1':int(nS1),'cands_per_S1':len(D)/nS1,'v2_top1_mean':float(t1.mean()),'v2_top1_q10/50':np.quantile(t1,[.1,.5]).round(3).tolist(),'v2_top2_mean':float(t2.mean()),'v2_top1-top2_median':float(np.median(t1-t2)),
       'v2_frac_pairs_.2-.97':float(D.sv.between(.2,.97).mean()),'prod_frac_pairs_.2-.97':float(D.sp.between(.2,.97).mean()),
       **{f'v2_per_S1_>={t}':float((D.sv>=t).sum()/nS1) for t in (.7,.8,.9)},**{f'prod_per_S1_>={t}':float((D.sp>=t).sum()/nS1) for t in (.7,.8,.9)}}
    for k,m in R.items():
        o[f'{k}_per_S1']=float(m.sum()/nS1); o[f'{k}_linked']=float(D.q[m].nunique()/nS1)
    for k in ('v2_T0.7','v2_T0.8','v2_T0.9'): o[f'{k}/subA']=float(R[k].sum()/max(R['subA'].sum(),1))
    sl={}
    for name,m in {'S2':D.src=='S2','S3':D.src=='S3','empty_addr':D.emp==1,'native':D.nat==1,'normal':(D.emp==0)&(D.nat==0)}.items():
        m=m.values; sl[name]={'pairs_share':float(m.mean()),'subA_per_S1':float((R['subA'].values&m).sum()/nS1),'v2_T0.8_per_S1':float((R['v2_T0.8'].values&m).sum()/nS1),'ratio_T0.8/subA':float((R['v2_T0.8'].values&m).sum()/max((R['subA'].values&m).sum(),1))}
        if lab is not None:
            add=R['v2_T0.8'].values&~R['subA'].values&m; rem=R['subA'].values&~R['v2_T0.8'].values&m
            sl[name]['v2_additions_precision']=float(lab[add].mean()) if add.any() else None; sl[name]['v2_removals_precision']=float(lab[rem].mean()) if rem.any() else None
    o['slices']=sl
    # score-bin distribution of pairs where v2 and production disagree
    add=R['v2_T0.8']&~R['subA']; o['v2_only_T0.8_per_S1']=float(add.sum()/nS1); o['subA_only_vs_T0.8_per_S1']=float((R['subA']&~R['v2_T0.8']).sum()/nS1)
    o['v2_only_prod_score_hist']=(np.histogram(D.sp[add],[0,.2,.5,.65,.8,.85,.9,.95,1.01])[0]/max(add.sum(),1)).round(3).tolist()
    o['v2_only_empty/native/normal']=[float(D.emp[add].mean()),float(D.nat[add].mean()),float(((D.emp==0)&(D.nat==0))[add].mean())]
    o['v2_only_prod_margin<.05_share']=float((D.mp[add]<.05).mean())
    if lab is not None:
        for k,m in R.items(): o[f'{k}_precision_raw']=float(lab[m.values].mean())
        o['v2_only_T0.8_precision']=float(lab[add.values].mean()); o['subA_only_precision']=float(lab[(R['subA']&~R['v2_T0.8']).values].mean())
    return o
RES={}
s1r=pd.read_csv(RP+'replay_data/test_source1.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3,usecols=['entity_id','country']); cr=dict(zip(code_q(s1r.entity_id),s1r.country))
gt=pd.read_parquet(RP+'replay_gt.parquet'); gtk=set(zip(code_q(gt.q_id),code_p(gt.p_id)))
H=pd.read_parquet('pilot/final_optimization/02_ablation/hybrid_H5_oof.parquet'); Dr=load(None,glob.glob(RP+'work_prod/scored/*.parquet'),H); Dr['cty']=Dr.q.map(cr); lab_all=np.array([(a,b) in gtk for a,b in zip(Dr.q.values,Dr.p.values)])
for c in ('US','India'):
    m=(Dr.cty==c).values; D=Dr[m].reset_index(drop=True); RES[f'REPLAY/{c}']=summarize(D,rules(D),(s1r.country==c).sum(),lab_all[m]); print('REPLAY',c,{k:round(v,4) for k,v in RES[f'REPLAY/{c}'].items() if isinstance(v,float)},flush=True)
s1t=pd.read_csv('dataset/test/test_source1.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3,usecols=['country']).country.value_counts()
for c in ('US','India','France'):
    D=load(glob.glob(V2+f'test_scored/{c}__*.parquet'),glob.glob(f'submission_work/scored/{c}__*.parquet')); R=rules(D)
    RES[f'TEST/{c}']=summarize(D,R,s1t[c]); print('TEST',c,{k:round(v,4) for k,v in RES[f'TEST/{c}'].items() if isinstance(v,float)},flush=True)
    # cross-file ownership: records given to different S1 by subA vs v2 T0.8 (within a file exclusivity holds by the margin rule)
    a=D[R['subA'].values][['q','p']]; b=D[R['v2_T0.8'].values][['q','p']]; j=a.merge(b,on='p',suffixes=('_a','_b')); RES[f'TEST/{c}']['records_different_owner_subA_vs_T0.8']=int((j.q_a!=j.q_b).sum())
    for k in R: RES[f'TEST/{c}'][f'{k}_within_file_collisions']=int(D.p[R[k].values].duplicated().sum())
    del D,R
json.dump(RES,open(OUT,'w'),indent=1,default=float); print('saved')
