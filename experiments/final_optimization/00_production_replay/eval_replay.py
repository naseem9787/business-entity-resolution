"""Evaluate production-replay scores against training ground truth.
usage: python eval_replay.py <work_dir_name> [<tag>]
A: production decision (thr/delta from production meta) -> overall / country / source / entity bucket / difficulty slices, FP taxonomy
B: density ablation by withholding a fraction w of replay S1 (their records become unowned distractors, competitors disappear); density = 1/(1-w)
D: decision-layer grid (threshold x margin) + conditional thresholds; selection on half of the entities, reported on the other half
E: score calibration tables (top1 / top2 / margin / counts above thresholds)
All written to <work>/eval_<tag>.json and printed."""
import os,sys,json,re,glob,time,unicodedata,importlib,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,'..','..','..')); os.chdir(ROOT); sys.path.insert(0,'pilot')
from metric import f05_per_query
W=os.path.join(HERE,sys.argv[1]); TAG=sys.argv[2] if len(sys.argv)>2 else 'main'; T0=time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
meta=json.load(open(os.path.join(W,'models','meta.json'))); THR=meta['threshold']; DEL=meta['delta']
RD=os.path.join(HERE,'replay_data/')
s1=pd.read_csv(RD+'test_source1.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3)
gt=pd.read_parquet(os.path.join(HERE,'replay_gt.parquet')); own=pd.read_parquet(os.path.join(HERE,'all_owner.parquet')); owner=pd.Series(own.owner.values,index=own.p_id.values)
C=pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(W,'scored','*.parquet'))],ignore_index=True)
C=C.sort_values('s',ascending=False,kind='stable').drop_duplicates(['q_id','p_id']).reset_index(drop=True)
Q=s1.set_index('entity_id'); qids=s1.entity_id.values; qmap=dict(zip(qids,range(len(qids)))); C=C[C.q_id.isin(qmap)].reset_index(drop=True)
C['qi']=C.q_id.map(qmap).values; C['country']=Q.country.reindex(C.q_id).values
tk=set(zip(gt.q_id,gt.p_id)); C['label']=np.array([(a,b) in tk for a,b in zip(C.q_id,C.p_id)],np.int8)
C['p_owner']=owner.reindex(C.p_id).fillna('').values; C['p_foreign']=(C.p_owner!='')&~C.p_owner.isin(qmap)
ntrue_all=gt.groupby('q_id').size().reindex(qids).fillna(0).values.astype(float); qcountry=s1.country.values
in_c=np.bincount(C.qi.values[C.label.values==1],minlength=len(qids)); log('candidates',len(C),'true pairs',int(ntrue_all.sum()),'in candidates',int(in_c.sum()))
# ---- pair metadata for difficulty slices
need=set(C.p_id)
PM=[]
for fn in ('test_source2.tsv','test_source3.tsv'):
    for ch in pd.read_csv(RD+fn,sep='\t',dtype=str,keep_default_na=False,quoting=3,chunksize=1_000_000):
        ch=ch[ch.entity_id.isin(need)]; PM.append(pd.DataFrame({'p_id':ch.entity_id.values,'p_empty':(ch.business_address=='').values,'p_native':ch.business_name.str.contains('[^\x00-ɏ]',regex=True).values}))
PM=pd.concat(PM).set_index('p_id'); C['p_empty']=PM.p_empty.reindex(C.p_id).fillna(False).values.astype(bool); C['p_native']=PM.p_native.reindex(C.p_id).fillna(False).values.astype(bool)
LEG=re.compile(r'\b(inc|llc|ltd|limited|private|pvt|corp|corporation|co|company|llp|lp|plc|the|and|of)\b')
def core(x):
    x=unicodedata.normalize('NFKD',x.lower()); x=''.join(c for c in x if not unicodedata.combining(c)); x=re.sub(r'[^\w\s]',' ',x); return re.sub(r'\s+','',LEG.sub(' ',x))
allS1=pd.read_csv('dataset/train/train_source1.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3,usecols=['business_name','country']); ck=allS1.country+'|'+allS1.business_name.map(core); vc=ck.value_counts()
qcommon=(s1.country+'|'+s1.business_name.map(core)).map(vc).fillna(1).values>=2; C['q_common']=qcommon[C.qi.values]
C['ncomp']=C.groupby('p_id').q_id.transform('size').values-1; C['high_comp']=C.ncomp.values>=2
log('metadata done')
def margin_other(pcode,s):
    o=np.lexsort((-s,pcode)); pg=pcode[o]; ss=s[o]; st=np.r_[True,pg[1:]!=pg[:-1]]; gs=np.flatnonzero(st); gid=np.cumsum(st)-1; gz=np.diff(np.r_[gs,len(ss)])
    t2=np.where(gz>1,ss[np.minimum(gs+1,len(ss)-1)],0.0); t1=ss[gs]; pos=np.arange(len(ss))-gs[gid]; r=np.where(pos==0,t2[gid],t1[gid]); out=np.empty(len(s)); out[o]=r; return out
def evaluate(D,pred,qsel,ntrue,full=False):
    qi=D.qi.values; lab=D.label.values; nq=len(qids)
    tp=np.bincount(qi[pred&(lab==1)],minlength=nq).astype(float); npd=np.bincount(qi[pred],minlength=nq).astype(float); f=f05_per_query(tp,npd,ntrue)
    r=dict(F05=float(f[qsel].mean()),precision=float(tp[qsel].sum()/max(npd[qsel].sum(),1)),recall=float(tp[qsel].sum()/max(ntrue[qsel].sum(),1)),n_pred=int(npd[qsel].sum()),n_true=int(ntrue[qsel].sum()),
           empty_pred_rate=float((npd[qsel]==0).mean()),false_merges=int(npd[qsel].sum()-tp[qsel].sum()))
    rowsel=qsel[qi]; r['rejected_true_in_cands']=int(((lab==1)&~pred&rowsel).sum()); r['blocking_lost']=int((ntrue[qsel]-np.bincount(qi[lab==1],minlength=nq)[qsel]).clip(0).sum())
    if full:
        r['country']={c:float(f[qsel&(qcountry==c)].mean()) for c in np.unique(qcountry)}
        r['country_detail']={c:dict(precision=float(tp[qsel&(qcountry==c)].sum()/max(npd[qsel&(qcountry==c)].sum(),1)),recall=float(tp[qsel&(qcountry==c)].sum()/max(ntrue[qsel&(qcountry==c)].sum(),1)),empty_pred=float((npd[qsel&(qcountry==c)]==0).mean()),matches_per_S1=float(npd[qsel&(qcountry==c)].mean()),true_per_S1=float(ntrue[qsel&(qcountry==c)].mean())) for c in np.unique(qcountry)}
        b=np.select([ntrue==0,ntrue==1,ntrue<=3,ntrue<=6],['0','1','2-3','4-6'],'7+'); r['bucket']={k:float(f[qsel&(b==k)].mean()) for k in ('0','1','2-3','4-6','7+') if (qsel&(b==k)).any()}
        r['singleton_acc']=float((npd[qsel&(ntrue==0)]==0).mean())
        def pp(m):
            m=m&rowsel; t=int((pred&(lab==1)&m).sum()); fp=int((pred&(lab==0)&m).sum()); fn=int(((lab==1)&~pred&m).sum()); return dict(TP=t,FP=fp,FN_in_cands=fn,precision=t/max(t+fp,1),recall_in_cands=t/max(t+fn,1))
        r['source']={s:pp(D.src.values==s) for s in ('s2','s3')}
        r['difficulty']={'empty_addr':pp(D.p_empty.values),'native_script':pp(D.p_native.values),'normal_addr_latin':pp(~D.p_empty.values&~D.p_native.values),'common_name_S1':pp(D.q_common.values),'high_competition(>=2 other S1)':pp(D.high_comp.values),'low_competition':pp(~D.high_comp.values)}
        fpm=pred&(lab==0)&rowsel; r['fp_taxonomy']={'unowned':int((fpm&(D.p_owner.values=='')).sum()),'owned_by_other_replay_S1':int((fpm&(D.p_owner.values!='')&~D.p_foreign.values).sum()),'owned_by_S1_outside_replay(foreign)':int((fpm&D.p_foreign.values).sum())}
    return r
R={'meta':dict(work=W,threshold=THR,delta=DEL,n_S1=len(qids),by_country=s1.country.value_counts().to_dict(),candidates=len(C),true_pairs=int(ntrue_all.sum()),candidate_recall=float(in_c.sum()/ntrue_all.sum()))}
rng=np.random.default_rng(0); u=rng.random(len(qids)); half=rng.random(len(qids))<0.5
# ---------- B: density ablation (withhold S1)
R['A_B']={}
for w in (0.0,0.2,0.33,0.5):
    keepq=u>=w; D=C[keepq[C.qi.values]].reset_index(drop=True); ntrue=np.where(keepq,ntrue_all,0.0)
    D['smo']=margin_other(pd.factorize(D.p_id)[0],D.s.values.astype(float)); pred=(D.s.values>=THR)&(D.s.values-D.smo.values>=DEL)
    r=evaluate(D,pred,keepq,ntrue,full=True); r['density_factor']=1/(1-w); R['A_B'][f'w={w}']=r
    log(f'w={w} density x{1/(1-w):.2f}: F0.5 {r["F05"]:.4f} P {r["precision"]:.4f} R {r["recall"]:.4f} country {r["country"]} FM {r["false_merges"]} rej {r["rejected_true_in_cands"]} fp_tax {r["fp_taxonomy"]}')
# ---------- D: decision grid at test-like density (w=0.2) and robustness at w=0 / w=0.5
TH=[.5,.55,.6,.65,.7,.75,.8,.85,.9]; MG=[0,.01,.025,.05,.075,.1,.15]; R['D']={}
for w in (0.0,0.2,0.5):
    keepq=u>=w; D=C[keepq[C.qi.values]].reset_index(drop=True); ntrue=np.where(keepq,ntrue_all,0.0); D['smo']=margin_other(pd.factorize(D.p_id)[0],D.s.values.astype(float)); s=D.s.values; mg=s-D.smo.values
    qi=D.qi.values; lab=D.label.values; nq=len(qids)
    def F(pred,sel):
        tp=np.bincount(qi[pred&(lab==1)],minlength=nq).astype(float); npd=np.bincount(qi[pred],minlength=nq).astype(float); f=f05_per_query(tp,npd,ntrue); return f[sel].mean()
    grid={}
    for t in TH:
        for m in MG:
            pred=(s>=t)&(mg>=m); grid[f'{t}|{m}']=dict(all=F(pred,keepq),A=F(pred,keepq&half),B=F(pred,keepq&~half),US=F(pred,keepq&(qcountry=='US')),India=F(pred,keepq&(qcountry=='India')))
    R['D'][f'w={w}']=grid; b=max(grid,key=lambda k:grid[k]['all']); log(f'grid w={w}: production {grid[f"{THR}|{DEL}"]["all"]:.4f} | best {b} {grid[b]["all"]:.4f} US {grid[b]["US"]:.4f} India {grid[b]["India"]:.4f}')
    if w==0.2:
        # conditional rules on top of the best global (chosen on half A, reported on half B)
        bA=max(grid,key=lambda k:grid[k]['A']); t0,m0=map(float,bA.split('|')); cond={}
        for name,mask in {'empty_addr':D.p_empty.values,'native':D.p_native.values,'high_comp':D.high_comp.values,'common_name':D.q_common.values,'India':(D.country.values=='India'),'US':(D.country.values=='US')}.items():
            best=(F((s>=t0)&(mg>=m0),keepq&half),0.0)
            for dt in (-.1,-.05,.05,.1,.15,.2):
                pred=(s>=np.where(mask,t0+dt,t0))&(mg>=m0); fa=F(pred,keepq&half)
                if fa>best[0]+1e-9: best=(fa,dt)
            dt=best[1]; pred=(s>=np.where(mask,t0+dt,t0))&(mg>=m0); cond[name]=dict(delta_thr=dt,A=best[0],B=F(pred,keepq&~half),B_base=F((s>=t0)&(mg>=m0),keepq&~half),US=F(pred,keepq&(qcountry=='US')),India=F(pred,keepq&(qcountry=='India')))
        R['D_conditional']=dict(base=bA,rules=cond); log('conditional (selected on half A, B = held-out half):',{k:(v['delta_thr'],round(v['B']-v['B_base'],4)) for k,v in cond.items()})
# ---------- E: calibration tables
def calib(df,sc,qcol,cty):
    g=df.groupby(qcol)[sc]; top1=g.max(); top2=g.apply(lambda x:x.nlargest(2).iloc[-1] if len(x)>1 else 0.0) if len(df)<3_000_000 else None
    r={'top1_mean':float(top1.mean()),'top1_q':top1.quantile([.1,.25,.5,.75,.9]).round(3).tolist(),'frac_pairs_in_.2-.97':float(df[sc].between(.2,.97).mean())}
    for t in (.5,.6,.65,.7,.75,.8,.9): r[f'per_S1_>={t}']=float((df[sc]>=t).groupby(df[qcol]).sum().mean())
    if top2 is not None: r['top2_mean']=float(top2.mean()); r['top1-top2_median']=float((top1-top2).median())
    return r
keepq=u>=0.2; D=C[keepq[C.qi.values]]; R['E']={'replay_w0.2':{c:calib(D[D.country==c],'s','q_id',c) for c in ('US','India')}}
O=pd.read_parquet('pilot/exp2/oof_scores_H10.parquet',columns=['q_id','country','s_B0']); R['E']['benchmark_oof']={c:calib(O[O.country==c],'s_B0','q_id',c) for c in ('US','India')}
json.dump(R,open(os.path.join(W,f'eval_{TAG}.json'),'w'),indent=1,default=float); log('saved'); print(json.dumps(R['E'],indent=1))
