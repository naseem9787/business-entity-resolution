"""Train v2 stage-1 variants and evaluate EVERYTHING on the production-faithful replay with one protocol:
test-like density (withhold 20% S1), owner-presence correction (per-country block rate measured in 00_production_replay/owner_presence.json),
decision layer (T_BASE,T_EMPTY,T_NATIVE; margin .05) chosen on entity half A, reported on half B.
A = production model + production rule; B = production model + calibrated rule; C* = v2 models on v2 (norm3) replay candidates."""
import os,sys,json,glob,time,numpy as np,pandas as pd,lightgbm as lgb
sys.path.insert(0,'pilot'); import v2feats
from metric import f05_per_query
T0=time.time(); RP='pilot/final_optimization/00_production_replay/'; OUT='pilot/final_optimization/02_ablation/'
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
LGB=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=8)
s1=pd.read_csv(RP+'replay_data/test_source1.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3,usecols=['entity_id','country','business_address'])
qids=s1.entity_id.values; qmap=dict(zip(qids,range(len(qids)))); qc=s1.country.values; nq=len(qids)
sys.path.insert(0,'pilot'); import importlib; rb=importlib.import_module('10_regional_bench'); ms=importlib.import_module('make_submission')
rm=json.load(open('submission_work/models/meta.json'))['route_map']; qstate=np.array([ms.routing_key(a,c,rm,rb.parse_state,['US','India']) for a,c in zip(s1.business_address,qc)])
gt=pd.read_parquet(RP+'replay_gt.parquet'); gts=set(zip(gt.q_id,gt.p_id)); ntrue_all=gt.groupby('q_id').size().reindex(qids).fillna(0).values.astype(float)
own=pd.read_parquet(RP+'all_owner.parquet'); owner=dict(zip(own.p_id,own.owner)); Qset=set(qids); OP=json.load(open(RP+'owner_presence.json'))['fp_rates']
u=np.random.default_rng(0).random(nq); half=np.random.default_rng(1).random(nq)<0.5; keepq=u>=0.2; ntrue=np.where(keepq,ntrue_all,0.0)
def prep(D):
    D=D[D.q_id.isin(qmap)].sort_values('s',ascending=False,kind='stable').drop_duplicates(['q_id','p_id']).reset_index(drop=True)
    D['qi']=D.q_id.map(qmap).values; D['lab']=np.array([(a,b) in gts for a,b in zip(D.q_id,D.p_id)]); o=D.p_id.map(owner).fillna('')
    absent=(o!='')&~o.isin(Qset)&~D.lab.values; pb=np.where(qc[D.qi.values]=='US',1-OP['US|owner_present']/OP['US|owner_absent'],1-OP['India|owner_present']/OP['India|owner_absent'])
    blocked=absent.values&(np.random.default_rng(11).random(len(D))<pb); D=D[~blocked&keepq[D.qi.values]].reset_index(drop=True)
    D['smo']=ms.margin_other(pd.factorize(D.p_id)[0],D.s.values.astype(np.float64)); return D
def fq(D,pred):
    qi=D.qi.values; lab=D.lab.values; tp=np.bincount(qi[pred&lab],minlength=nq).astype(float); npd=np.bincount(qi[pred],minlength=nq).astype(float); return f05_per_query(tp,npd,ntrue),tp,npd
def rule(D,tb,te,tn,m=0.05):
    thr=np.select([D.p_empty.values==1,D.p_native.values==1],[te,tn],tb); return (D.s.values>=thr)&(D.s.values-D.smo.values>=m)
def evaluate(D,name,fixed=None):
    D=prep(D); best=None
    grid=[(tb,min(tb+de,.99),max(tb+dn,.3)) for tb in (.5,.6,.65,.7,.75,.8,.85,.9,.93) for de in (0,.05,.1,.15) for dn in (-.2,-.15,-.1,-.05,0)] if fixed is None else [fixed]
    for g in grid:
        f,_,_=fq(D,rule(D,*g)); a=f[keepq&half].mean()
        if best is None or a>best[0]: best=(a,g)
    g=best[1]; pred=rule(D,*g); f,tp,npd=fq(D,pred); sel=keepq&~half; lab=D.lab.values; rows=sel[D.qi.values]
    def sl(m): m=m&rows; t=int((pred&lab&m).sum()); fp=int((pred&~lab&m).sum()); fn=int((lab&~pred&m).sum()); return dict(P=round(t/max(t+fp,1),4),R_in_cands=round(t/max(t+fn,1),4),FP=fp,FN=fn)
    r=dict(model=name,rule=dict(T_BASE=g[0],T_EMPTY=g[1],T_NATIVE=g[2],MARGIN=.05),F05_B=float(f[sel].mean()),US_B=float(f[sel&(qc=='US')].mean()),India_B=float(f[sel&(qc=='India')].mean()),
           F05_all=float(f[keepq].mean()),precision=float(tp[sel].sum()/max(npd[sel].sum(),1)),recall=float(tp[sel].sum()/max(ntrue[sel].sum(),1)),singleton_acc=float((npd[sel&(ntrue==0)]==0).mean()),
           linkage_rate=float((npd[sel]>0).mean()),false_merges=int(npd[sel].sum()-tp[sel].sum()),rejected_true=int((lab&~pred&rows).sum()),cand_recall=float(np.bincount(D.qi.values[lab],minlength=nq)[sel].sum()/ntrue[sel].sum()),
           empty_addr=sl(D.p_empty.values==1),native=sl(D.p_native.values==1))
    log(f'{name:38s} B {r["F05_B"]:.4f} US {r["US_B"]:.4f} IN {r["India_B"]:.4f} P {r["precision"]:.4f} R {r["recall"]:.4f} single {r["singleton_acc"]:.3f} cand {r["cand_recall"]:.4f} FM {r["false_merges"]} rej {r["rejected_true"]} rule {g}'); return r
RES=[]
# ---------------- A / B : production model on production replay candidates
Pm=pd.concat([pd.read_parquet(f) for f in glob.glob(RP+'work_prod/scored/*.parquet')],ignore_index=True)
FL=pd.concat([pd.read_parquet(f,columns=['p_id','p_empty','p_native']) for f in glob.glob(RP+'work_v2/feats/*.parquet')]).drop_duplicates('p_id').set_index('p_id')
need=set(Pm.p_id)-set(FL.index); extra=[]
if need:
    for fn in ('test_source2.tsv','test_source3.tsv'):
        for ch in pd.read_csv(RP+'replay_data/'+fn,sep='\t',dtype=str,keep_default_na=False,quoting=3,chunksize=1_000_000,usecols=['entity_id','business_name','business_address']):
            ch=ch[ch.entity_id.isin(need)]; extra.append(pd.DataFrame({'p_id':ch.entity_id.values,'p_empty':(ch.business_address=='').astype(np.int8).values,'p_native':ch.business_name.str.contains('[^\x00-ɏ]',regex=True).astype(np.int8).values}))
    FL=pd.concat([FL,pd.concat(extra).set_index('p_id')])
Pm['p_empty']=FL.p_empty.reindex(Pm.p_id).fillna(0).values; Pm['p_native']=FL.p_native.reindex(Pm.p_id).fillna(0).values
RES.append(evaluate(Pm,'A production model + rule .65/.05',fixed=(.65,.65,.65))); RES.append(evaluate(Pm,'B production model + rule .85/.95/.70',fixed=(.85,.95,.70))); RES.append(evaluate(Pm,'B* production model + re-tuned rule'))
# ---------------- C : v2 models on v2 replay candidates
V=pd.concat([pd.read_parquet(f).assign(shard=os.path.basename(f)[:-8]) for f in glob.glob(RP+'work_v2/feats/*.parquet')],ignore_index=True)
V=V[V.q_id.isin(qmap)].reset_index(drop=True); V['label']=np.array([(a,b) in gts for a,b in zip(V.q_id,V.p_id)]).astype(np.int8); V['state']=qstate[V.q_id.map(qmap).values]
B=pd.read_parquet(OUT+'bench_v2_feats.parquet'); META=['q_id','p_id','label','country','src','fold','p_empty','p_native','shard','state','s']
base=[c for c in B.columns if c not in META and not c.startswith('x_')]; xall=[c for c in B.columns if c.startswith('x_')]
FS={'C0 tokenizer-only (norm3, base 67 feats)':base,'C1 +translit (native evidence)':base+[c for c in xall if v2feats.GROUPS['translit'](c)],
    'C2 +competition':base+[c for c in xall if v2feats.GROUPS['competition'](c)],'C3 full v2 (all 93 feats)':base+xall}
for name,fs in FS.items():
    m=lgb.train(LGB,lgb.Dataset(B[fs].to_numpy(np.float32),B.label.values),400); D=V[['q_id','p_id','p_empty','p_native']].copy(); D['s']=m.predict(V[fs].to_numpy(np.float32)); RES.append(evaluate(D,name+' [bench-trained]'))
    if name.startswith('C3'): m.save_model(OUT+'stage1_C3_bench.txt')
# production-density training: leave-one-replay-STATE-out, replay-only and bench+replay
for tag,use_bench in (('C4 full v2, replay LOSO',False),('C5 full v2, bench+replay LOSO',True)):
    s=np.zeros(len(V))
    for st in np.unique(V.state):
        tr=V.state.values!=st; Xtr=V.loc[tr,base+xall].to_numpy(np.float32); ytr=V.label.values[tr]
        if use_bench: Xtr=np.vstack([Xtr,B[base+xall].to_numpy(np.float32)]); ytr=np.concatenate([ytr,B.label.values])
        m=lgb.train(LGB,lgb.Dataset(Xtr,ytr),400); s[~tr]=m.predict(V.loc[~tr,base+xall].to_numpy(np.float32)); log(tag,'fold',st)
    D=V[['q_id','p_id','p_empty','p_native']].copy(); D['s']=s; RES.append(evaluate(D,tag))
json.dump(RES,open(OUT+'v2_results.json','w'),indent=1,default=float); log('saved')
