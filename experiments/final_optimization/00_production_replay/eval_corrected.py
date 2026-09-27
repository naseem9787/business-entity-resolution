"""Owner-presence-corrected production replay + decision-layer search.
Correction: a candidate row whose record is owned by an S1 OUTSIDE the replay gets a fixed Bernoulli 'owner would block it' flag with
P(block)=1-rate_present/rate_absent (per country, measured in owner_presence.json). Blocked rows can never be predicted. This mimics the test, where every owner is present.
Reports pessimistic (no correction) and corrected numbers; the decision layer is selected on entity half A and reported on half B, at test-like density (w=0.2) and checked at w=0 / w=0.5."""
import os,sys,json,re,time,numpy as np,pandas as pd
R='pilot/final_optimization/00_production_replay/'; sys.path.insert(0,'pilot'); from metric import f05_per_query
T0=time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
C=pd.read_parquet(R+'replay_scored_labelled.parquet'); OP=json.load(open(R+'owner_presence.json'))['fp_rates']
s1=pd.read_csv(R+'replay_data/test_source1.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3,usecols=['entity_id','country']); qids=s1.entity_id.values; qmap=dict(zip(qids,range(len(qids)))); qc=s1.country.values; nq=len(qids)
gt=pd.read_parquet(R+'replay_gt.parquet'); ntrue_all=gt.groupby('q_id').size().reindex(qids).fillna(0).values.astype(float)
C['qi']=C.q_id.map(qmap).values
need=set(C.p_id); PM=[]
for fn in ('test_source2.tsv','test_source3.tsv'):
    for ch in pd.read_csv(R+'replay_data/'+fn,sep='\t',dtype=str,keep_default_na=False,quoting=3,chunksize=1_000_000,usecols=['entity_id','business_name','business_address']):
        ch=ch[ch.entity_id.isin(need)]; PM.append(pd.DataFrame({'p_id':ch.entity_id.values,'p_empty':(ch.business_address=='').values,'p_native':ch.business_name.str.contains('[^\x00-ɏ]',regex=True).values}))
PM=pd.concat(PM).set_index('p_id'); C['p_empty']=PM.p_empty.reindex(C.p_id).fillna(False).values.astype(bool); C['p_native']=PM.p_native.reindex(C.p_id).fillna(False).values.astype(bool); log('meta')
rng=np.random.default_rng(11); pblock=np.zeros(len(C))
for c in ('US','India'):
    m=(C.grp.values=='owner_absent')&(C.cty.values==c); pblock[m]=1-OP[f'{c}|owner_present']/OP[f'{c}|owner_absent']
C['blocked']=rng.random(len(C))<pblock
C['s2nd_q']=C.groupby('q_id').s.transform(lambda x: x.nlargest(2).iloc[-1] if len(x)>1 else 0.0)
def smo_of(D):
    pc=pd.factorize(D.p_id)[0]; s=D.s.values; o=np.lexsort((-s,pc)); pg=pc[o]; ss=s[o]; st=np.r_[True,pg[1:]!=pg[:-1]]; gs=np.flatnonzero(st); gid=np.cumsum(st)-1; gz=np.diff(np.r_[gs,len(ss)])
    t2=np.where(gz>1,ss[np.minimum(gs+1,len(ss)-1)],0); pos=np.arange(len(ss))-gs[gid]; r=np.where(pos==0,t2[gid],ss[gs][gid]); out=np.empty(len(s)); out[o]=r; return out
u=np.random.default_rng(0).random(nq); half=np.random.default_rng(1).random(nq)<0.5
def setup(w,corrected):
    keepq=u>=w; D=C[keepq[C.qi.values]].reset_index(drop=True)
    if corrected: D=D[~D.blocked.values].reset_index(drop=True)     # the true owner (present in the test) takes the record
    D['smo']=smo_of(D); D['ncomp']=D.groupby('p_id').q_id.transform('size').values-1; return keepq,D,np.where(keepq,ntrue_all,0.0)
def fq(D,pred,ntrue):
    qi=D.qi.values; lab=D.lab.values; tp=np.bincount(qi[pred&lab],minlength=nq).astype(float); npd=np.bincount(qi[pred],minlength=nq).astype(float); return f05_per_query(tp,npd,ntrue),tp,npd
def report(D,pred,ntrue,sel):
    f,tp,npd=fq(D,pred,ntrue); lab=D.lab.values; rows=sel[D.qi.values]
    def sl(m): m=m&rows; t=int((pred&lab&m).sum()); fp=int((pred&~lab&m).sum()); fn=int((lab&~pred&m).sum()); return dict(precision=t/max(t+fp,1),recall_in_cands=t/max(t+fn,1),FP=fp,FN=fn)
    return dict(F05=float(f[sel].mean()),US=float(f[sel&(qc=='US')].mean()),India=float(f[sel&(qc=='India')].mean()),precision=float(tp[sel].sum()/max(npd[sel].sum(),1)),recall=float(tp[sel].sum()/max(ntrue[sel].sum(),1)),
                singleton_acc=float((npd[sel&(ntrue==0)]==0).mean()),linked_rate=float((npd[sel]>0).mean()),matches_per_S1=float(npd[sel].mean()),false_merges=int(npd[sel].sum()-tp[sel].sum()),
                empty_addr=sl(D.p_empty.values),native=sl(D.p_native.values),high_comp=sl(D.ncomp.values>=2),
                F05_by_bucket={b:float(f[sel&m].mean()) for b,m in {'0':ntrue==0,'1':ntrue==1,'2-3':(ntrue>=2)&(ntrue<=3),'4-6':(ntrue>=4)&(ntrue<=6),'7+':ntrue>=7}.items() if (sel&m).any()})
OUT={}
TH=[.5,.55,.6,.65,.7,.75,.8,.85,.9,.925,.95,.97,.98]; MG=[0,.01,.025,.05,.075,.1,.15,.2,.3]
for corrected in (False,True):
    tag='corrected' if corrected else 'pessimistic'; OUT[tag]={}
    for w in (0.0,0.2,0.5):
        keepq,D,ntrue=setup(w,corrected); s=D.s.values; mg=s-D.smo.values; res={}
        for t in TH:
            for m in MG:
                f,_,_=fq(D,(s>=t)&(mg>=m),ntrue); res[(t,m)]=(f[keepq].mean(),f[keepq&half].mean(),f[keepq&~half].mean(),f[keepq&(qc=='US')].mean(),f[keepq&(qc=='India')].mean())
        prod=res[(.65,.05)]; bA=max(res,key=lambda k:res[k][1]); b=res[bA]
        OUT[tag][f'w={w}']=dict(production=dict(zip(['all','A','B','US','India'],prod)),best_on_A=list(bA),best=dict(zip(['all','A','B','US','India'],b)),grid={f'{k[0]}|{k[1]}':v[0] for k,v in res.items()},
                                 production_detail=report(D,(s>=.65)&(mg>=.05),ntrue,keepq))
        log(f'{tag} w={w}: production {prod[0]:.4f} (US {prod[3]:.4f} India {prod[4]:.4f}) | best-on-A {bA} -> held-out B {b[2]:.4f} vs prod B {prod[2]:.4f} | all {b[0]:.4f} US {b[3]:.4f} India {b[4]:.4f}')
# ---- conditional rules at test-like density, corrected; base = best global on half A
keepq,D,ntrue=setup(0.2,True); s=D.s.values; mg=s-D.smo.values; t0,m0=OUT['corrected']['w=0.2']['best_on_A']
masks={'empty_addr':D.p_empty.values,'native':D.p_native.values,'high_comp':D.ncomp.values>=2,'India':D.cty.values=='India','US':D.cty.values=='US','top2_close':(D.s2nd_q.values>=0.9)}
base=(s>=t0)&(mg>=m0); fb,_,_=fq(D,base,ntrue); cond={}
for name,mk in masks.items():
    best=(fb[keepq&half].mean(),0.0,0.0)
    for dt in (-.15,-.1,-.05,.02,.04,.06,.08,.1):
        if t0+dt>=0.995: continue
        for dm in (0,.05,.1):
            f,_,_=fq(D,(s>=np.where(mk,t0+dt,t0))&(mg>=np.where(mk,m0+dm,m0)),ntrue); a=f[keepq&half].mean()
            if a>best[0]+2e-4: best=(a,dt,dm)
    _,dt,dm=best; pred=(s>=np.where(mk,t0+dt,t0))&(mg>=np.where(mk,m0+dm,m0)); f,_,_=fq(D,pred,ntrue)
    cond[name]=dict(d_thr=dt,d_margin=dm,gain_B=float(f[keepq&~half].mean()-fb[keepq&~half].mean()),gain_US=float(f[keepq&(qc=='US')].mean()-fb[keepq&(qc=='US')].mean()),gain_India=float(f[keepq&(qc=='India')].mean()-fb[keepq&(qc=='India')].mean()))
OUT['conditional']=dict(base=[t0,m0],rules=cond); log('conditional (held-out half B gains):',{k:(v['d_thr'],v['d_margin'],round(v['gain_B'],4),round(v['gain_US'],4),round(v['gain_India'],4)) for k,v in cond.items()})
# combined: keep only rules that gain on B AND in both countries
keep=[k for k,v in cond.items() if v['gain_B']>0.0005 and v['gain_US']>=-0.0002 and v['gain_India']>=-0.0002 and k not in('India','US')]
thr=np.full(len(D),t0); mar=np.full(len(D),m0)
for k in keep: thr=np.where(masks[k],np.maximum(thr,t0+cond[k]['d_thr']) if cond[k]['d_thr']>0 else np.minimum(thr,t0+cond[k]['d_thr']),thr); mar=np.where(masks[k],np.maximum(mar,m0+cond[k]['d_margin']),mar)
combo=(s>=thr)&(mg>=mar)
for name,pred in (('production_.65_.05',(s>=.65)&(mg>=.05)),('best_global',base),('best_global+conditional',combo)):
    OUT.setdefault('final_candidates',{})[name]=report(D,pred,ntrue,keepq); r=OUT['final_candidates'][name]
    log(f'{name:25s} F {r["F05"]:.4f} US {r["US"]:.4f} India {r["India"]:.4f} P {r["precision"]:.4f} R {r["recall"]:.4f} singleton {r["singleton_acc"]:.4f} linked {r["linked_rate"]:.3f} m/S1 {r["matches_per_S1"]:.2f} FM {r["false_merges"]} empty P/R {r["empty_addr"]["precision"]:.3f}/{r["empty_addr"]["recall_in_cands"]:.3f} native P/R {r["native"]["precision"]:.3f}/{r["native"]["recall_in_cands"]:.3f}')
OUT['final_rule']=dict(threshold=t0,margin=m0,conditional={k:cond[k] for k in keep})
# robustness of the final rule at other densities
for w in (0.0,0.5):
    kq,Dw,nt=setup(w,True); sw=Dw.s.values; mw=sw-Dw.smo.values; th=np.full(len(Dw),t0); ma=np.full(len(Dw),m0)
    mk={'empty_addr':Dw.p_empty.values,'native':Dw.p_native.values,'high_comp':Dw.ncomp.values>=2,'top2_close':Dw.s2nd_q.values>=0.9}
    for k in keep: th=np.where(mk[k],t0+cond[k]['d_thr'],th); ma=np.where(mk[k],m0+cond[k]['d_margin'],ma)
    for name,pred in (('production',(sw>=.65)&(mw>=.05)),('final',(sw>=th)&(mw>=ma))):
        f,_,_=fq(Dw,pred,nt); OUT.setdefault('robust',{})[f'w={w}|{name}']=dict(F05=float(f[kq].mean()),US=float(f[kq&(qc=='US')].mean()),India=float(f[kq&(qc=='India')].mean()))
log('robustness',OUT['robust'])
json.dump(OUT,open(R+'decision_layer.json','w'),indent=1,default=float); log('saved')
