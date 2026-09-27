"""Exp2 step 2: ablations A-H (candidate recall / budget) + learned blocking ranker H (cross-fitted over the Exp1 state folds).
Reads pilot/exp2/union_*.parquet (from 20_retrieval.py). Writes pilot/exp2/ranked_{country}_{src}.parquet (scores per union row), ablation_raw.json.
Ranker features never use the label of any evaluated fold; country identity is NOT a feature (open-set safe)."""
import sys,os,json,gc,time,numpy as np,pandas as pd,lightgbm as lgb
sys.path.insert(0,'pilot')
OUT='pilot/exp2/'; BENCH='pilot/exp1/'; T0=time.time(); KS=(5,10,15,20,30,50,75,100)
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
LGB=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=8); RANK_ROUNDS=250
COMBOS=[(c,s) for c in ('India','US') for s in ('s2','s3')]
ntrue=pd.read_parquet(BENCH+'ntrue.parquet').ntrue
def bucket_code(n): return np.select([n==1,n==2,n<=5],[0,1,2],3)   # 1,2,3-5,6+
BN=['1','2','3-5','6+']
def group_starts(qi):
    st=np.r_[True,qi[1:]!=qi[:-1]]; s=np.flatnonzero(st); return s,np.cumsum(st)-1
def seg_max(v,qi):
    s,g=group_starts(qi); return np.maximum.reduceat(v,s)[g]
def load(c,s):
    D=pd.read_parquet(OUT+f'union_{c}_{s}.parquet'); qinfo=pd.read_parquet(OUT+f'qinfo_{c}.parquet')
    D['fused']=D.cos_nw+2*D.cos_aw+D.cos_nc+2*D.cos_ac
    return D,qinfo
def rule_scores(D):
    """hand-weighted per-channel scores for ablations A-G (weights are fixed heuristics, NOT tuned)"""
    f=D.fused.values; qmask=(D.r_quota.values<=3)&(D.p_empty.values==1)&(D.cos_nc3.values>=0.6)
    A=np.minimum.reduce([D.r_nw.values,D.r_aw.values,D.r_nc.values,D.r_ac.values])<999
    kx=(D.key_kx.values>0); kn=(D.key_kn.values>0); rv=D.rev_rank.values
    cfg={'A_baseline4':(A,f),
         'B_+name3gram':(A|(D.r_nc3.values<999),f+D.cos_nc3.values),
         'C_+emptyaddr_quota':(A|qmask,f+100*qmask),
         'D_+addr_key':(A|kx|kn,f+1.5*kx+1.0*kn),
         'E_+numnorm_addr':(A|(D.r_an.values<999),f+2*D.cos_an.values),
         'F_+reverse':(A|(rv<999),f+0.5*(rv<=3)),
         'G_all_channels':(np.ones(len(D),bool),f+D.cos_nc3.values+2*D.cos_an.values+1.5*kx+1.0*kn+0.5*(rv<=3)+100*qmask)}
    return cfg
def ranker_matrix_cols(D):
    """all ranker inputs as float32 column arrays (no labels, no country)"""
    qi=D.qi.values; pj=D.pj.values; f=D.fused.values.astype(np.float32); cols={}
    for c in ('cos_nw','cos_aw','cos_nc','cos_ac','cos_all','cos_nc3','cos_an'): cols[c]=D[c].values
    cols['fused']=f
    for c in ('r_nw','r_aw','r_nc','r_ac','r_nc3','r_an','r_quota','key_kx','key_kn','rev_rank','p_empty','q_nlen','p_nlen','q_ntok','p_ntok','q_atok','p_atok'): cols[c]=D[c].values.astype(np.float32)
    cols['gap_q_fused']=f-seg_max(f,qi)
    o=np.argsort(pj,kind='stable'); pjo=pj[o]; s,g=group_starts(pjo); pm=np.maximum.reduceat(f[o],s)[g]; pf=np.diff(np.r_[s,len(pjo)])[g].astype(np.float32)
    gp=np.empty(len(f),np.float32); gp[o]=f[o]-pm; cols['gap_p_fused']=gp; cf=np.empty(len(f),np.float32); cf[o]=pf; cols['p_freq']=cf
    s,g=group_starts(qi); cols['q_union']=np.diff(np.r_[s,len(qi)])[g].astype(np.float32)
    cols['name_addr_cos_gap']=(D.cos_nc3.values-D.cos_aw.values).astype(np.float32)
    return cols
def rank_within_query(qi,score,allowed):
    idx=np.flatnonzero(allowed); o=np.lexsort((-score[idx],qi[idx])); idx=idx[o]; q=qi[idx]; s,g=group_starts(q); rk=np.arange(len(idx))-s[g]+1; return idx,rk
def evaluate_cfg(D,qbucket,allowed,score):
    """raw counts so they can be aggregated across combos exactly"""
    qi=D.qi.values; lab=D.label.values==1; idx,rk=rank_within_query(qi,score,allowed); lab_s=lab[idx]; nq=len(qbucket); tb=qbucket[qi[lab]]; te=D.p_empty.values[lab]==1
    tot=dict(true=int(lab.sum()),true_bucket=[int((tb==i).sum()) for i in range(4)],true_empty=int(te.sum())); res={}
    cnt_all=np.bincount(qi[allowed],minlength=nq)
    for K in KS:
        sel=idx[(rk<=K)&lab_s]; qb=qbucket[qi[sel]]; nk=np.minimum(cnt_all,K)
        res[K]=dict(found=int(len(sel)),found_bucket=[int((qb==i).sum()) for i in range(4)],found_empty=int((D.p_empty.values[sel]==1).sum()),cands_mean=float(nk.mean()),cands_p95=float(np.percentile(nk,95)),cands_max=int(nk.max()),cands_sum=int(nk.sum()))
    return tot,res
# =========================================================================================== 1. sample for ranker training
log('building ranker training sample')
SAM=[];
for c,s in COMBOS:
    D,qinfo=load(c,s); cols=ranker_matrix_cols(D); _,Gscore=rule_scores(D)['G_all_channels']; allowed=np.ones(len(D),bool)
    idx,rk=rank_within_query(D.qi.values,Gscore,allowed); rank=np.empty(len(D),np.int32); rank[idx]=rk
    rng=np.random.default_rng(1); keep=(D.label.values==1)|(rank<=25)|(rng.random(len(D))<0.02)
    M=np.column_stack([cols[k][keep] for k in cols]).astype(np.float32); SAM.append((M,D.label.values[keep],qinfo.fold.values[D.qi.values[keep]],np.full(keep.sum(),int(s=='s3'),np.float32))); names=list(cols); log(c,s,'sample rows',keep.sum(),'of',len(D)); del D,cols,M; gc.collect()
Xs=np.vstack([np.column_stack([m,src]) for m,_,_,src in SAM]); ys=np.concatenate([y for _,y,_,_ in SAM]); fs=np.concatenate([f for _,_,f,_ in SAM]); names=names+['src3']; del SAM; gc.collect()
log('training sample',Xs.shape,'positives',int(ys.sum()))
models={}
for k in range(5):
    tr=fs!=k; models[k]=lgb.train(LGB,lgb.Dataset(Xs[tr],ys[tr]),RANK_ROUNDS); log('ranker fold',k,'trained on',int(tr.sum()))
imp=pd.Series(models[0].feature_importance('gain'),names).sort_values(ascending=False); log('ranker top features',(imp/imp.sum()).round(3).head(12).to_dict())
del Xs,ys,fs; gc.collect()
# =========================================================================================== 2. score every union row (cross-fitted) + evaluate all configs
RAW={'names':names,'importance':(imp/imp.sum()).round(4).to_dict()}
for c,s in COMBOS:
    D,qinfo=load(c,s); cols=ranker_matrix_cols(D); qi=D.qi.values; fold=qinfo.fold.values[qi]; H=np.zeros(len(D),np.float32); src3=np.full(len(D),int(s=='s3'),np.float32)
    for a in range(0,len(D),1_500_000):
        b=min(a+1_500_000,len(D)); M=np.column_stack([cols[k][a:b] for k in cols]+[src3[a:b]]).astype(np.float32)
        for k in range(5):
            m=fold[a:b]==k
            if m.any(): H[a:b][m]=models[k].predict(M[m])
        del M
    cfg=rule_scores(D); cfg['H_learned_ranker']=(np.ones(len(D),bool),H)
    qb=bucket_code(ntrue.reindex(qinfo.id).fillna(0).values.astype(int)); R={}
    for name,(allowed,score) in cfg.items():
        tot,res=evaluate_cfg(D,qb,allowed,score); R[name]=dict(total=tot,K=res,allowed_per_query=float(np.bincount(D.qi.values[allowed],minlength=len(qb)).mean()))
        log(c,s,f'{name:22s}','recall@15',round(res[15]['found']/tot['true'],4),'@30',round(res[30]['found']/tot['true'],4),'@100',round(res[100]['found']/tot['true'],4),'cands@15',round(res[15]['cands_mean'],2))
    # channel marginal value: true pairs found ONLY thanks to a new channel (not in any baseline-4 channel list)
    lab=D.label.values==1; base=np.minimum.reduce([D.r_nw.values,D.r_aw.values,D.r_nc.values,D.r_ac.values])<999
    R['channel_unique_recall']={'true':int(lab.sum()),'baseline4_any':int((lab&base).sum()),**{n:int((lab&~base&m).sum()) for n,m in {'nc3':D.r_nc3.values<999,'an':D.r_an.values<999,'quota':D.r_quota.values<999,'key':(D.key_kx.values>0)|(D.key_kn.values>0),'reverse':D.rev_rank.values<999}.items()},'any_channel':int(lab.sum())}
    RAW[f'{c}/{s}']=R
    pd.DataFrame({'score_H':H,'score_G':cfg['G_all_channels'][1].astype(np.float32),'fused':D.fused.values.astype(np.float32)}).to_parquet(OUT+f'ranked_{c}_{s}.parquet'); del D,cols,H,cfg; gc.collect()
json.dump(RAW,open(OUT+'ablation_raw.json','w'),indent=1); log('saved ablation_raw.json')
