import numpy as np, pandas as pd
def f05_per_query(tp,npred,ntrue):
    """competition metric per S1 entity. empty truth: 1 if empty prediction else 0."""
    tp=np.asarray(tp,float); npred=np.asarray(npred,float); ntrue=np.asarray(ntrue,float)
    fp=npred-tp; fn=ntrue-tp; den=1.25*tp+0.25*fn+fp
    f=np.where(den>0,1.25*tp/np.maximum(den,1e-12),0.)
    f=np.where(ntrue==0,(npred==0).astype(float),f)
    return f
def evaluate(qids,pred_mask,label,truth_n,all_qids):
    """qids/pred_mask/label are per-candidate arrays. truth_n: Series q_id->#true matches (all sources). all_qids: every evaluated S1 id (incl. those with no candidates)."""
    d=pd.DataFrame({'q':qids,'pred':pred_mask.astype(int),'tp':(pred_mask&(label==1)).astype(int)}).groupby('q').sum()
    d=d.reindex(all_qids).fillna(0)
    nt=truth_n.reindex(all_qids).fillna(0).values
    f=f05_per_query(d.tp.values,d.pred.values,nt)
    P=np.where(d.pred>0,d.tp/np.maximum(d.pred,1),np.nan); R=np.where(nt>0,d.tp/np.maximum(nt,1),np.nan)
    single=nt==0
    return dict(F05=float(f.mean()),precision_micro=float(d.tp.sum()/max(d.pred.sum(),1)),recall_micro=float(d.tp.sum()/max(nt.sum(),1)),
                precision_macro=float(np.nanmean(P)),recall_macro=float(np.nanmean(R)),singleton_acc=float((d.pred.values[single]==0).mean()) if single.any() else None,
                singleton_share=float(single.mean()),false_merges=int((d.pred-d.tp).sum()),missed=int((nt-d.tp.values).sum()),n_pred=int(d.pred.sum()),n_true=int(nt.sum()),F05_nonsingleton=float(f[~single].mean()),n_queries=len(all_qids)),f

# ================= Exp1 additions (existing functions above are unchanged) =================
def query_counts(qidx,pred,label,nq):
    """per-query tp / #predicted from integer query index arrays (fast, no pandas)"""
    pred=np.asarray(pred,bool); tp=np.bincount(qidx[pred&(label==1)],minlength=nq).astype(float); npred=np.bincount(qidx[pred],minlength=nq).astype(float)
    return tp,npred
def f05_from_pred(qidx,pred,label,ntrue,nq):
    tp,npred=query_counts(qidx,pred,label,nq); return f05_per_query(tp,npred,ntrue)
def paired_bootstrap(fa,fb,n=1000,seed=0):
    """paired bootstrap over entities of mean(fb)-mean(fa); returns (delta, lo95, hi95)"""
    d=np.asarray(fb,float)-np.asarray(fa,float); rng=np.random.default_rng(seed); m=len(d); out=np.empty(n)
    for i in range(0,n,50):
        k=min(50,n-i); idx=rng.integers(0,m,size=(k,m)); out[i:i+k]=d[idx].mean(1)
    return float(d.mean()),float(np.percentile(out,2.5)),float(np.percentile(out,97.5))
def oracle_attribution(qidx,pred,label,ntrue,nq,ntrue_in_cands,fp_masks=None,fn_masks=None):
    """What macro F0.5 would be if an error class were fixed by an oracle. fp_masks/fn_masks: dict name->bool array over candidate rows."""
    pred=np.asarray(pred,bool); label=np.asarray(label); fp=pred&(label==0); fn=(~pred)&(label==1)
    base=f05_from_pred(qidx,pred,label,ntrue,nq).mean(); res={'base':(base,0.0)}
    lost=np.maximum(ntrue-ntrue_in_cands,0)
    def add(name,p2,extra_tp=None):
        tp,npred=query_counts(qidx,p2,label,nq)
        if extra_tp is not None: tp=tp+extra_tp; npred=npred+extra_tp
        f=f05_per_query(tp,npred,ntrue).mean(); res[name]=(float(f),float(f-base))
    add('remove_all_false_merges',pred&~fp)
    for k,m in (fp_masks or {}).items(): add(f'remove_FP[{k}]',pred&~(fp&m))
    add('recover_all_model_rejected_pairs',pred|fn)
    for k,m in (fn_masks or {}).items(): add(f'recover_FN[{k}]',pred|(fn&m))
    add('recover_all_blocking_lost',pred,extra_tp=lost)
    add('recover_model_rejected+blocking_lost',pred|fn,extra_tp=lost)
    return res
