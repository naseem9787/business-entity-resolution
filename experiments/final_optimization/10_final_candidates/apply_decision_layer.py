"""Re-apply a NEW decision layer to the CACHED production test scores (submission_work/scored, read-only). No re-inference, no retraining.
Rule (selected on production-faithful labelled replay, see 00_production_replay/PRODUCTION_REPLAY_REPORT.md):
  keep (q,p) iff  s >= thr(p)  and  s - max_{q'!=q} s(q',p) >= MARGIN
  thr(p) = T_EMPTY if S2/S3 address empty, T_NATIVE if name has non-Latin script (and address non-empty), else T_BASE
Candidates are unchanged -> candidate_pairs.tsv is byte-identical to the baseline's.
usage: python apply_decision_layer.py <out_dir> [T_BASE MARGIN T_EMPTY T_NATIVE]"""
import os,sys,glob,json,shutil,subprocess,numpy as np,pandas as pd
out=sys.argv[1]; T_BASE,MARGIN,T_EMPTY,T_NATIVE=(map(float,sys.argv[2:6]) if len(sys.argv)>=6 else (0.85,0.30,0.95,0.70)); os.makedirs(out,exist_ok=True)
TD='dataset/test/'; flags={}
for i in (2,3):
    for ch in pd.read_csv(TD+f'test_source{i}.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3,chunksize=1_000_000,usecols=['entity_id','business_name','business_address']):
        e=(ch.business_address=='').values; n=ch.business_name.str.contains('[^\x00-ɏ]',regex=True).values
        flags.update(zip(ch.entity_id.values,np.where(e,1,np.where(n,2,0)).astype(np.int8)))
print('pool flags',len(flags),flush=True)
keep=[]; stats={}
for c in ('France','India','US'):
    C=pd.concat([pd.read_parquet(f) for f in glob.glob(f'submission_work/scored/{c}__*.parquet')],ignore_index=True).sort_values('s',ascending=False,kind='stable').drop_duplicates(['q_id','p_id']).reset_index(drop=True)
    pc=pd.factorize(C.p_id)[0]; s=C.s.values.astype(np.float64); o=np.lexsort((-s,pc)); pg=pc[o]; ss=s[o]; st=np.r_[True,pg[1:]!=pg[:-1]]; gs=np.flatnonzero(st); gid=np.cumsum(st)-1; gz=np.diff(np.r_[gs,len(ss)])
    t2=np.where(gz>1,ss[np.minimum(gs+1,len(ss)-1)],0.0); pos=np.arange(len(ss))-gs[gid]; r=np.where(pos==0,t2[gid],ss[gs][gid]); smo=np.empty(len(s)); smo[o]=r
    f=C.p_id.map(flags).fillna(0).values; thr=np.select([f==1,f==2],[T_EMPTY,T_NATIVE],T_BASE); k=(s>=thr)&(s-smo>=MARGIN)
    base=(s>=0.65)&(s-smo>=0.05)
    stats[c]=dict(candidates=len(C),new_matches=int(k.sum()),baseline_rule_matches=int(base.sum()),new_linked_S1=int(C.q_id[k].nunique()),S1=int(C.q_id.nunique()),empty_addr_matches=int((k&(f==1)).sum()),native_matches=int((k&(f==2)).sum()))
    keep.append(C.loc[k,['q_id','p_id']]); print(c,stats[c],flush=True); del C
K=pd.concat(keep,ignore_index=True)
ids1=pd.read_csv(TD+'test_source1.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3,usecols=['entity_id']).entity_id.tolist()
g=K.sort_values('q_id',kind='stable').groupby('q_id',sort=False).p_id.agg(','.join).to_dict()
with open(os.path.join(out,'matching_results.tsv'),'w',encoding='utf8',newline='') as fh:
    fh.write('source1_entity_id\tmatched_entity_ids\n')
    for q in ids1: fh.write(q+'\t'+g.get(q,'')+'\n')
shutil.copyfile('output/candidate_pairs.tsv',os.path.join(out,'candidate_pairs.tsv'))
n=np.array([len(g.get(q,'').split(',')) if g.get(q,'') else 0 for q in ids1])
summ=dict(rule=dict(T_BASE=T_BASE,MARGIN=MARGIN,T_EMPTY=T_EMPTY,T_NATIVE=T_NATIVE),S1=len(ids1),predicted_matches=int(n.sum()),empty_rate=float((n==0).mean()),matches_per_S1=float(n.mean()),by_country=stats)
v=subprocess.run([sys.executable,'utils/validate_submission.py','--matching',os.path.join(out,'matching_results.tsv'),'--candidate',os.path.join(out,'candidate_pairs.tsv'),'--test-dir',TD,'--check-ids'],capture_output=True,text=True)
summ['validator']=v.stdout[-600:]; summ['validator_pass']=v.returncode==0
json.dump(summ,open(os.path.join(out,'summary.json'),'w'),indent=1); print(json.dumps({k:summ[k] for k in ('rule','S1','predicted_matches','empty_rate','matches_per_S1','validator_pass')},indent=1)); print(v.stdout[-500:])
