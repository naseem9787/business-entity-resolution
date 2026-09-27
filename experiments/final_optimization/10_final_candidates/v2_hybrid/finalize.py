"""Apply the v2 hybrid rule to the re-scored test candidates -> <D>/output/matching_results.tsv (+ candidate_pairs identical to baseline) + validator."""
import os,sys,glob,json,shutil,subprocess,numpy as np,pandas as pd
sys.path.insert(0,'pilot'); import make_submission as ms
D='pilot/final_optimization/10_final_candidates/v2_hybrid/'; rule=json.load(open(D+'meta.json'))['rule']
if len(sys.argv)>1: tb=float(sys.argv[1]); rule=dict(T_BASE=tb,T_EMPTY=tb,T_NATIVE=max(tb-.2,.3),MARGIN=.05)
O=D+(sys.argv[2] if len(sys.argv)>2 else 'output')+'/'; os.makedirs(O,exist_ok=True)
files=sorted(glob.glob(D+'test_scored/*.parquet')); assert len(files)==72,len(files)
R=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True).sort_values('s',ascending=False,kind='stable').drop_duplicates(['q_id','p_id']).reset_index(drop=True)
smo=ms.margin_other(pd.factorize(R.p_id)[0],R.s.values.astype(np.float64)); thr=np.select([R.p_empty.values==1,R.p_native.values==1],[rule['T_EMPTY'],rule['T_NATIVE']],rule['T_BASE']); k=(R.s.values>=thr)&(R.s.values-smo>=rule['MARGIN'])
ids1=pd.read_csv('dataset/test/test_source1.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3,usecols=['entity_id','country']); g=R[k].sort_values('q_id',kind='stable').groupby('q_id',sort=False).p_id.agg(','.join).to_dict()
with open(O+'matching_results.tsv','w',encoding='utf8',newline='') as f:
    f.write('source1_entity_id\tmatched_entity_ids\n')
    for q in ids1.entity_id: f.write(q+'\t'+g.get(q,'')+'\n')
shutil.copyfile('output/candidate_pairs.tsv',O+'candidate_pairs.tsv')
n=np.array([len(g[q].split(',')) if q in g else 0 for q in ids1.entity_id]); cty=ids1.country.values
S=dict(rule=rule,candidate_rows=len(R),predicted_matches=int(n.sum()),empty_rate=float((n==0).mean()),matches_per_S1=float(n.mean()),by_country={c:dict(matches=int(n[cty==c].sum()),empty_rate=float((n[cty==c]==0).mean())) for c in np.unique(cty)},
       baseline_submitted=dict(matches=5642165,empty_rate=0.0596),subA_submitted=dict(matches=5313551,empty_rate=0.0689))
print(json.dumps(S,indent=1)); json.dump(S,open(O+'summary.json','w'),indent=1)
v=subprocess.run([sys.executable,'utils/validate_submission.py','--matching',O+'matching_results.tsv','--candidate',O+'candidate_pairs.tsv','--test-dir','dataset/test','--check-ids'],capture_output=True,text=True); print(v.stdout[-500:])
