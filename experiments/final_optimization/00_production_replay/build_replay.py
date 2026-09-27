"""Production replay data: training S1 of HELD-OUT states (never used by the benchmark that trained the production models) formatted as a test set,
against the FULL training S2/S3 pools (so production routing + India broadcast pool are reproduced exactly by make_submission.predict).
Writes replay_data/test_source{1,2,3}.tsv, replay_gt.parquet, replay_meta.json."""
import os,sys,json,shutil,importlib,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,'..','..','..')); os.chdir(ROOT); sys.path.insert(0,'pilot')
rb=importlib.import_module('10_regional_bench'); ms=importlib.import_module('make_submission')
meta=json.load(open('submission_work/models/meta.json')); bench=json.load(open('pilot/exp1/folds.json'))['region']
D='dataset/train/'; RD=HERE+'/replay_data/'; os.makedirs(RD,exist_ok=True)
s1=pd.read_csv(D+'train_source1.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3)
s1['key']=[ms.routing_key(a,c,meta['route_map'],rb.parse_state,ms.CFG['PARSER_COUNTRIES']) for a,c in zip(s1.business_address,s1.country)]
cnt=s1.groupby(['country','key']).size()
excl={c:{ms.re.sub(r'[^0-9a-zA-Z]+','_',s) for s in v} for c,v in bench.items()}
TARGET={'US':40000,'India':45000}; rng=np.random.default_rng(7); chosen={}
for c,tgt in TARGET.items():
    el=[k for k,n in cnt[c].items() if k not in excl[c] and not k.startswith('_') and 8000<=n<=26000]; order=list(rng.permutation(el)); ch=[]; tot=0
    while order and tot<tgt: k=order.pop(0); ch.append(k); tot+=int(cnt[c][k])
    chosen[c]=ch; print(c,'replay states',ch,'S1',tot,flush=True)
sel=s1[[k in chosen[c] for c,k in zip(s1.country,s1.key)]]
sel[['entity_id','business_name','business_address','country']].to_csv(RD+'test_source1.tsv',sep='\t',index=False,quoting=3)
for i in (2,3): shutil.copyfile(D+f'train_source{i}.tsv',RD+f'test_source{i}.tsv')
gt=pd.read_csv(D+'train_ground_truth.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3); gt=gt[gt.source1_entity_id.isin(set(sel.entity_id))]
g=gt.assign(p_id=gt.matched_entity_ids.str.split(',')).explode('p_id'); g=g[g.p_id!=''][['source1_entity_id','p_id']].rename(columns={'source1_entity_id':'q_id'})
g.to_parquet(HERE+'/replay_gt.parquet')
# ALL owned ids (any S1) -> lets the evaluation tell 'unowned' from 'owned by an S1 outside the replay' (foreign-owned) distractors
allo=gt_all=pd.read_csv(D+'train_ground_truth.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3)
a=allo.assign(p_id=allo.matched_entity_ids.str.split(',')).explode('p_id'); a=a[a.p_id!=''][['source1_entity_id','p_id']].rename(columns={'source1_entity_id':'owner'}); a.to_parquet(HERE+'/all_owner.parquet')
json.dump(dict(states=chosen,n_S1=len(sel),n_S1_by_country=sel.country.value_counts().to_dict(),true_pairs=len(g),excluded_benchmark_states={c:sorted(v) for c,v in excl.items()}),open(HERE+'/replay_meta.json','w'),indent=1)
print('S1',len(sel),sel.country.value_counts().to_dict(),'true pairs',len(g))
