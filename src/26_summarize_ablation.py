"""Exp2: aggregate ablation_raw.json into ABSOLUTE recall tables (denominator = ALL true pairs of the benchmark, not just those present in the union)."""
import json,numpy as np,pandas as pd
OUT='pilot/exp2/'; BENCH='pilot/exp1/'
RAW=json.load(open(OUT+'ablation_raw.json')); truth=pd.read_parquet(BENCH+'truth_pairs.parquet'); nt=pd.read_parquet(BENCH+'ntrue.parquet').ntrue
S1=pd.read_parquet(BENCH+'bench_s1.parquet',columns=['id','country'])
bk=lambda n: 0 if n==1 else 1 if n==2 else 2 if n<=5 else 3
truth['bucket']=nt.reindex(truth.q_id).values.astype(int); truth['bucket']=[bk(n) for n in truth.bucket]; truth['country']=S1.set_index('id').country.reindex(truth.q_id).values
emp={}
for s in ('s2','s3'): p=pd.read_parquet(BENCH+f'bench_{s}.parquet',columns=['id','addr_raw']); emp.update(zip(p.id,p.addr_raw==''))
truth['empt']=truth.p_id.map(emp).astype(bool)
TOT={}
for (c,s),g in truth.groupby(['country','src']): TOT[f'{c}/{s}']=dict(true=len(g),bucket=[int((g.bucket==i).sum()) for i in range(4)],empty=int(g['empt'].sum()))
CFG=[k for k in RAW['India/s2'] if k not in ('channel_unique_recall',)]; KS=['5','10','15','20','30','50','75','100']; COMB=[k for k in RAW if '/' in k]
def agg(cfg,K,sel):
    f=sum(RAW[c][cfg]['K'][K]['found'] for c in sel); t=sum(TOT[c]['true'] for c in sel); return f/t
def agg_b(cfg,K,sel,i): return sum(RAW[c][cfg]['K'][K]['found_bucket'][i] for c in sel)/sum(TOT[c]['bucket'][i] for c in sel)
def agg_e(cfg,K,sel): return sum(RAW[c][cfg]['K'][K]['found_empty'] for c in sel)/sum(TOT[c]['empty'] for c in sel)
SEL={'ALL':COMB,'India':[c for c in COMB if c.startswith('India')],'US':[c for c in COMB if c.startswith('US')],'S2':[c for c in COMB if c.endswith('s2')],'S3':[c for c in COMB if c.endswith('s3')]}
out={'totals':TOT}
print('ABLATION @K=15 per source (30 candidates/entity): ABSOLUTE recall of all true pairs')
rows=[]
for cfg in CFG:
    r={'config':cfg}; r.update({k:round(agg(cfg,'15',v),4) for k,v in SEL.items()}); r['empty_addr_subset']=round(agg_e(cfg,'15',COMB),4); r['union_rows_per_query']=round(np.mean([RAW[c][cfg]['allowed_per_query'] for c in COMB]),1); rows.append(r)
A=pd.DataFrame(rows); print(A.to_string(index=False)); out['ablation_K15']=rows
print('\nBudget curve, absolute recall (ALL)'); rows=[]
for cfg in CFG:
    r={'config':cfg}; r.update({f'K{k}':round(agg(cfg,k,COMB),4) for k in KS}); rows.append(r)
print(pd.DataFrame(rows).to_string(index=False)); out['budget_all']=rows
for cfg in ('A_baseline4','G_all_channels','H_learned_ranker'):
    print(f'\nCandidate budget table for {cfg}: recall by slice (absolute) | cands/source mean, p95 (mean over the 4 combos)')
    rows=[]
    for k in KS:
        r={'K':int(k),'ALL':agg(cfg,k,COMB),'India':agg(cfg,k,SEL['India']),'US':agg(cfg,k,SEL['US']),'S2':agg(cfg,k,SEL['S2']),'S3':agg(cfg,k,SEL['S3']),**{f'bucket_{n}':agg_b(cfg,k,COMB,i) for i,n in enumerate(['1','2','3-5','6+'])},'empty_addr':agg_e(cfg,k,COMB),
           'cands_per_source_mean':float(np.mean([RAW[c][cfg]['K'][k]['cands_mean'] for c in COMB])),'cands_per_source_p95':float(np.mean([RAW[c][cfg]['K'][k]['cands_p95'] for c in COMB])),'cands_per_source_max':int(max(RAW[c][cfg]['K'][k]['cands_max'] for c in COMB))}
        rows.append({a:(round(b,4) if isinstance(b,float) else b) for a,b in r.items()})
    print(pd.DataFrame(rows).to_string(index=False)); out[f'budget_table_{cfg}']=rows
print('\nCHANNEL UNIQUE RECALL (true pairs found ONLY by a new channel, not by any baseline-4 list; share of all true pairs)')
rows=[]
for c in COMB:
    u=RAW[c]['channel_unique_recall']; t=TOT[c]['true']; rows.append({'combo':c,'baseline4_any':round(u['baseline4_any']/t,4),**{k:round(u[k]/t,4) for k in ('nc3','an','quota','key','reverse')},'any_channel(ceiling)':round(u['any_channel']/t,4)})
print(pd.DataFrame(rows).to_string(index=False)); out['channel_unique']=rows
out['ranker_importance']=RAW['importance']; json.dump(out,open(OUT+'ablation_summary.json','w'),indent=1,default=float)
