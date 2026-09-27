"""Exp1 step 0-2: density-faithful regional benchmark.
Region = whole US/India states (ALL S1 in the region are queries; pool = every S2/S3 record that could compete inside the region),
so that every competing S1 owner is present. Output -> pilot/exp1/bench_{s1,s2,s3}.parquet, folds.json, bench_stats.json
Memory-lean: chunked reads, ids compared through 64-bit hashes (no giant python sets)."""
import sys,os,re,json,time,importlib,multiprocessing as mp,numpy as np,pandas as pd
sys.path.insert(0,'pilot')
from norm import US_ST,US2,IN_ST            # (read-only use of the existing tables)
D='dataset/train/'; OUT='pilot/exp1/'; os.makedirs(OUT,exist_ok=True); CH=500_000
T0=time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]',*a,flush=True)
# ---------------- state parsing (US: 2-letter code; India: canonical state name) ----------------
def _nc(c): return ' '.join(re.sub(r'\d+',' ',c.lower().replace('.',' ')).split())
US_CODE={v:v for v in list(US_ST.values())+list(US2.values())+['dc']}
US_NAME={**US_ST,**US2,'district of columbia':'dc','washington dc':'dc'}
_IN_INV={}
for k,v in IN_ST.items(): _IN_INV.setdefault(v,k)
_IN_INV['od']='odisha'; _IN_INV['ts']='telangana'; _IN_INV['tg']='telangana'; _IN_INV['uk']='uttarakhand'
IN_CODE={c:n for c,n in _IN_INV.items()}
IN_NAME={k:k for k in IN_ST}; IN_NAME.update({'orissa':'odisha','odisha':'odisha','nct of delhi':'delhi','chattisgarh':'chhattisgarh','pondicherry':'puducherry','puducherry':'puducherry','uttaranchal':'uttarakhand'})
MAPS={'US':(US_CODE,US_NAME),'India':(IN_CODE,IN_NAME)}
def parse_state(addr,country):
    m=MAPS.get(country)
    if m is None or not addr: return ''
    code,name=m; fc=fn=''
    for c in addr.split(','):
        k=_nc(c)
        if k in code: fc=code[k]
        elif k in name: fn=name[k]
    return fc or fn            # an explicit code component wins over a full-name component (city "Washington")
def parse_chunk(args): return [parse_state(a,c) for a,c in zip(*args)]
def parse_many(pool,addrs,cs):
    n=len(addrs); k=max(1,n//40000+1); step=(n+k-1)//k
    parts=pool.map(parse_chunk,[(list(addrs[i:i+step]),list(cs[i:i+step])) for i in range(0,n,step)])
    return [x for p in parts for x in p]
def rd(f,**kw): return pd.read_csv(D+f,sep='\t',dtype=str,keep_default_na=False,quoting=3,**kw)
def h64(s): return pd.util.hash_pandas_object(s,index=False).to_numpy(np.uint64)
if __name__=='__main__':
    pool=mp.Pool(4); stats={}
    # ---- 1. S1 states -> region selection
    parts=[]
    for ch in rd('train_source1.tsv',chunksize=CH,usecols=['entity_id','business_address','country']):
        ch['state']=parse_many(pool,ch.business_address.values,ch.country.values); parts.append(ch[['entity_id','country','state']])
    s1st=pd.concat(parts,ignore_index=True); del parts
    stats['s1_total']=len(s1st); stats['s1_state_parse_rate_global']=float((s1st.state!='').mean())
    stats['s1_state_parse_rate_by_country']={c:float((g.state!='').mean()) for c,g in s1st.groupby('country')}
    log('S1 states parsed',stats['s1_state_parse_rate_by_country'])
    rng=np.random.default_rng(0); region={}; folds={}
    for c in ('US','India'):
        cnt=s1st[(s1st.country==c)&(s1st.state!='')].state.value_counts().sort_index()
        elig=[s for s in cnt.index if cnt[s]<=25000]; order=list(rng.permutation(elig)); chosen=[]; tot=0
        while order and (tot<50000 or len(chosen)<5):
            s=order.pop(0); chosen.append(s); tot+=int(cnt[s])
        region[c]=chosen
        # greedy balanced assignment of states to 5 folds
        load=np.zeros(5); f={}
        for s in sorted(chosen,key=lambda s:-cnt[s]): k=int(load.argmin()); f[s]=k; load[k]+=cnt[s]
        folds[c]=f; log(c,'region states',len(chosen),'S1',tot,'fold loads',load.astype(int).tolist(),chosen)
    stats['region']={c:region[c] for c in region}
    rset={(c,s) for c in region for s in region[c]}
    inreg=np.array([(c,s) in rset for c,s in zip(s1st.country.values,s1st.state.values)])
    R=set(s1st.entity_id[inreg]); stats['s1_region']=len(R)
    stats['s1_region_by_country']=s1st[inreg].country.value_counts().to_dict()
    stats['s1_region_state_parse_rate']=1.0   # by construction (region = parsed states); global rates above are the validity check
    share={c:stats['s1_region_by_country'][c]/float((s1st.country==c).sum()) for c in region}; stats['region_share_of_country']=share
    log('region S1',len(R),share)
    # ---- 2. ground truth: hashed set of every owned S2/S3 id; owner map for region S1
    hs=[]; own=[]
    for ch in rd('train_ground_truth.tsv',chunksize=CH):
        ch=ch[ch.matched_entity_ids!='']
        if ch.empty: continue
        e=ch.assign(m=ch.matched_entity_ids.str.split(',')).explode('m')[['source1_entity_id','m']]
        hs.append(h64(e.m)); own.append(e[e.source1_entity_id.isin(R)])
    allm=np.sort(np.concatenate(hs)); own=pd.concat(own,ignore_index=True); del hs
    owner=dict(zip(own.m,own.source1_entity_id)); log('gt hashed',len(allm),'region-owned pairs',len(owner))
    # ---- 3. S1 region rows
    s1=pd.concat([ch[ch.entity_id.isin(R)] for ch in rd('train_source1.tsv',chunksize=CH)],ignore_index=True)
    s1=s1.merge(s1st[['entity_id','state']],on='entity_id',how='left'); log('S1 rows',len(s1))
    s1['fold']=[folds[c][s] for c,s in zip(s1.country,s1.state)]
    # ---- 4. pools
    pools={}
    for src,fn in (('s2','train_source2.tsv'),('s3','train_source3.tsv')):
        keep=[]; cnt=dict(total=0,owned_region=0,owned_outside=0,unowned=0,unowned_instate=0,unowned_nostate=0,unowned_nostate_kept=0,unowned_otherstate=0,outside_owned_sample=0,outside_owned_sample_in_region_state=0)
        for ch in rd(fn,chunksize=CH):
            cnt['total']+=len(ch); h=h64(ch.entity_id); has=np.isin(h,allm)
            ow=ch.entity_id.map(owner); own_reg=ow.notna().to_numpy(); ch['owner']=ow.fillna('')
            cnt['owned_region']+=int(own_reg.sum()); cnt['owned_outside']+=int((has&~own_reg).sum()); un=~has
            cnt['unowned']+=int(un.sum())
            # sample of outside-owned records: how many carry a parsed state inside the region (excluded by rule)
            samp=(has&~own_reg)&(h%np.uint64(20)==0)
            if samp.any():
                stt=parse_many(pool,ch.business_address.values[samp],ch.country.values[samp]); cnt['outside_owned_sample']+=int(samp.sum())
                cnt['outside_owned_sample_in_region_state']+=sum((c,s) in rset for c,s in zip(ch.country.values[samp],stt))
            sel=own_reg.copy(); ch['state']=''
            if un.any():
                u=ch[un]; stt=np.array(parse_many(pool,u.business_address.values,u.country.values),dtype=object)
                ins=np.array([(c,s) in rset for c,s in zip(u.country.values,stt)]); ns=(stt=='')
                pc=np.array([share.get(c,0.0) for c in u.country.values]); hh=h[un].astype(np.float64)/2**64
                kept_ns=ns&(hh<pc); cnt['unowned_instate']+=int(ins.sum()); cnt['unowned_nostate']+=int(ns.sum()); cnt['unowned_nostate_kept']+=int(kept_ns.sum()); cnt['unowned_otherstate']+=int((~ins&~ns).sum())
                selu=ins|kept_ns; idx=np.flatnonzero(un); sel[idx[selu]]=True; ch.loc[ch.index[idx],'state']=stt
            keep.append(ch[sel])
        p=pd.concat(keep,ignore_index=True); pools[src]=p; cnt['pool']=len(p); cnt['pool_unowned_share']=float((p.owner=='').mean()); cnt['pool_per_region_s1']=len(p)/len(s1)
        cnt['global_records']=int(cnt['total']); stats[src]=cnt; log(src,cnt)
    stats['global_ratio_pool_per_s1']={'s2':5034616/2206821,'s3':5285603/2206821}; stats['global_unowned_share']={'s2':1340997/5034616,'s3':1340857/5285603}
    ok={}
    ok['s1_parse_rate>=0.97']=all(v>=0.97 for v in stats['s1_state_parse_rate_by_country'].values())
    for s in ('s2','s3'):
        ok[f'{s}_pool_ratio_within_10pct']=abs(stats[s]['pool_per_region_s1']/stats['global_ratio_pool_per_s1'][s]-1)<=0.10
        ok[f'{s}_unowned_share_within_3pts']=abs(stats[s]['pool_unowned_share']-stats['global_unowned_share'][s])<=0.03
    stats['validity_checks']=ok; log('VALIDITY',ok)
    # ---- 5. normalisation (re-uses the existing 01_prep.work) and save
    prep=importlib.import_module('01_prep')
    def prepare(df,extra):
        df=df.reset_index(drop=True); ch=[df.iloc[i:i+50000] for i in range(0,len(df),50000)]
        out=pd.concat(pool.map(prep.work,ch),ignore_index=True)
        for c in extra: out[c]=df[c].values
        return out
    prepare(s1,['state','fold']).to_parquet(OUT+'bench_s1.parquet'); log('s1 saved')
    for s in ('s2','s3'): prepare(pools[s],['owner','state']).to_parquet(OUT+f'bench_{s}.parquet'); log(s,'saved')
    json.dump(dict(region=region,folds=folds),open(OUT+'folds.json','w'),indent=1); json.dump(stats,open(OUT+'bench_stats.json','w'),indent=1,default=float); log('done')
