"""Exp2 step 3: retrieval-loss taxonomy for every true pair MISSED by Exp1's top-15 (12,793 pairs).
For each miss: (a) retrieval status, (b) text cause flags + primary cause, (c) which new channel / config recovers it."""
import sys,os,json,re,collections,numpy as np,pandas as pd
sys.path.insert(0,'pilot')
from rapidfuzz import fuzz
OUT='pilot/exp2/'; BENCH='pilot/exp1/'
ORDW=set('first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth'.split()); ORDN=re.compile(r'^\d+(st|nd|rd|th)$')
truth=pd.read_parquet(BENCH+'truth_pairs.parquet'); f1=pd.read_parquet(BENCH+'feats_top15.parquet',columns=['q_id','p_id','label']); hit=set(zip(f1.q_id[f1.label==1],f1.p_id[f1.label==1]))
miss=truth[[ (q,p) not in hit for q,p in zip(truth.q_id,truth.p_id)]].reset_index(drop=True); print('true pairs',len(truth),'exp1 top15 misses',len(miss),flush=True)
S1=pd.read_parquet(BENCH+'bench_s1.parquet').set_index('id'); rows=[]
def group_bounds(qi):
    st=np.r_[True,qi[1:]!=qi[:-1]]; s=np.flatnonzero(st); e=np.r_[s[1:],len(qi)]; return s,e,qi[s]
for country in ('India','US'):
    for src in ('s2','s3'):
        P=pd.read_parquet(BENCH+f'bench_{src}.parquet'); P=P[P.country==country].reset_index(drop=True); NP=len(P); pidx=dict(zip(P.id,range(NP)))
        qinfo=pd.read_parquet(OUT+f'qinfo_{country}.parquet'); qidx=dict(zip(qinfo.id,range(len(qinfo))))
        m=miss[(miss.src==src)&miss.q_id.isin(qidx)&miss.p_id.isin(pidx)].copy(); m['qi']=m.q_id.map(qidx); m['pj']=m.p_id.map(pidx)
        if m.empty: continue
        D=pd.read_parquet(OUT+f'union_{country}_{src}.parquet'); R=pd.read_parquet(OUT+f'ranked_{country}_{src}.parquet'); D['fused']=R.fused.values; D['G']=R.score_G.values; D['H']=R.score_H.values
        keys=D.qi.values.astype(np.int64)*NP+D.pj.values; mk=m.qi.values.astype(np.int64)*NP+m.pj.values; pos=np.searchsorted(keys,mk); pos=np.minimum(pos,len(keys)-1); present=keys[pos]==mk
        qs,qe,qq=group_bounds(D.qi.values); qstart=dict(zip(qq,qs)); qend=dict(zip(qq,qe))
        allowedA=np.minimum.reduce([D.r_nw.values,D.r_aw.values,D.r_nc.values,D.r_ac.values])<999
        def rank_of(i,col,allowed=None):
            a,b=qstart[D.qi.values[i]],qend[D.qi.values[i]]; sc=D[col].values[a:b]; ok=np.ones(b-a,bool) if allowed is None else allowed[a:b]; return int((sc[ok]>sc[i-a]).sum())+1
        dfn=collections.Counter(t for s in P.nc for t in set(s.split())); dfa=collections.Counter(t for s in P.ac for t in set(s.split())); md=max(50,int(0.0066*NP))
        for r,(row,pr,i) in enumerate(zip(m.itertuples(),present,pos)):
            q=S1.loc[row.q_id]; p=P.iloc[row.pj]; d={'country':country,'src':src,'q_id':row.q_id,'p_id':row.p_id,'in_union':bool(pr)}
            # ---- text causes
            qa,pa=q.ac.split(),p.ac.split(); nq_=set(t for t in qa if any(c.isdigit() for c in t)); np_=set(t for t in pa if any(c.isdigit() for c in t)); aq=set(qa)-nq_; ap=set(pa)-np_
            ajac=len(aq&ap)/max(len(aq|ap),1); nsim=fuzz.ratio(q.nc.replace(' ',''),p.nc.replace(' ',''))/100; asim=fuzz.token_set_ratio(q.ac,p.ac)/100 if pa else 0.0
            shared=(set(q.nc.split())&set(p.nc.split()))|(set(qa)&set(pa)); supp=bool(shared) and all(max(dfn.get(t,0),dfa.get(t,0))>md for t in shared)
            d.update(f_empty_addr=p.addr_raw=='',f_native=(not p.name_raw.isascii()) or (not p.addr_raw.isascii()),
                     f_number=bool(nq_ and np_ and nq_!=np_ and ajac>=0.4),f_ordinal=any(ORDN.match(t) or t in ORDW for t in qa+pa),
                     f_common_suppr=supp,f_name_typo=(nsim<0.97 and nsim>=0.5 and (ajac>=0.5 or asim>=0.85)),f_addr_typo=(nsim>=0.9 and pa!=[] and asim<0.8),
                     f_alias_name=(nsim<0.5 and (ajac>=0.5 or asim>=0.85)),name_sim=nsim,addr_sim=asim)
            d['text_cause']=('empty_address' if d['f_empty_addr'] else 'native_script' if d['f_native'] else 'number_or_ordinal_variation' if d['f_number'] else 'common_token_suppression' if d['f_common_suppr'] else 'name_typo' if d['f_name_typo'] else 'address_typo' if d['f_addr_typo'] else 'alias_name' if d['f_alias_name'] else 'other')
            if pr:
                d.update(ch_baseline4=bool(allowedA[i]),ch_nc3=D.r_nc3.values[i]<999,ch_an=D.r_an.values[i]<999,ch_quota=D.r_quota.values[i]<999,ch_key=bool(D.key_kx.values[i]>0 or D.key_kn.values[i]>0),ch_reverse=D.rev_rank.values[i]<999,
                         rank_A=rank_of(i,'fused',allowedA) if allowedA[i] else 9999,rank_G=rank_of(i,'G'),rank_H=rank_of(i,'H'))
            rows.append(d)
        print(country,src,'misses',len(m),'in exp2 union',int(present.sum()),flush=True)
T=pd.DataFrame(rows)
for c in ('ch_baseline4','ch_nc3','ch_an','ch_quota','ch_key','ch_reverse'): T[c]=T[c].fillna(False).astype(bool)
for c in ('rank_A','rank_G','rank_H'): T[c]=T[c].fillna(99999)
T['status']=np.where(~T.in_union,'absent_from_all_channels',np.where(T.ch_baseline4,'in_baseline_channels_but_ranked_below_top15','found_only_by_new_channel'))
T['H_top15']=T.rank_H<=15; T['G_top15']=T.rank_G<=15; T.to_parquet(OUT+'taxonomy_rows.parquet')
res={'total_misses':len(T),'status':T.status.value_counts().to_dict(),'text_cause':T.text_cause.value_counts().to_dict(),
     'status_x_text_cause':pd.crosstab(T.text_cause,T.status).to_dict(),'by_country':T.groupby('country').size().to_dict(),'by_country_status':pd.crosstab(T.country,T.status).to_dict(),
     'flag_counts':{c:int(T[c].sum()) for c in T.columns if c.startswith('f_')},'flag_counts_by_country':{c:T.groupby('country')[c].sum().to_dict() for c in T.columns if c.startswith('f_')}}
rec={}
for tc,g in T.groupby('text_cause'):
    rec[tc]=dict(n=len(g),**{c:int(g[c].sum()) for c in ('ch_baseline4','ch_nc3','ch_an','ch_quota','ch_key','ch_reverse')},recovered_by_G_top15=int(g.G_top15.sum()),recovered_by_H_top15=int(g.H_top15.sum()))
res['recovery_by_text_cause']=rec
res['recovery_total']=dict(n=len(T),in_any_new_channel=int((T.ch_nc3|T.ch_an|T.ch_quota|T.ch_key|T.ch_reverse).sum()),G_top15=int(T.G_top15.sum()),H_top15=int(T.H_top15.sum()),
   only_new_channel_found={c:int((T[c]&~T.ch_baseline4).sum()) for c in ('ch_nc3','ch_an','ch_quota','ch_key','ch_reverse')})
json.dump(res,open(OUT+'taxonomy.json','w'),indent=1,default=float); print(json.dumps(res,indent=1,default=float))
