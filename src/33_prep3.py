"""Exp3: re-derive normalised columns of the Exp1 benchmark frames with the corrected tokenizer (norm3). Raw columns and extras (owner/state/fold) are kept. -> pilot/exp3/bench3_{s1,s2,s3}.parquet"""
import sys,multiprocessing as mp,pandas as pd
sys.path.insert(0,'pilot/exp3'); sys.path.insert(0,'pilot')
from norm3 import clean,translit,core,addr_canon
def work(df):
    d=df[[c for c in df.columns if c not in('nn','an','nt','at','ac','nc')]].copy()
    d['nn']=[clean(x) for x in d.name_raw]; d['an']=[clean(x) for x in d.addr_raw]; d['nt']=[translit(x) for x in d.nn]; d['at']=[translit(x) for x in d.an]
    d['ac']=[' '.join(addr_canon(x)) for x in d['at']]; d['nc']=[' '.join(core(x.split())) for x in d['nt']]; return d
if __name__=='__main__':
    for s in ('s1','s2','s3'):
        df=pd.read_parquet(f'pilot/exp1/bench_{s}.parquet'); ch=[df.iloc[i:i+50000] for i in range(0,len(df),50000)]
        with mp.Pool(6) as p: out=pd.concat(p.map(work,ch),ignore_index=True)
        out.to_parquet(f'pilot/exp3/bench3_{s}.parquet'); print(s,len(out),flush=True)
