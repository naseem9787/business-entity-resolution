import pandas as pd, sys, multiprocessing as mp
sys.path.insert(0,'pilot')
from norm import *
def work(df):
    d=pd.DataFrame({'id':df.entity_id.values,'country':df.country.values,'name_raw':df.business_name.values,'addr_raw':df.business_address.values})
    nn=[clean(x) for x in df.business_name]; an=[clean(x) for x in df.business_address]
    d['nn']=nn; d['an']=an
    d['nt']=[translit(x) for x in nn]; d['at']=[translit(x) for x in an]
    d['ac']=[' '.join(addr_canon(x)) for x in d['at']]
    d['nc']=[' '.join(core(x.split())) for x in d['nt']]
    return d
if __name__=='__main__':
    for s in ['s1','s2','s3']:
        df=pd.read_parquet(f'pilot/{s}.parquet'); ch=[df.iloc[i:i+50000] for i in range(0,len(df),50000)]
        with mp.Pool(6) as p: out=pd.concat(p.map(work,ch),ignore_index=True)
        out.to_parquet(f'pilot/{s}_n.parquet'); print(s,len(out),flush=True)
