import json,glob,sys
import pandas as pd
pd.set_option('display.width',250,'display.max_columns',30)
rows=[];curve=[]
for f in sorted(glob.glob('pilot/report_block_*.json')):
    R=json.load(open(f)); m=R['meta']; tag=f"{m['country']}/{m['src']}"
    for k,v in R.items():
        if k in('meta','union_recall_ceiling'): continue
        if v['kind']=='join': rows.append((tag,k,'join',v['recall'],v['cand']['mean'],v['cand']['p99'],v['cand']['max']))
        elif v['kind']=='sparse': rows.append((tag,k,'sparse(any shared feat)',v['recall_any_shared'],v['cand_any_shared']['mean'],v['cand_any_shared']['p99'],v['cand_any_shared']['max']))
        else:
            for K in ('5','10','20','30','50','100'): curve.append((tag,k,int(K),v['recall_at_K'][K],v['mean_cands_at_K'][K]))
    rows.append((tag,'UNION of 4 retrievers (top100 each)','union',R['union_recall_ceiling'],R['meta']['union_pairs']/m['nq'],None,None))
d=pd.DataFrame(rows,columns=['combo','strategy','kind','recall','mean_cand','p99_cand','max_cand'])
print(d.pivot_table(index=['strategy','kind'],columns='combo',values='recall').round(3).to_string())
print(); print(d.pivot_table(index=['strategy','kind'],columns='combo',values='mean_cand').round(1).to_string())
c=pd.DataFrame(curve,columns=['combo','fusion','K','recall','mean_cands'])
print(); print(c[c.fusion.isin(['fuse_addr_heavy','fuse_sum4'])].pivot_table(index=['fusion','K'],columns='combo',values='recall').round(4).to_string())
