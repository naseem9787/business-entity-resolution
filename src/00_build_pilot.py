import pandas as pd, numpy as np, json
D='dataset/train/'
rd=lambda f: pd.read_csv(D+f,sep='\t',dtype=str,keep_default_na=False,quoting=3)
s1,s2,s3,gt=rd('train_source1.tsv'),rd('train_source2.tsv'),rd('train_source3.tsv'),rd('train_ground_truth.tsv')
print(len(s1),len(s2),len(s3),len(gt))
gt['ids']=gt.matched_entity_ids.map(lambda x:[i for i in x.split(',') if i])
# pair table
g=gt[['source1_entity_id','ids']].explode('ids').dropna()
g=g[g.ids!='']
g.columns=['s1','m']
print('pairs',len(g),'dup matched ids',g.m.duplicated().sum(),'distinct',g.m.nunique())
matched=set(g.m)
print('S2 unmatched',(~s2.entity_id.isin(matched)).sum(),'S3 unmatched',(~s3.entity_id.isin(matched)).sum())
print('S1 dup names',s1.business_name.duplicated().sum())
for n,s in [('s1',s1),('s2',s2),('s3',s3)]:
    print(n,'country',s.country.value_counts().to_dict(),'empty addr',(s.business_address=='').mean().round(3),'empty name',(s.business_name=='').mean().round(4))
rng=np.random.default_rng(0)
N=200000
samp=s1.sample(N,random_state=0)
sid=set(samp.entity_id)
gp=g[g.s1.isin(sid)]
frac=N/len(s1)
def pool(s):
    mine=s.entity_id.isin(set(gp.m))
    rest=s[~mine].sample(frac=frac,random_state=1)
    return pd.concat([s[mine],rest])
p2,p3=pool(s2),pool(s3)
print('pool',len(p2),len(p3))
samp.to_parquet('pilot/s1.parquet');p2.to_parquet('pilot/s2.parquet');p3.to_parquet('pilot/s3.parquet');gp.to_parquet('pilot/gt_pairs.parquet')
gp2=gp.copy();gp2['src']=gp2.m.str[:2]
print(gp2.groupby('src').size())
