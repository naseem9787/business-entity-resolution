#!/usr/bin/env python3
"""v2 submission pipeline = production pipeline (make_submission.py, unchanged, imported) with:
  * norm3 tokenizer (Indic combining marks kept inside words) injected as module 'norm' BEFORE any normalisation runs (also in worker processes)
  * v2 per-shard evidence features (pilot/v2feats.py) + v2 stage-1 LightGBM
  * calibrated conditional decision layer: s >= thr(p) and margin >= MARGIN, thr = T_EMPTY (empty S2/S3 address) / T_NATIVE (non-Latin name) / T_BASE
Stages: predict | validate  (models in <work>/models_v2: stage1.txt + meta.json; ranker.txt copied from production)
SAVE_FEATS=1 additionally writes the full per-shard feature matrix (used to build production-density training data)."""
import os,sys,json,glob,time,gc,collections,argparse,shutil,subprocess,importlib
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'exp3')); sys.path.insert(0,HERE)
import norm3; sys.modules['norm']=norm3                                     # <- tokenizer fix
import numpy as np,pandas as pd,lightgbm as lgb
import make_submission as ms
import v2feats
from feats import build,idf_tables
TRANSLIT=os.path.join(HERE,'final_optimization','02_ablation','translit_map.json')
def log(*a): print(f'[{time.time()-ms.T0:7.0f}s]',*a,flush=True)
def process_shard_v2(Q,Ps,ranker,stage1,feats,tmp,save_path=None):
    parts=[]; PP=[]
    for src in ('s2','s3'):
        P=Ps[src]
        if P is None or len(P)==0: continue
        ms.SRC3[0]=float(src=='s3'); F=ms.shard_source(Q,P,ranker,tmp); F['src']=src
        Qi=Q.set_index('id'); Pi=P.set_index('id'); idn,ida=idf_tables([Q,P]); r=ms.CFG['N_REF']/(len(Q)+len(P)); idn['__cnt__']={k:v*r for k,v in idn['__cnt__'].items()}
        X=pd.concat([build(F.iloc[s:s+300000],Qi,Pi,idn,ida) for s in range(0,len(F),300000)],ignore_index=True)
        X['q_id']=F.q_id.values; X['p_id']=F.p_id.values; X['src']=src
        X=X.sort_values(['q_id','fused'],ascending=[True,False],kind='stable').reset_index(drop=True); f=X.fused.values; g=pd.Series(f).groupby(X.q_id.values)
        X['fused_rank']=g.rank(ascending=False,method='first').values; X['fused_gap_top']=f-g.transform('max').values; X['n_cands']=g.transform('size').values; parts.append(X)
        PP.append(P[['id','country','nc','ac','an','nn','addr_raw','name_raw']].assign(src=src))
    if not parts: return pd.DataFrame({'q_id':[],'p_id':[],'src':[],'s':[],'p_empty':[],'p_native':[]})
    X=pd.concat(parts,ignore_index=True); g=X.groupby('q_id').fused; X['fused_rank_all']=g.rank(ascending=False,method='first'); X['fused_gap_top_all']=X.fused-g.transform('max')
    P=pd.concat(PP,ignore_index=True).drop_duplicates('id'); X=v2feats.add(X,Q,P,TRANSLIT,norm3.LEGAL)
    Pi=P.set_index('id'); X['p_empty']=(Pi.addr_raw.reindex(X.p_id).values=='').astype(np.int8); X['p_native']=Pi.name_raw.reindex(X.p_id).str.contains('[^\x00-ɏ]',regex=True).fillna(False).values.astype(np.int8)
    if save_path: X.to_parquet(save_path)
    X['s']=stage1.predict(X[feats].to_numpy(np.float32)) if stage1 is not None else 0.0
    return X[['q_id','p_id','src','s','p_empty','p_native']]
def decide(R,rule):
    R=R.sort_values('s',ascending=False,kind='stable').drop_duplicates(['q_id','p_id']).reset_index(drop=True)
    smo=ms.margin_other(pd.factorize(R.p_id)[0],R.s.values.astype(np.float64))
    thr=np.select([R.p_empty.values==1,R.p_native.values==1],[rule['T_EMPTY'],rule['T_NATIVE']],rule['T_BASE']); keep=(R.s.values>=thr)&(R.s.values-smo>=rule['MARGIN']); return R,keep
def predict(a):
    M=os.path.join(a.work,'models_v2'); meta=json.load(open(os.path.join(M,'meta.json'))) if os.path.exists(os.path.join(M,'meta.json')) else {'stage1_features':None,'rule':None}
    ranker=lgb.Booster(model_file=os.path.join(M,'ranker.txt')); stage1=lgb.Booster(model_file=os.path.join(M,'stage1.txt')) if os.path.exists(os.path.join(M,'stage1.txt')) else None
    rm=json.load(open('submission_work/models/meta.json'))['route_map']; a2=argparse.Namespace(**{**vars(a)}); ms.partition(a2,{'route_map':rm},a.data_dir)
    shards=collections.defaultdict(set)
    for f in glob.glob(os.path.join(a.work,'parts','s1','*.parquet')): c,k,_=os.path.basename(f)[:-8].split('__'); shards[c].add(k)
    tmp=os.path.join(a.work,'tmp'); os.makedirs(tmp,exist_ok=True); os.makedirs(os.path.join(a.work,'scored'),exist_ok=True); save=os.environ.get('SAVE_FEATS')=='1'
    if save: os.makedirs(os.path.join(a.work,'feats'),exist_ok=True)
    n=sum(len(v) for v in shards.values()); done=0
    for c in sorted(shards):
        for k in sorted(shards[c]):
            done+=1; fn=os.path.join(a.work,'scored',f'{c}__{k}.parquet')
            if os.path.exists(fn): continue
            Q=ms.load_parts(a2,'s1',c,k); Ps={}
            for src in ('s2','s3'):
                parts=[ms.load_parts(a2,src,c,k)]
                if k not in ('_ALL','_NOSTATE'): parts.append(ms.load_parts(a2,src,c,'_NOSTATE'))
                parts=[p for p in parts if p is not None]; Ps[src]=pd.concat(parts,ignore_index=True) if parts else None
            t=time.time(); R=process_shard_v2(Q,Ps,ranker,stage1,meta['stage1_features'],tmp,os.path.join(a.work,'feats',f'{c}__{k}.parquet') if save else None)
            R.to_parquet(fn); log(f'shard {done}/{n} {c}/{k}: {len(Q)} S1, {len(R)} rows, {time.time()-t:.0f}s'); del Q,Ps,R; gc.collect()
    if meta.get('rule') is None: log('no stage-1 model/rule -> features only'); return
    R=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.work,'scored','*.parquet')))],ignore_index=True); R,keep=decide(R,meta['rule'])
    ids1=pd.read_csv(os.path.join(a.data_dir,'test_source1.tsv'),sep='\t',dtype=str,keep_default_na=False,quoting=3,usecols=['entity_id']).entity_id.tolist(); os.makedirs(a.out,exist_ok=True)
    def write(df,path,col):
        g=df.sort_values('q_id',kind='stable').groupby('q_id',sort=False).p_id.agg(','.join).to_dict()
        with open(path,'w',encoding='utf8',newline='') as f:
            f.write(f'source1_entity_id\t{col}\n')
            for q in ids1: f.write(q+'\t'+g.get(q,'')+'\n')
    write(R,os.path.join(a.out,'candidate_pairs.tsv'),'candidate_entity_ids'); write(R[keep],os.path.join(a.out,'matching_results.tsv'),'matched_entity_ids'); log('predict done: rows',len(R),'matches',int(keep.sum()))
def validate(a):
    p=subprocess.run([sys.executable,'utils/validate_submission.py','--matching',os.path.join(a.out,'matching_results.tsv'),'--candidate',os.path.join(a.out,'candidate_pairs.tsv'),'--test-dir',a.data_dir,'--check-ids'],capture_output=True,text=True); print(p.stdout[-1500:]); return p.returncode==0
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['predict','validate']); ap.add_argument('--data-dir',default='dataset/test'); ap.add_argument('--work',default='submission_work_v2'); ap.add_argument('--out',default='output_v2'); a=ap.parse_args()
    os.makedirs(a.work,exist_ok=True)
    if a.stage=='predict': predict(a)
    else: sys.exit(0 if validate(a) else 1)
