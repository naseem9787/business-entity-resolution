"""Run the UNMODIFIED production predict() (pilot/make_submission.py, production models copied from submission_work/models) on replay data.
usage: python run_replay.py <work_dir_name> [--no-india-broadcast]"""
import os,sys,shutil,argparse,importlib
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,'..','..','..')); os.chdir(ROOT); sys.path.insert(0,'pilot')
ms=importlib.import_module('make_submission')
if __name__=='__main__':
    name=sys.argv[1]; W=os.path.join(HERE,name); os.makedirs(W,exist_ok=True)
    if not os.path.exists(os.path.join(W,'models')): shutil.copytree('submission_work/models',os.path.join(W,'models'))
    if '--no-india-broadcast' in sys.argv:
        src=os.path.join(HERE,'work_prod','parts')          # re-use the production partition, drop India's unroutable broadcast pool, keep only India S1
        for s in ('s1','s2','s3'):
            os.makedirs(os.path.join(W,'parts',s),exist_ok=True)
            for f in os.listdir(os.path.join(src,s)):
                if not f.startswith('India__'): continue
                if s!='s1' and f.startswith('India___NOSTATE__'): continue
                shutil.copyfile(os.path.join(src,s,f),os.path.join(W,'parts',s,f))
    a=argparse.Namespace(data_dir=os.path.join(HERE,'replay_data'),work=W,out=os.path.join(W,'output'),exp1='pilot/exp1/',exp2='pilot/exp2/',team='replay',zip_dir=W,smoke_rows=0)
    ms.predict(a)
