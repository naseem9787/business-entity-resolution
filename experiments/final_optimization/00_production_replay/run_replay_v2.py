"""v2 replay: norm3 + v2 features on the SAME held-out replay states; saves full feature matrices (work_v2/feats) for offline model training/evaluation."""
import os,sys,shutil,argparse
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,'..','..','..')); os.chdir(ROOT); sys.path.insert(0,'pilot')
import make_submission_v2 as v2          # top-level import -> norm3 injected in every worker process too
if __name__=='__main__':
    W=os.path.join(HERE,'work_v2'); os.makedirs(os.path.join(W,'models_v2'),exist_ok=True); shutil.copyfile('submission_work/models/ranker.txt',os.path.join(W,'models_v2','ranker.txt'))
    os.environ['SAVE_FEATS']='1'; v2.predict(argparse.Namespace(data_dir=os.path.join(HERE,'replay_data'),work=W,out=os.path.join(W,'output')))
