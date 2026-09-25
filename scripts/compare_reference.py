import json, os, subprocess, sys, tempfile
from pathlib import Path
import torch
import argparse
parser = argparse.ArgumentParser(description='Compare this release against trusted original research scripts using an offline tiny model.')
parser.add_argument('--reference-dir', type=Path, required=True)
source = parser.parse_args().reference_dir.resolve()
root = Path(__file__).resolve().parents[1]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
bridge='import torch,runpy,sys; original=torch.load; torch.load=lambda *a,**k: original(*a,**dict(k,weights_only=False)); script=sys.argv.pop(1); runpy.run_path(script,run_name="__main__")'
def run(script,args,reference=False):
 cmd=[sys.executable,'-c',bridge,str(script)] if reference else [sys.executable,str(script)]
 r=subprocess.run(cmd+list(map(str,args)),cwd=root,env=env,text=True,capture_output=True)
 if r.returncode: raise RuntimeError(r.stdout+r.stderr)
with tempfile.TemporaryDirectory(prefix='s2d_reference_') as d:
 d=Path(d); model=d/'model'
 run(root/'scripts/create_tiny_model.py',[model])
 common=['--model_name_or_path',model,'--train_data_path',root/'examples/train.json','--model_dtype','float32','--steer_layer','0','--readout_last_n_layers','2','--max_length','32','--epochs','1','--batch_size','2','--num_exemplars','4']
 run(source/'s2d_train.py',common+['--out_dir',d/'reference'],True)
 run(root/'train.py',common+['--out_dir',d/'release'])
 checkpoints=[torch.load(d/n/'s2d_ckpt_train.pt',weights_only=False) for n in ['reference','release']]
 torch.testing.assert_close(checkpoints[0]['sv_state_dict']['v'],checkpoints[1]['sv_state_dict']['v'],rtol=0,atol=0)
 import numpy as np
 np.testing.assert_array_equal(checkpoints[0]['centroids'],checkpoints[1]['centroids'])
 common=['--model_name_or_path',model,'--ckpt_path',d/'reference/s2d_ckpt_train.pt','--valid_data_path',root/'examples/valid.json','--test_data_paths',root/'examples/test.json','--dtype','float32','--batch_size','2']
 run(source/'s2d_eval.py',common+['--out_dir',d/'eval_reference'],True)
 run(root/'evaluate.py',common+['--out_dir',d/'eval_release'])
 for name in ['test.json_scores.json','summary.json']:
  vals=[json.loads((d/n/'s2d_ckpt_train'/name).read_text()) for n in ['eval_reference','eval_release']]
  assert vals[0]==vals[1],name
 print('PASS: original and release have bit-identical trained vector/centroids, and identical evaluation scores/summary on offline tiny Llama.')
