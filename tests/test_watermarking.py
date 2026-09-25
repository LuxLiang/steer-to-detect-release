import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from watermarking.algorithms import GumbelProcessor, KGWProcessor, TokenDetector, make_processor

ROOT=Path(__file__).resolve().parents[1]


def metadata(algorithm):
    config=json.loads((ROOT/'watermarking/configs/synthid.json').read_text())
    return dict(algorithm=algorithm,key=42,gamma=.25,delta=2.,vocab_size=32,rng_device_type='cpu',temperature=.7,top_k=8,synthid_config=config)


def test_processors_are_independent_of_batch_composition():
    ids=torch.tensor([[1,2],[3,4]])
    scores=torch.zeros(2,32)
    for processor in [GumbelProcessor(42),KGWProcessor(.25,2.,42)]:
        batch=processor(ids,scores)
        single=torch.cat([processor(ids[i:i+1],scores[i:i+1]) for i in range(2)])
        torch.testing.assert_close(batch,single,rtol=0,atol=0)


def test_synthid_state_reset_and_detection_alignment():
    meta=metadata('synthid')
    first,second=make_processor(meta,'cpu'),make_processor(meta,'cpu')
    ids=torch.tensor([[8,9,10,11,12]])
    for token in [13,14,15,16,17]:
        scores=torch.arange(32,dtype=torch.float32).reshape(1,-1)/32
        a,b=first(ids,scores),second(ids,scores)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ids=torch.cat([ids,torch.tensor([[token]])],dim=1)
    # Context at the fifth step is exactly the first four continuation tokens.
    token_ids=ids[:,5:]
    g=first.compute_g_values(token_ids)
    keys,_=first._compute_keys(token_ids[:,:4],token_ids[:,4:])
    torch.testing.assert_close(g,first.sample_g_values(keys),rtol=0,atol=0)
    score=TokenDetector(meta,'cpu').score(ids[0].tolist(),5,5)
    assert score==float(g.mean())
    with pytest.raises(ValueError,match='RNG device'):
        TokenDetector(dict(meta,rng_device_type='cuda'),'cpu')


@pytest.mark.parametrize('algorithm',['gumbel','kgw','synthid'])
def test_generation_detection_and_s2d_export(tmp_path,algorithm):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    def run(*args):
        result=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    model=tmp_path/'model'
    run('scripts/create_tiny_model.py',model)
    dataset,metrics,pairs=tmp_path/'dataset.json',tmp_path/'metrics.json',tmp_path/'pairs.json'
    run('-m','watermarking.generate','--algorithm',algorithm,'--model_name_or_path',model,
        '--data_path','examples/test.json','--output',dataset,'--prompt_len','4','--gen_len','8',
        '--max_samples','2','--top_k','8','--dtype','float32','--device','cpu')
    content=json.loads(dataset.read_text())
    assert len(content['records'])==2
    for record in content['records']:
        for key in ['full_ids_human','full_ids_watermarked','full_ids_unwatermarked']:
            assert len(record[key])==12
            assert record[key][:4]==record['full_ids_human'][:4]
    run('-m','watermarking.evaluate','--dataset',dataset,'--output',metrics,'--lengths','5','8')
    result=json.loads(metrics.read_text())
    for size in ['5','8']:
        for reference in ['human','unwatermarked']:
            assert result['results'][size][reference]['num_pairs']==2
            assert 0<=result['results'][size][reference]['roc_auc']<=1
    run('-m','watermarking.export_pairs','--dataset',dataset,'--output',pairs)
    content=json.loads(pairs.read_text())
    assert content and all(p['human_text'] and p['direct_prompt'] for p in content)
    # Verify the exported format is actually accepted by S2D's own reader.
    from s2d.data import expand_pairs_for_evaluation
    texts,labels,ids,stats=expand_pairs_for_evaluation(content,'direct_prompt','human_text')
    assert stats['used_pairs']==len(content)
