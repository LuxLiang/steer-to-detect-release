"""Offline verification: actual model training, checkpoint reload and evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from s2d.metrics import select_threshold_np_quantile
from s2d.readout import last_ratio_tokens_mean_pool, evaluation_last_ratio_tokens_mean_pool

ROOT = Path(__file__).resolve().parents[1]


def test_masked_pooling():
    # Padded positions must not contribute, for either padding convention.
    h = torch.tensor([[[99.], [99.], [2.], [4.]], [[1.], [3.], [5.], [7.]]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    expected = torch.tensor([[4.], [6.]])
    torch.testing.assert_close(last_ratio_tokens_mean_pool(h, mask, .5), expected)
    torch.testing.assert_close(evaluation_last_ratio_tokens_mean_pool(h, mask, .5), expected)
    torch.testing.assert_close(evaluation_last_ratio_tokens_mean_pool(h.flip(1), mask.flip(1), .5, 'right'), torch.tensor([[2.], [2.]]))


def test_quantile_ties_are_reported():
    # Legacy >= threshold handling does not guarantee achieved FPR <= target.
    result = select_threshold_np_quantile(np.array([0, 0, 0, 1]), np.array([1., 1., 1., 2.]), .01)
    assert result['fpr'] == 1.0


def test_offline_training_and_evaluation(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    def run(*args):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
    model, output = tmp_path / 'model', tmp_path / 'train'
    run('scripts/create_tiny_model.py', model)
    run('scripts/run_config.py', 'train', 'configs/train_defaults.json', '--model_name_or_path', model, '--train_data_path', 'examples/train.json', '--out_dir', output, '--model_dtype', 'float32', '--steer_layer', '0', '--readout_last_n_layers', '2', '--max_length', '32', '--epochs', '1', '--batch_size', '2', '--num_exemplars', '4')
    ckpt = output / 's2d_ckpt_train.pt'
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert torch.isfinite(state['sv_state_dict']['v']).all()
    assert state['sv_state_dict']['v'].abs().sum() > 0
    assert state['readout_layers'] == [0, 1]
    assert state['training_args']['seed'] == 2025
    common = ['--model_name_or_path', model, '--ckpt_path', ckpt, '--valid_data_path', 'examples/valid.json', '--test_data_paths', 'examples/test.json', '--dtype', 'float32', '--batch_size', '2']
    for name in ['first', 'second']:
        run('evaluate.py', *common, '--out_dir', tmp_path / name)
    summaries = [json.loads((tmp_path / name / 's2d_ckpt_train' / 'summary.json').read_text()) for name in ['first', 'second']]
    assert summaries[0] == summaries[1]
    assert 0 <= summaries[0]['test_results']['test.json']['roc_auc'] <= 1
    run('scripts/summarize_results.py', tmp_path / 'first', '--output', tmp_path / 'summary.csv')
    import csv
    with (tmp_path / 'summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert all(row['dataset'] == 'test.json' for row in rows)
    # Exercise the preserved fixed-token, train-centroid ablation as well.
    run('evaluate_no_steer.py', *common, '--train_data_path', 'examples/train.json', '--out_dir', tmp_path / 'ablation')
    assert list((tmp_path / 'ablation').rglob('*.json'))
