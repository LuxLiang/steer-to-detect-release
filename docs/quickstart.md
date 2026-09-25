# Offline quick start

Run these commands from the repository root after installing dependencies.

## Train and evaluate

These commands create a **random, tiny Llama**, then run actual training and
held-out evaluation. The example pairs are synthetic format fixtures, not real
human/AI provenance labels. Their metrics are not evidence of detection quality.
No pretrained model or benchmark download is needed after installing dependencies.

```bash
python scripts/create_tiny_model.py outputs/tiny-model

python train.py \
  --model_name_or_path outputs/tiny-model \
  --train_data_path examples/train.json \
  --out_dir outputs/demo-train \
  --model_dtype float32 \
  --steer_layer 0 --readout_last_n_layers 2 \
  --max_length 32 --epochs 1 --batch_size 2 --num_exemplars 4

python evaluate.py \
  --model_name_or_path outputs/tiny-model \
  --ckpt_path outputs/demo-train/s2d_ckpt_train.pt \
  --valid_data_path examples/valid.json \
  --test_data_paths examples/test.json \
  --out_dir outputs/demo-eval \
  --dtype float32 --batch_size 2
```

Evaluation writes `summary.json` and per-example `test.json_scores.json` under
`outputs/demo-eval/s2d_ckpt_train/`. Higher scores indicate the model-generated
class (`1`); human-written text is class `0`. Scores are differences of
cosine-similarity logits, not calibrated probabilities.

## No-steering ablation

```bash
python evaluate_no_steer.py \
  --model_name_or_path outputs/tiny-model \
  --ckpt_path outputs/demo-train/s2d_ckpt_train.pt \
  --train_data_path examples/train.json \
  --valid_data_path examples/valid.json \
  --test_data_paths examples/test.json \
  --out_dir outputs/demo-no-steer --dtype float32 --batch_size 2
```

This preserves the existing **fixed-token** ablation: it recomputes class centroids
from training representations without steering, using the last 64 valid tokens by
default. Main S2D uses a token **ratio**. This is not a controlled “steering only”
comparison; see the reproduction notes before interpreting its results.


For watermark examples, see [watermarking.md](watermarking.md).
