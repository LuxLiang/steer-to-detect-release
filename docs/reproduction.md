# Reproduction and metric semantics

## Protocol

1. Fix the base model identifier and revision, data version, disjoint splits,
   random seeds, steering position/layer, readout layers, and token pooling.
2. Train on the training split. Select thresholds on the validation split.
3. Evaluate once on the held-out test split, carrying validation thresholds over.
4. Save configuration, package versions, per-sample scores, and summary metrics.

`configs/train_defaults.json` reflects source defaults, including seed 2025,
512 paired examples, steering layer 11, lambda 2, last 8 post-steer blocks,
last 25% of tokens, AdamW learning rate 0.001, 10 epochs, and batch size 8.
The source optimizer uses PyTorch AdamW's default weight decay in addition to
any explicitly supplied `--l2_reg`; those are different mechanisms.

Checkpoint names derive from the input JSON basename, or the suffix after
`_llm_type_`. Sampling runs additionally use `_run_N`. Reusing an output directory
and the same input name overwrites its checkpoint. Test JSON basenames must be
unique within an evaluation run because outputs are keyed by basename.

## Scores and thresholds

The score is `(cos(rep, centroid_1) - cos(rep, centroid_0)) / temperature`.
Scores are not calibrated probabilities.

- `roc_auc`: ranking performance on the indicated split.
- `tpr_at_target_fpr.*.tpr_interp`: interpolation of that split's empirical ROC.
  On test data, this is a test-ROC summary, not an independently calibrated
  operating threshold.
- `metrics_at_lowfpr_threshold`: actual performance using a validation-selected
  human-score quantile threshold (`method="higher"`, prediction `score >= threshold`).
  Ties and finite validation samples can make achieved FPR exceed the target.
- `metrics_at_youden_threshold`: performance at the validation-selected maximum
  Youden-J threshold.

The target list defaults to 0.01, 0.005, and 0.0001. Small validation sets cannot
substantiate the smallest target rates; report sample counts and achieved FPR.
The original JSON writer may emit `Infinity` for a non-finite ROC
threshold; consumers requiring strict JSON should handle that explicitly.

## No-steering protocol

The no-steering ablation recomputes centroids from training representations without
steering. Its last-k-token pooling defaults to 64, while the main method uses a
ratio. `--centroid_normalize_per_sample` additionally changes centroid estimation.
Keep these settings explicit; this variant does not isolate the effect of steering
alone. Its legacy checkpoint is read for metadata rather than to inject a vector.

## Summarize completed runs

```bash
python scripts/summarize_results.py outputs/eval --output outputs/summary.csv
```

This exports per-test AUROC and validation-threshold operating metrics from each
`summary.json`.

## Model support

Block discovery recognizes `model.layers`, `transformer.h`, and
`model.decoder.layers`. Attention/MLP steering additionally requires `self_attn`
and `mlp` on the chosen block. Training uses `device_map="auto"`; evaluation uses
one device. Tests use tiny Llama on CPU.
