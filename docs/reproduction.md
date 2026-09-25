# Reproduction and metric semantics

## What is preserved

The core uses the research script's steering injection, FP32 trainable vector,
readout order, pooling arithmetic, exemplar initialization, AdamW optimization,
EMA centroids, scoring, and threshold routines. Train-time and evaluation-time
pooling remain separate because their arithmetic implementations differ. Hook
registration installs steering before readout at the same block.

Training/evaluation extraction is recorded in `source_manifest.json`. The release
adds required path arguments, explicit legacy checkpoint loading, training
arguments/model identity in checkpoint metadata, and independently configurable
validation field names. The original single-file no-steering method is retained
in `s2d/ablation.py` with path and loading changes only.

## Protocol

1. Fix the base model identifier and revision, data version, disjoint splits,
   random seeds, steering position/layer, readout layers, and token pooling.
2. Train on the training split. Select thresholds on the validation split.
3. Evaluate once on the held-out test split, carrying validation thresholds over.
4. Save configuration, package versions, per-sample scores, and summary metrics.
5. Match each claimed paper result to its exact configuration and artifacts.

`configs/train_defaults.json` reflects source defaults, including seed 2025,
512 paired examples, steering layer 11, lambda 2, last 8 post-steer blocks,
last 25% of tokens, AdamW learning rate 0.001, 10 epochs, and batch size 8.
It does not establish which configuration produced a particular paper table.
The source optimizer uses PyTorch AdamW's default weight decay in addition to
any explicitly supplied `--l2_reg`; those are different mechanisms.

Checkpoint names derive from the input JSON basename, or the suffix after
`_llm_type_`. Sampling runs additionally use `_run_N`. Reusing an output directory
and the same input name overwrites its checkpoint. Test JSON basenames must be
unique within an evaluation run because outputs are keyed by basename.

## Scores and thresholds

The score is `(cos(rep, centroid_1) - cos(rep, centroid_0)) / temperature`.
It is called LLR in the original implementation, but is not a calibrated
probability or a demonstrated generative likelihood ratio.

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
Threshold routines and interpolation have deliberately not been replaced during
packaging. The original JSON writer may emit `Infinity` for a non-finite ROC
threshold; consumers requiring strict JSON should handle that explicitly.

## No-steering protocol

The archived ablation recomputes centroids from training representations without
steering. Its last-k-token pooling defaults to 64, while the main method uses a
ratio. `--centroid_normalize_per_sample` additionally changes centroid estimation.
Keep these settings explicit; this variant does not isolate the effect of steering
alone. Its legacy checkpoint is read for metadata rather than to inject a vector.

## Summarize completed runs

```bash
python scripts/summarize_results.py outputs/eval --output outputs/summary.csv
```

This exports per-test AUROC and validation-threshold operating metrics from each
`summary.json`. It does not average incompatible settings or infer paper tables.

## Remaining reproduction work

Paper-to-configuration mapping, actual benchmark preparation, GPU memory/runtime
measurement, multi-seed main-result reproduction, multi-GPU validation, and
release-checkpoint selection remain to be completed. Baseline references are in the README. Watermark generation/detection is included
with a separately documented protocol in `watermarking.md`; real-model watermark
results are not certified by this candidate.

## Model support

Block discovery recognizes `model.layers`, `transformer.h`, and
`model.decoder.layers`. Attention/MLP steering additionally requires `self_attn`
and `mlp` on the chosen block. This structural support does not guarantee every
model family works. Training retains `device_map="auto"`; evaluation uses one
device. Only tiny Llama on CPU has been exercised in the release tests.
