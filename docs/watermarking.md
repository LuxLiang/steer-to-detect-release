# Watermark experiments

This release includes generation, native watermark detection, and export for S2D
for three methods present in the research workspace:

| CLI name | Implementation | Native score |
| --- | --- | --- |
| `gumbel` | Research Gumbel-noise sampling variant, previous-token SHA-256 seed | Sum of `-log(1-u)` |
| `kgw` | Green-list logit bias, previous-token keyed permutation | Green-token z-score |
| `synthid` | MarkLLM-derived research uniform-g variant, 30 keys by default | Mean g-value over continuation ngrams |

These are research adaptations. In particular, `gumbel` adds Gumbel noise and
then uses multinomial sampling, matching the original driver's approach; it is
not an exact Gumbel-max argmax sampler. `synthid` retains the local **uniform-g**
extension and is not an interchangeable implementation of every SynthID variant.
The native detector is a watermark/key detector; S2D is a human/AI detector.
Their tasks differ for unwatermarked machine text.

## Offline smoke example

Run from the repository root after installing the main requirements. No additional
watermark dependency or remote model download is required for this example.

```bash
python scripts/create_tiny_model.py outputs/tiny-model

python -m watermarking.generate \
  --algorithm kgw \
  --model_name_or_path outputs/tiny-model \
  --data_path examples/test.json \
  --output outputs/kgw/dataset.json \
  --prompt_len 4 --gen_len 8 --max_samples 2 \
  --top_k 8 --device cpu --dtype float32

python -m watermarking.evaluate \
  --dataset outputs/kgw/dataset.json \
  --output outputs/kgw/metrics.json --lengths 5 8

python -m watermarking.export_pairs \
  --dataset outputs/kgw/dataset.json \
  --output outputs/kgw/test.json
```

Use `--algorithm gumbel` or `--algorithm synthid` with distinct output paths to
run the other methods. The random tiny model verifies execution only; detection
quality on these fixtures is not meaningful. The tiny model has fewer than 50
vocabulary entries, so use `--top_k 8` as shown.

## Generation with a real model

```bash
python -m watermarking.generate \
  --algorithm synthid \
  --model_name_or_path MODEL_ID_OR_DIRECTORY \
  --data_path data/test.json \
  --output outputs/synthid/test-dataset.json \
  --prompt_len 50 --gen_len 256 --max_samples 1000 \
  --temperature 0.7 --top_k 50 --seed 2025

python -m watermarking.evaluate \
  --dataset outputs/synthid/test-dataset.json \
  --output outputs/synthid/test-metrics.json \
  --lengths 64 128 256
```

Input is a JSON array containing `human_text` (or `--human_field`). Each eligible
record must contain at least `prompt_len + gen_len` tokens. The generator uses
the exact initial token IDs as the shared prompt and records the following human
continuation alongside matched watermarked and unwatermarked generations. Both
model generations have the same fixed continuation length and sampling settings;
SynthID applies temperature and top-k inside its processor, so a second external
warper is disabled. Generation is sequential, one record at a time, on one device.

Default CPU precision is float32; CUDA defaults to bfloat16. Use `--dtype` for an
explicit choice. Model loading uses a single device; multi-GPU loading is not
provided. `--key` controls Gumbel/KGW seeds, and `--gamma` / `--delta` control KGW.
SynthID uses the entire supplied `--synthid_config`, copied into output metadata.
Existing output files are refused unless `--overwrite` is explicitly supplied.

The output contains `schema_version`, generation `metadata`, and `records`.
Each record stores prompt length, exact human/watermarked/unwatermarked token IDs,
and decoded continuation text. There is no decode/re-tokenize round trip in the
native detector. Native evaluation loads no language model or tokenizer.

## Evaluate S2D on watermarked and unwatermarked text

Export the test continuations to the standard paired format:

```bash
python -m watermarking.export_pairs \
  --dataset outputs/synthid/test-dataset.json \
  --output outputs/synthid/test-watermarked.json

python -m watermarking.export_pairs \
  --dataset outputs/synthid/test-dataset.json \
  --llm_source unwatermarked \
  --output outputs/synthid/test-unwatermarked.json

python evaluate.py \
  --model_name_or_path S2D_BASE_MODEL \
  --ckpt_path outputs/train/s2d_ckpt_train.pt \
  --valid_data_path data/valid.json \
  --test_data_paths outputs/synthid/test-watermarked.json,outputs/synthid/test-unwatermarked.json \
  --out_dir outputs/synthid/s2d-eval
```

The export maps the chosen model continuation to `direct_prompt` and the human
continuation to `human_text`. Empty decoded text pairs are counted and skipped.
Use a separate validation split for threshold selection and never reuse the
watermark test records for training/calibration. Specify whether S2D was trained
on ordinary text or on a separately generated watermarked training split.
Different generation and S2D base models are allowed: S2D tokenizes the exported
text using its own base model, so its token lengths need not equal native-detector
lengths. The S2D evaluation API and threshold rules are unchanged.

## Detection windows and metrics

All native scores use **continuation tokens only**. The prompt provides context
for the first Gumbel/KGW token but does not itself contribute to the score. Native
evaluation reports two comparisons, watermarked versus human and watermarked
versus unwatermarked, for each requested continuation length. Both sides of a
record must meet that length. Per-record scores, eligible-pair counts, skipped
counts, AUROC, and empirical ROC-interpolated TPR are saved as JSON.

For SynthID, the original logits processor starts with a zero context. We score
only ngrams entirely inside the continuation, omitting its first `ngram_len - 1`
tokens. At length L the default configuration therefore scores L-4 ngrams. The
native score is the unmasked mean g-value, following the research pivotal-score
path; it does not apply the separate MarkLLM repeated-context/EOS detector masks.
Lengths shorter than `ngram_len` are rejected.

TPR here is interpolated from the evaluated split's ROC. It is not a calibrated
threshold with guaranteed false-positive control. Raw native scores should not
be interpreted as p-values without additional assumptions and validation.

Random tables/permutations depend on PyTorch and device type. Metadata records
PyTorch version and CPU/CUDA type; detection requires matching device type. Use
the same PyTorch version for reproducibility. CUDA-generated artifacts cannot be
assumed to have matching scores on CPU. Generation seed and watermark keys are
experimental parameters, not authentication secrets.

## Differences from historical scripts

This is a documented, consistent release protocol, **not a promise of bitwise
reproduction of historical watermark tables**:

- Removed hard-coded local paths, working-directory imports, and implicit cache reuse.
- Use exact token IDs and a single shared prompt; no extra prompt special tokens
  are introduced by re-tokenization.
- Exclude prompts consistently. The old KGW detector included the prompt window.
- Store full token IDs for all three sources, vocabulary size, device, seed, and config.
- Create a new SynthID processor for each independent generation, avoiding state
  carry-over between records; avoid the old fixed-50-token wrapper.
- Slice SynthID text into continuation-only text before S2D export; the old wrapper
  returned a full decoded sequence.
- Require enough human tokens and fixed model continuation lengths for matched
  comparisons. The old drivers used different early-stop/filtering rules.
- Respect explicit top-k for all methods; historical Gumbel relied on model defaults.
- Include the mean-g SynthID detector, not the old suite of six pivotal transforms,
  Bayesian detector training, visualization framework, or pickle caches.

Checksums identify the original inputs in `watermark_source_manifest.json`.
Third-party provenance and license text are in [the notices](../THIRD_PARTY_NOTICES.md).
GPU behavior and real-model paper results still require validation.
