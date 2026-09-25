# Local validation

Validated on 2026-09-25 using the existing Python 3.11 environment on CPU. CUDA
was unavailable to PyTorch. This was not a fresh dependency installation.

| Package | Installed version |
| --- | --- |
| torch | 2.11.0+cu130 |
| transformers | 4.51.3 |
| numpy | 1.26.4 |
| scikit-learn | 1.8.0 |
| accelerate | 1.14.0 |
| tokenizers | 0.21.4 |

## Tests

`python -m pytest -q` covers:

- Last-ratio pooling with left/right padding and exclusion of padded positions.
- Legacy quantile-threshold behavior with tied human scores.
- Offline generation of a random two-layer Llama and a local tokenizer.
- One training epoch on four synthetic paired records through the JSON config CLI.
- Nonzero, finite steering-vector update and checkpoint metadata.
- Reloading the checkpoint and two exactly matching evaluation runs.
- CSV export of held-out operating-point metrics.
- The original fixed-token no-steering ablation on the same local model.

## Original-script regression

An additional comparison used the original `s2d_train.py` and `s2d_eval.py` files
identified by the source manifest. It independently ran original and packaged
training with identical seed, tiny model, and data, then compared evaluation
using the same checkpoint:

- Steering-vector parameters: exact equality (zero numerical tolerance).
- Class centroids: exact array equality.
- Per-example test scores: identical JSON values.
- Full validation/test summary: identical JSON values.

The comparison explicitly sets `weights_only=False` for original-script checkpoint
loading so their NumPy-containing checkpoints work with the installed PyTorch.
It does not alter the original files or their numerical method.

To repeat with a trusted reference checkout:

```bash
python scripts/compare_reference.py --reference-dir /path/to/original/llm_det
```

## Limits

These checks establish CPU smoke-test behavior and equivalence on a tiny Llama;
they do not establish real-model accuracy, GPU/multi-GPU compatibility, runtime,
peak memory, or agreement with paper tables. Full benchmark datasets and released
checkpoints are not included. Dependency ranges have not been tested as a matrix.

## Watermark extension validation

The watermark suite adds five tests covering per-row keyed processors, independent
SynthID state and generation/detection ngram alignment, and one offline integration
run for each of Gumbel, KGW, and SynthID. Each integration run generates matched
human/watermarked/unwatermarked records, scores two continuation lengths, exports
S2D pairs, and checks them with S2D's data reader. RNG-device mismatch is rejected.
The copied SynthID class bodies were also compared by AST to the source and match
exactly. Gumbel and KGW processors matched their original single-sequence CPU
outputs exactly on a fixed numerical input. Adapter and protocol differences are listed in `watermarking.md`.

No real-model watermark accuracy, GPU behavior, or historical table equivalence
is claimed by these CPU smoke tests.
