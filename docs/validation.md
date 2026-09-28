# Local validation

Recorded on 2026-09-25 with Python 3.11 and tiny Llama on CPU.

| Package | Version |
| --- | --- |
| torch | 2.11.0+cu130 |
| transformers | 4.51.3 |
| numpy | 1.26.4 |
| scikit-learn | 1.8.0 |
| accelerate | 1.14.0 |
| tokenizers | 0.21.4 |

`python -m pytest -q` covers pooling, thresholds, training, checkpoint reload,
evaluation, CSV export, no-steering ablation, and all three watermark workflows.
Watermark checks include processor state, token alignment, and S2D export.

Reference-script checks produced identical steering vectors, centroids, scores,
and summary metrics on the tiny model. To repeat with a trusted reference checkout:

```bash
python scripts/compare_reference.py --reference-dir /path/to/original/llm_det
```

The SynthID class bodies matched the source by AST comparison. Gumbel and KGW
processors matched source outputs on fixed CPU inputs.
