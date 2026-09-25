# Steer-to-Detect (S2D)

**Steer-to-Detect: Probing Hidden Representations for Detection of LLM-Generated Texts**

S2D learns a steering vector on a frozen language model and detects generated
text using its hidden representations. This repository includes training,
evaluation, a no-steering ablation, and Gumbel/KGW/SynthID watermark experiments.

## Installation

Python 3.11 is the locally tested version. Install a PyTorch build appropriate
for your hardware, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start with the [offline quick start](docs/quickstart.md) to run a tiny model
without downloading weights. Example data is synthetic and only tests the workflow.

## Train and evaluate

Prepare separate training, validation, and test files in the
[paired JSON format](docs/data.md). Supply a local model directory or a
Hugging Face model ID:

```bash
python scripts/run_config.py train configs/train_defaults.json \
  --model_name_or_path MODEL_ID_OR_DIRECTORY \
  --train_data_path data/train.json \
  --out_dir outputs/train

python evaluate.py \
  --model_name_or_path MODEL_ID_OR_DIRECTORY \
  --ckpt_path outputs/train/s2d_ckpt_train.pt \
  --valid_data_path data/valid.json \
  --test_data_paths data/test.json \
  --out_dir outputs/eval
```

Defaults use steering layer 11 (zero-based) and bfloat16. Small models need a lower
layer; CPU runs should use `--model_dtype float32` for training and `--dtype float32`
for evaluation. Run either entry point with `--help` for all options.

Evaluation writes per-sample scores and `summary.json`. Higher scores indicate
model-generated text (`1`); human text is `0`. Scores are not calibrated
probabilities. See [reproduction notes](docs/reproduction.md) for threshold rules,
model support, and the no-steering ablation's different pooling protocol.

## Watermark experiments

The watermark module supports **Gumbel-noise, KGW, and a SynthID uniform-g research
variant**, with generation, native detection, and export to the S2D data format:

```bash
python -m watermarking.generate --help
python -m watermarking.evaluate --help
python -m watermarking.export_pairs --help
```

See the [watermark guide](docs/watermarking.md) for complete commands and protocol
changes from the historical scripts. Native scores use continuation tokens only.

## Baselines

We use the following comparison methods in our research experiments. Their code
is not bundled; references and local adaptation notes are provided here.

| Baseline | Reference implementation or paper |
| --- | --- |
| Likelihood | [Fast-DetectGPT baseline collection](https://github.com/baoguangsheng/fast-detect-gpt) |
| LogRank | [Fast-DetectGPT baseline collection](https://github.com/baoguangsheng/fast-detect-gpt) |
| Fast-DetectGPT | [Official implementation](https://github.com/baoguangsheng/fast-detect-gpt) |
| Binoculars | [Official implementation](https://github.com/ahans30/Binoculars) |
| RADAR | [Official implementation](https://github.com/IBM/RADAR) |
| RAIDAR | [Raidar: geneRative AI Detection viA Rewriting](https://arxiv.org/abs/2401.12970) |
| RoBERTa-based supervised detector | [RoBERTa](https://arxiv.org/abs/1907.11692) |
| ImBD (Imitate Before Detect) | [Paper](https://arxiv.org/abs/2412.10432) |
| AdaDetectGPT | [Official implementation](https://github.com/Mamba413/AdaDetectGPT) |
| L2D (Learn-to-Distance) | [Official implementation](https://github.com/Mamba413/L2D) |
| RepreGuard | [Official implementation](https://github.com/Chen-X666/RepreGuard) |


See [baseline notes](docs/baselines.md) for implementation details: the local
LogRank scorer is not LRR, and the RAIDAR-named adapter uses AdaDist components.

## Repository layout

| Path | Purpose |
| --- | --- |
| `train.py`, `evaluate.py`, `evaluate_no_steer.py` | S2D command-line entry points |
| `s2d/` | Training, steering, readout, data, and metrics |
| `watermarking/` | Watermark generation, detection, and S2D export |
| `configs/`, `examples/` | Defaults and synthetic examples |
| `scripts/`, `tests/` | Utilities and offline tests |
| `docs/` | Usage, protocols, provenance, and validation |

## Tests and scope

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

CPU tiny-model tests cover S2D and all three watermark workflows. Real-model GPU
results, multi-GPU execution, and full paper reproduction remain unverified.
Datasets, base model weights, and trained release checkpoints are not included.
See [validation](docs/validation.md) and [release status](docs/release_status.md).

Research checkpoints contain NumPy arrays and use `weights_only=False`; load
only trusted checkpoints and use the same base model as training.

## Citation and license

Paper link, author list, and citation metadata are pending.
Original project code is released under the [MIT License](LICENSE). The
MarkLLM-derived SynthID core retains its Apache-2.0 license; see
[third-party notices](THIRD_PARTY_NOTICES.md).
