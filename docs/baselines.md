# Baselines and research-adapter provenance

The README lists the comparison methods used in the research experiments. Their
implementations are intentionally not redistributed here. The table below records
the evidence used to identify those names from the research workspace; filenames
refer to that workspace, not files included in this release.

| Method | Research source | Interpretation |
| --- | --- | --- |
| Likelihood | `baselines/likelihood_eval.py` | Likelihood-based scoring |
| LogRank | `baselines/logrank_eval.py` | Negative mean log-rank; not LRR despite its output directory name |
| Fast-DetectGPT | `baselines/fastdet_eval.py` | Conditional probability-curvature baseline |
| Binoculars | `baselines/binoculars_eval.py` | Binoculars evaluation adapter |
| RADAR | `baselines/radar_eval.py` | RADAR evaluation adapter |
| RAIDAR | `baselines/raidar_train_eval.py` | Local rewriting-based adapter using AdaDist training/evaluation; do not describe it as an unchanged official implementation |
| RoBERTa | `baselines/roberta_train_eval.py` | Locally trained supervised detector; model/backbone configuration must be specified |
| ImBD | `baselines/imbd_train.py`, `imbd_eval.py`, `spo.py` | ImBD/SPO adaptation; the filename spelling is `imbd`, the method is ImBD |
| AdaDetectGPT | `baselines/adadetect_train.py`, `adadetect_eval.py` | Paired-data adaptation of AdaDetectGPT |
| L2D | `baselines/l2d_train_eval.py`, `AdaDist/` | Learn-to-Distance adaptation; AdaDist is an implementation component here, not automatically an additional baseline |
| RepreGuard | `baselines/repreGuard_detector.py`, `repreGuard_eval.py` | Representation-based detector and evaluation adapter |

The inventory does not establish that every method appears in every paper table.
Before releasing a full reproduction claim, record each method's actual source
revision, scoring backbone/checkpoint, training data and count, preprocessing,
random seeds, validation protocol, score direction, and aggregation rule. The
current README deliberately reports no unverified baseline performance numbers.

Method references are linked in the [README](../README.md#baselines).
Watermark experiment provenance is documented separately in
[watermarking.md](watermarking.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
