# Data format and preparation

Each split is a UTF-8 JSON array of paired records:

```json
[
  {
    "human_text": "A human-written text.",
    "direct_prompt": "A model-generated text."
  }
]
```

Every valid pair expands to two samples: generated text first (label 1), human
text second (label 0). Pairs with missing, empty, or non-string fields are skipped
as in the research implementation. Input should contain JSON objects, and every
split must have at least one valid pair. Train, validation, and test records must
be disjoint. Do not use the example fixtures to report detection performance.

Training uses `--llm_field` and `--human_field` (defaults `direct_prompt` and
`human_text`), then stores those names in the checkpoint. Main evaluation uses
the checkpoint's fields for test data unless overridden. Validation defaults
remain `direct_prompt` and `human_text`, independently of test fields, to preserve
the original attack-evaluation protocol. Use `--valid_llm_field` and
`--valid_human_field` for a different validation schema. The original no-steering
ablation uses its resolved field names for all splits.

The research workspace contains DetectRL and RAID data. This release does not
redistribute those files. Acquire the intended benchmark from its original
maintainers and use its permitted preprocessing and split protocol. The exact
upstream versions, download links, preprocessing scripts, split manifests, and
checksums for the paper are still release tasks; do not infer them from filenames.

`--max_train_pairs` uses the leading pairs, not a fresh random split. The original
`--bootstrap_samples` option samples pairs **without replacement**, despite its
name; `--bootstrap_runs` repeats that sampling. This behavior is preserved.
