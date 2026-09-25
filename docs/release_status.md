# Release status

This is a research code release candidate. Validation scope and remaining work
are recorded below.

Prepared:

- Modular S2D training and evaluation with portable entry points.
- Original no-steering ablation with explicit protocol documentation.
- Gumbel/KGW/SynthID watermark generation, native detection, and S2D data export.
- README baseline references, research-adapter notes, and third-party notices.
- MIT license for original code; Apache-2.0 retained for the SynthID dependency.
- Minimal direct dependency list without local package-build paths.
- Synthetic train/validation/test examples and an offline tiny-model generator.
- Original training defaults, config runner, and CSV result export.
- Source checksums and behavior-change notes.
- Offline tests and reference-comparison results in `validation.md`.

Remaining release work:

- Confirm paper title, authors, affiliations if desired, paper URL,
  and preferred citation. Add a populated `CITATION.cff`.
- Map paper tables to exact data splits, model revisions, configurations, and seeds.
- Verify benchmark redistribution/preparation permissions and add upstream links.
- Run clean-environment installation and at least one paper-scale GPU experiment;
  record runtime, peak memory, and reproduced metrics.
- Select checkpoint artifacts and hosting, if releasing trained checkpoints.
- Validate real-model GPU watermark experiments and reconcile the documented
  continuation-only protocol with any historical paper tables.
- Baselines are documented by reference, not bundled; record exact versions and
  experimental settings. Multi-GPU and rebuttal implementations remain outside scope.

The first candidate deliberately contains core method code and watermark adapters rather than the full
research workspace. Original datasets, model weights, experiment logs, and reports
have not been copied. The source scripts themselves are identified by SHA-256 in
`source_manifest.json`; no historical Git metadata has been imported.
