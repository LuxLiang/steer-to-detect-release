# Third-party notices

## MarkLLM-derived SynthID core

`watermarking/vendor/synthid_core.py` contains the `SynthIDUtils`, `SynthIDState`,
and `SynthIDLogitsProcessor` classes extracted from the research workspace's
adaptation of [THU-BPM/MarkLLM](https://github.com/THU-BPM/MarkLLM).

Original notice: Copyright 2024 THU-BPM MarkLLM.
Licensed under the Apache License, Version 2.0; the license text is included in
[watermarking/vendor/LICENSE-APACHE-2.0](watermarking/vendor/LICENSE-APACHE-2.0).
The source copyright/license header is retained. The local source checksum is
recorded in [the watermark source manifest](docs/watermark_source_manifest.json).

Changes made for this release on 2026-09-25:

- Extracted three classes and their necessary imports into a self-contained module.
- Removed imports of the MarkLLM auto-loader, model/config wrappers, visualization,
  and Bayesian detector modules that are not used by these classes.
- Retained the extracted class bodies, including the research uniform-g extension.
- Added adapters outside the vendored file for explicit configuration, per-sequence
  state initialization, stored-token detection, and portable command-line use.

The research source already differs from upstream MarkLLM. This is not a claim
that the local uniform-g variant or its experimental results are supplied or
endorsed by MarkLLM or the original SynthID authors. Exact upstream revision of
the research adaptation was not available in the source copy.

The configuration in `watermarking/configs/synthid.json` is copied from that
research workspace and contains public experimental seed/key values.

The project's original code is licensed under [MIT](LICENSE). The vendored
MarkLLM-derived code remains under Apache-2.0; the MIT license does not replace
that third-party license.
