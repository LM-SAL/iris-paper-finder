# Evaluation data

`reviewed_cases.jsonl` is the active 16-case corpus covering 15 papers.
Each case has a PDF location and checksum, the expected `expected_include` decision (`YES`, `NO`, or `UNCERTAIN`), the expected `expected_basis` tags, and an evidence note.
The labels follow the ADS IRIS library's inclusion practice described in the main README.
`iris-papers evaluate` reports include accuracy, exact basis matches, and one row per paper, scored against the labels currently in the cases file.

Six PDFs are in `tests/data/pdfs/`.
The other ten, not all of which may be redistributed, are fetched by `iris-papers evaluate --prepare-pdfs` from each case's `pdf_links` (arXiv, CORE or the publisher) into `data/pdfs/reviewed/`.
The two copies of the Ellerman-bomb paper use separate directories.
For `instrument_description_only` the reviewed PDF is the accepted manuscript, because Wiley stamps every download of the published PDF.

The three cases added on 2026-09-29 cover the previously untested positive `INSTRUMENT_OR_SOFTWARE` and `COMPANION_PAPER` tags, plus the preprint variant of `mention_previous_iris_work` that produced a false positive in a local v4.1 run.
The new positive labels were checked against the source text by Codex, not independently adjudicated by curators; `label_source` and `evidence_note` record that provenance and the relevant PDF pages.
Sources: [IRIS mission paper](https://arxiv.org/abs/1401.2491v1), [companion modeling paper](https://arxiv.org/abs/1905.03749v1), and [Ellerman-bomb preprint](https://arxiv.org/abs/1506.04426v1).
These targeted cases are a regression check, not a representative accuracy estimate.

## History

The default model (`gpt-5.6-luna`, effort `medium`) was chosen on 2026-09-28 by comparing five models on 223 papers against mixed reference labels; the comparison showed it was cheap and among the most accurate, not a proven winner.
Prompt iris-v4.2 (2026-09-30) tightened the `INSTRUMENT_OR_SOFTWARE`, `COMPANION_PAPER` and review rules.
The raw outputs, adjudications and scorer are not kept in the repository; they are in git history at commit `dc8df4d` (`data/eval/model_comparison_2026-09-28/`), together with the 2026-01-07 library snapshot that was the ground truth then.

Known remaining error: one paper was tagged `SYNTHETIC_OBSERVABLE` for O IV 279.93 Å, an EUV line, confusing Å with nm (the NUV window is 2782.7-2835.1 Å).
