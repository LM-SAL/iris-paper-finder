# Evaluation data

`reviewed_cases.jsonl` is the active 16-case corpus covering 15 papers. Each case has a PDF
location and checksum, the expected `expected_include` decision (`YES`, `NO`,
or `UNCERTAIN`), the expected `expected_basis` tags, and an evidence note. The
labels follow the ADS IRIS library's inclusion practice described in the main
README. `iris-papers evaluate` reports include accuracy, exact basis matches,
and one row per paper, scored against the labels currently in the cases file.

Six PDFs are in `tests/data/pdfs/`. The other ten, not all of which may be
redistributed, are fetched by `iris-papers evaluate --prepare-pdfs` from each
case's `pdf_links` (arXiv, CORE or the publisher) into
`data/pdfs/reviewed/`. The two copies of the Ellerman-bomb paper use separate
directories. For `instrument_description_only` the reviewed PDF is
the accepted manuscript, because Wiley stamps every download of the published
PDF.

The three cases added on 2026-09-29 cover the previously untested positive
`INSTRUMENT_OR_SOFTWARE` and `COMPANION_PAPER` tags, plus the preprint variant
of `mention_previous_iris_work` that produced a false positive in a local
v4.1 run. The new positive labels were checked against the source text by
Codex, not independently adjudicated by curators; `label_source` and
`evidence_note` record that provenance and the relevant PDF pages. Sources:
[IRIS mission paper](https://arxiv.org/abs/1401.2491v1),
[companion modeling paper](https://arxiv.org/abs/1905.03749v1), and
[Ellerman-bomb preprint](https://arxiv.org/abs/1506.04426v1).
These targeted cases are a regression check, not a representative accuracy
estimate. A live run on 2026-09-29 (`gpt-5.6-luna`, effort `medium`, prompt
iris-v4.1) got all 16 include decisions right and 15 of 16 basis sets; it
missed `SYNTHETIC_OBSERVABLE` on `observational_iris_mosaic`. The preprint
case came back `NO` here, although one earlier sweep run returned
`YES (COMPANION_PAPER)`, so single runs vary.

`ads_iris_library_2026-01-07.txt` preserves the ADS IRIS library snapshot used
as ground truth. Its `2025NatAs.tmp..221A` entry is now `2026NatAs..10...54A`.

## Prompt iris-v4.2 (2026-09-30)

v4.2 tightens three rules: `INSTRUMENT_OR_SOFTWARE` now requires IRIS itself
as the subject (papers that only use the IRIS-released Bifrost simulation,
other missions' instrument papers and general-purpose software no longer
qualify, matching the library); a numbered series paper is a
`COMPANION_PAPER` only if it models or extends the companion's IRIS
observations; and "substantially discusses IRIS" in a review means a
section, figure or sustained discussion. With `gpt-5.6-luna` at effort
`medium`:

- gold corpus: all 16 include decisions right, 15 of 16 basis sets;
- archived comparison set: 4 disagreements with the 221 reference labels
  (v4.1: 3), 2 of them on the 29 adjudicated papers (v4.1: 3). Three of the
  four flipped between two v4.2 runs with near-identical wording, so the
  difference is run-to-run variation on borderline papers;
- the 123 sweep papers that v4.1 flagged, compared with an independent
  `claude-opus-5-5` reading (itself using v4.1 rules): agreement rose from
  84 to 98 of 122. All four users of the Bifrost release, the SPICE
  instrument paper and the Ellerman-bomb preprint are now `NO`.

Known remaining error: one paper was tagged `SYNTHETIC_OBSERVABLE` for O IV
279.93 Å, an EUV line, confusing Å with nm (the NUV window is 2782.7-2835.1 Å).

## Archived model comparison (2026-09-28)

Five models were run with the whole-paper pipeline at reasoning effort
`medium` on 223 papers: the 13 gold cases, 100 year-stratified library papers,
and 110 non-library ADS candidates from 2022–2024. The archived review queue
contains 30 papers selected for model disagreement or disagreement with
library membership. There are two recorded judgments per queued paper,
agreeing on 29; reviewer identities are not recorded in the verdict file.

The original inputs, raw predictions (including evidence, prompt hashes and
token counts), and verdicts are preserved in
[`model_comparison_2026-09-28/`](model_comparison_2026-09-28/). Reproduce the
scores without keys, PDFs or network access:

```bash
uv run python data/eval/model_comparison_2026-09-28/score.py
```

`jobs.jsonl` preserves all 223 original input paths and checksums. The
`*.v4.0.jsonl` files concatenate each model's original gold/library and
non-library outputs; `*.v4.1.jsonl`, `adjudication.json` and `truth.json` are
unchanged copies from the scratch evaluation. Some original gold paths now
live under `data/pdfs/reviewed/`; the historical Wiley PDF also differs from
the accepted manuscript now used by `instrument_description_only`.

`reference_labels.jsonl` makes the source of each reference decision explicit:
29 adjudicated, 11 other gold cases, 182 model-consensus labels, and one
unresolved reviewer split. Two of the 29 adjudicated cases are also gold
cases. The unresolved split is excluded from scoring. The known wrong-paper
PDF `2022Ap&SS.367..121S` is additionally excluded from the corrected scores:
its saved input checksum is retained so the historical result remains
auditable. This leaves 221 usable inputs, including 181 consensus labels.

An "error" in the table is disagreement with these mixed reference labels,
not independently measured classification error. The scorer also reports
the 29 adjudicated cases and the original 13 gold cases separately. No
before/after comparison against the old retrieval pipeline is available.

| Model | Disagreements / 221, v4.0 | Disagreements / 221, v4.1 | Historical USD per 1000 papers |
|---|---:|---:|---:|
| `gpt-5.6-luna` | 6 | 3 | 4 |
| `gpt-6-luna` | 5 | 5 | 2 |
| `claude-sonnet-5-5` | 5 | – | 54 |
| `gpt-5-mini-2025-08-07` | 6 | – | 6 |
| `gpt-5.4-mini-2026-03-17` | 8 | – | 14 |

The known wrong-paper input was unanimously `NO`, so excluding it changes
the denominator from 222 to 221 without changing the disagreement counts.
Both v4.1 runs matched all 13 original gold decisions. On the 29 adjudicated
cases, `gpt-5.6-luna` disagreed on 3 and `gpt-6-luna` on 2; the aggregate
ranking also depends on consensus labels. The recorded judgments include
8 non-library `YES` papers and 6 `UNCERTAIN` reviews.

`gpt-5.6-luna` remains the historical default, not a proven accuracy winner.
The cost column records the original estimates using OpenAI Flex and
Anthropic Batch discounts; the normal CLI does not request Flex. It is not
a current pricing quote or an estimate for the expanded corpus.
