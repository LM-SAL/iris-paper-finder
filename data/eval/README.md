# Evaluation data

`reviewed_cases.jsonl` is the active 13-paper gold corpus. Each case has a PDF
location and checksum, the expected `expected_include` decision (`YES`, `NO`,
or `UNCERTAIN`), the expected `expected_basis` tags, and an evidence note. The
labels follow the ADS IRIS library's inclusion practice described in the main
README. `iris-papers evaluate` reports include accuracy, exact basis matches,
and one row per paper, scored against the labels currently in the cases file.

Six PDFs are in `tests/data/pdfs/`. The other seven, not all of which may be
redistributed, are fetched by `iris-papers evaluate --prepare-pdfs` from each
case's `pdf_links` (arXiv, CORE or the publisher) into
`data/pdfs/reviewed/<bibcode>.pdf`. For `instrument_description_only` that is
the accepted manuscript, because Wiley stamps every download of the published
PDF.

`ads_iris_library_2026-01-07.txt` preserves the ADS IRIS library snapshot used
as ground truth. Its `2025NatAs.tmp..221A` entry is now `2026NatAs..10...54A`.

## Model choice (2026-09-28)

Five models were run with the whole-paper pipeline at reasoning effort
`medium` on 223 papers: the 13 gold cases, 100 year-stratified library papers,
and 110 non-library ADS candidates from 2022–2024. Every paper on which the
models disagreed (30) was judged blind by two independent reviewers applying
the prompt's rules; they agreed on 29. An error is a decision that differs
from that judgement, or from the unanimous answer where the models agreed.

| Model | Errors, prompt v4.0 | Errors, prompt v4.1 | USD per 1000 papers |
|---|---:|---:|---:|
| `gpt-5.6-luna` | 6 | 3 | 4 |
| `gpt-6-luna` | 5 | 5 | 2 |
| `claude-sonnet-5-5` | 5 | – | 54 |
| `gpt-5-mini-2025-08-07` | 6 | – | 6 |
| `gpt-5.4-mini-2026-03-17` | 8 | – | 14 |

With v4.1 both luna models find every library paper that meets the policy
and every gold label; their remaining errors are extra `YES` or `UNCERTAIN`
papers that a curator review catches. `gpt-5.6-luna` became the default. The
judges also confirmed 8 of the 110 non-library candidates as IRIS papers
missing from the library, plus 6 reviews for a curator to decide.
