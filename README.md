# IRIS Paper LLM

Finds refereed papers that belong in the [ADS IRIS bibliography](https://ui.adsabs.harvard.edu/public-libraries/30bDOCvOTJiAgacWhJxkmA).
For one publication year it searches ADS, downloads the open-access PDFs, asks an OpenAI model whether each whole paper qualifies, and lists the qualifying papers that are not in the library yet.
Everything runs through the `iris-papers` command.

Created by Anthony R. Buonomo.

## What you need

- A terminal (macOS, Linux, or WSL on Windows), an internet connection and 1-2 GB of free disk space per publication year.
- A free ADS account, for the search.
  Its API token is at <https://scixplorer.org/user/settings/token> (ADS and SciX share accounts and tokens).
- An OpenAI account with a few dollars of API credit, for the classification.
  Its keys are at <https://platform.openai.com/api-keys>.
  A full year costs under $1.
  You don't need this key to search and download.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
   It also installs Python 3.14 or newer if needed.
2. Get the code, install it, and create your key file:

   ```bash
   git clone https://github.com/LM-SAL/iris-paper-finder.git
   cd iris-paper-finder
   uv sync
   cp -n .env.example .env
   ```

3. In `.env` (git ignores it), replace each `<...>` placeholder, brackets included, with the key itself: `ADS_TOKEN` and, when you are ready to classify, `OPENAI_API_KEY`.
4. Check the install with `uv run pytest`.
   It needs no keys or network and takes a second.

Each command checks the keys it needs as soon as it starts, so a missing or wrong key fails straight away, not after the downloads:

- `search` needs `ADS_TOKEN`.
- `download` needs no key.
- `classify` and `run` need both; `classify` needs `ADS_TOKEN` only to fetch the library (see below).
- `evaluate` needs `OPENAI_API_KEY`.

Run every command from this folder; the `data/` paths are relative to it.
Before each yearly run, update the code with `git pull` and `uv sync`.

## Finding the papers for one year

Run the previous year early in the new year (January or February), and run it again around July to pick up papers ADS indexed late.
The second run only classifies papers it has not seen, so it costs cents.

The work has four steps.
The examples use 2025.

### 1. Search ADS

```bash
uv run iris-papers search --year 2025
```

This writes the candidate papers (about 140 a year) with their open-access links to `data/metadata/2025.jsonl`.

### 2. Download the PDFs

```bash
uv run iris-papers download data/metadata/2025.jsonl
```

This saves each paper as `data/pdfs/2025/<bibcode>.pdf` and takes about 20 minutes, because requests are spaced 3 seconds apart.
Besides the ADS links it tries PDF addresses derived from Springer, Nature and Frontiers DOIs, the ADS scan, and the open copies (HAL and other repositories, arXiv, PubMed Central) that [OpenAlex](https://openalex.org) lists for the DOI.
It rejects a PDF whose first pages have no text or do not match the ADS abstract (ADS sometimes links a different paper with the same title).

It ends with one line of counts.
If some papers could not be downloaded (about 10%), it also warns `N papers have no PDF yet`; step 3 deals with them.

### 3. Get the failed downloads yourself

The papers that could not be downloaded are listed in `data/pdfs/2025/manual_downloads.jsonl`.
Open it in a text editor; each line is one paper, for example:

```json
{"bibcode": "2025A&A...697A.103F", "target": "data/pdfs/2025/2025A&A...697A.103F.pdf", "urls": ["https://www.aanda.org/10.1051/0004-6361/202451875/pdf"], "category": "blocked", "errors": ["HTTP 403 (bot protection: DataDome)"], "updated_at": "..."}
```

`category` says why it failed:

- `blocked`: the publisher's bot protection refuses scripts.
  This is typical for A&A, OUP, Wiley, AGU, Science and MDPI.
  Download the PDF in your browser.
- `transient`: a timeout or server error.
  Rerun step 2 later first.
- `permanent`: no usable open copy was found.
  Get the PDF through your institution's journal subscription or from the authors.

For each paper:

1. Open `https://ui.adsabs.harvard.edu/abs/<bibcode>` in a browser and get the PDF.
2. Save it at exactly the `target` path.
   In a terminal, quote the path, because bibcodes contain `&`: `mv ~/Downloads/paper.pdf 'data/pdfs/2025/2025A&A...697A.103F.pdf'`
3. When you have saved them all, rerun step 2.
   It checks each file you added and removes it from the list; papers already downloaded are skipped.

If a paper stays on the list after you saved it, open the saved file: it is probably a web page saved with a `.pdf` name, a scanned PDF without text, or a different paper.
Replace it with a real, text-based PDF of the right paper (the arXiv version works) at the same path.
A rejected file is kept as `<bibcode>.<sha256>.rejected`, so it is never classified; the list says why.

You can skip papers you cannot get.
They are not classified, but step 4 warns how many are missing and the report lists them with the reason and the path to save them at, so a curator can still check them.

### 4. Classify

```bash
uv run iris-papers classify data/pdfs/2025 --limit 5   # trial: about 2 minutes, a few cents
uv run iris-papers classify data/pdfs/2025
```

This first fetches the current ADS IRIS library, then sends each PDF's text to OpenAI, about 20 seconds per paper, and writes the report `data/results/2025.md`.
The trial checks that everything works; the second command continues with the remaining papers.
It ends with one line of counts for that command only, such as `"positive": 3`; the report has the totals.

Keep the computer awake during steps 2 and 4.
If a step stops, rerun the same command and it continues where it stopped.

### All steps at once

```bash
uv run iris-papers run --year 2025
```

`run` does steps 1, 2 and 4 in one go, in about an hour, and classifies whatever it could download.
For papers that failed to download, do step 3 and then rerun `run` or step 4; only the new papers are sent to OpenAI.

### Output files

| File | Contents |
|---|---|
| `data/results/2025.md` | the report: papers to review, papers without a PDF, failures, and every decision |
| `data/results/2025_to_review.txt` | ADS links of the papers to review |
| `data/results/2025.jsonl` | log of every classification attempt, with evidence |
| `data/ads_iris_library_<date>.txt` | the ADS IRIS library as fetched by step 4 |
| `data/metadata/2025.jsonl` | ADS candidates with their open-access links, DOIs and abstracts |
| `data/pdfs/2025/<bibcode>.pdf` | downloaded papers |
| `data/pdfs/2025/manual_downloads.jsonl` | papers that could not be downloaded |
| `data/pdfs/2025/download_attempts.jsonl` | log of every download attempt |

Every step is safe to rerun.
`search` refreshes the candidate list, `download` skips PDFs it already has, and `classify` skips papers already classified from the same PDF with the same prompt, model and reasoning effort; failed papers are retried.
Add `--force` to `classify` or `run` to reclassify everything.

`data/` is not in git.
Copy `data/results/<year>.*` somewhere safe when you finish.

## Reading the results

Each paper gets one decision:

- `YES`: it belongs in the library.
  Its `basis` tags say why (see below).
- `UNCERTAIN`: a curator should decide.
  This is a review that substantially discusses IRIS results (basis `REVIEW`), or a paper whose text was not enough to decide.
- `NO`: IRIS is only cited, mentioned, credited, or funding the work, or the acronym means something else.

The report starts with **To review**: every `YES` and `UNCERTAIN` paper that is not in the library, with its ADS link and the evidence (a page number and a reason for each tag).
The same links are in `<year>_to_review.txt`.
The library is the one fetched at the start of step 4; to compare against another list of bibcodes (one per line) instead, add `--library <file>` to `classify` or `run`.

Next, **No PDF, not classified** lists the candidates that could not be downloaded and were skipped; check them by hand, or add their PDFs (step 3) and rerun `classify`.

Page numbers count pages of the downloaded PDF, which is often the arXiv version, so they are not journal page numbers.
The model can be wrong: check the evidence before adding a paper.
`PROCESSING_FAILED` papers are listed under **Could not be classified** with the error; they are never counted as `NO` and are retried on the next run.

## What counts as an IRIS paper

The ADS IRIS library is the ground truth, and the classifier follows its current inclusion practice.
`YES` needs at least one of these tags:

- `OBSERVATIONAL_DATA`: the authors analyze or show IRIS observations, or products derived from them (IRIS^2 inversions, mosaics, catalogues, models trained on IRIS data), for the Sun or another star.
- `SYNTHETIC_OBSERVABLE`: the authors compute or analyze a synthetic observable inside an IRIS spectrograph window or slit-jaw channel, even if they never mention IRIS.
- `INSTRUMENT_OR_SOFTWARE`: the paper is about the IRIS instrument, calibration, operations, software, or data products.
- `COMPANION_PAPER`: the paper is part of a series whose companion analyzes IRIS data, and it models or extends those same observations.

On their own, these are `NO`: data availability, coincidental observations, citations, background, comparisons with published IRIS results, future-work suggestions, funding acknowledgements, IRIS lines observed only by other instruments, IRIS capabilities used to motivate another instrument, and reviews that only mention IRIS.
The IRIS coverage (FUV1 1331.7-1358.4 Å, FUV2 1389.0-1407.0 Å, NUV 2782.7-2835.1 Å; slit-jaw channels 1330, 1400, 2796, 2832 Å) and the full rules are in the prompt in [`classify.py`](src/iris_paper_llm/classify.py).

## Troubleshooting

Errors are printed after a `usage: iris-papers ...` line.
That line does not mean the command was mistyped; read the `error:` line after it.

| Error | What to do |
|---|---|
| `Missing credentials` or `Incorrect API key provided` | Check `OPENAI_API_KEY` in `.env`. A key exported in your shell overrides `.env`; remove it with `unset OPENAI_API_KEY` |
| `ADS_TOKEN is not set; add it to .env` | Set `ADS_TOKEN` in `.env` |
| `401 Client Error: UNAUTHORIZED` from `api.adsabs.harvard.edu` | The ADS token is wrong or expired; make a new one and update `.env` |
| `insufficient_quota` | Add credits to the OpenAI account |
| The model `does not exist` | The default model was retired; see Maintenance |
| `No such file or directory: 'data/...'` | Run the command from the project folder, and `search` before `download` |
| You need more detail | Add `LOG_LEVEL=DEBUG` before the command and save the output, e.g. `... 2> debug.log`; it includes the full paper text |

A missing or wrong key, or an unknown model, stops a command as soon as it starts.
Running out of OpenAI credit shows up only when the first paper is sent; it stops the run instead of recording a failure for each paper.

## Checking accuracy

`data/eval/reviewed_cases.jsonl` holds 16 labelled PDF cases covering 15 papers.
Run them after changing the prompt or the model; it costs a few cents:

```bash
uv run iris-papers evaluate --prepare-pdfs
```

`--prepare-pdfs` downloads the ten labelled PDFs that are not in the repository into `data/pdfs/reviewed/` and checks their SHA-256.
The report is written to `data/eval/classification_results.md`; the current default model gets all 16 decisions right.
The labels and the benchmark's limits are documented in [data/eval/README.md](data/eval/README.md).
Add `--model <name>` to test another model without changing anything.

## Maintenance

- **Model.** `DEFAULT_MODEL` and `DEFAULT_REASONING_EFFORT` in [`classify.py`](src/iris_paper_llm/classify.py) set the defaults.
  To switch model, first run the accuracy check with `--model <new-model>`, then change only the quoted name on the `DEFAULT_MODEL = "..."` line.
  Until you do, pass the same `--model` to every `run`, `classify` and `evaluate`.
  A new model or effort reclassifies every paper.
- **ADS query.** `iris_query` in [`ads.py`](src/iris_paper_llm/ads.py) combines the IRIS name, citations of the IRIS instrument paper, the `IRIS` acronym with IRIS line or slit-jaw terms, the IRIS data acknowledgement, and abstracts that synthesize IRIS-passband lines.
- **ADS to SciX.** SciX replaces the ADS interface from 2026-11-16 and keeps the same API, tokens and links, so nothing needs to change.
  If the API address ever moves, it is the single constant `ADS_API` in [`ads.py`](src/iris_paper_llm/ads.py).

## Development

Install the git hooks once with `uvx prek install` (or `prek install`).
They run ruff, ty, uv lock, codespell, zizmor and file checks on each commit; run them all with `uvx prek run --all-files`, and the tests with `uv run pytest`.

The tests use real open-access PDFs from `tests/data/pdfs/` (credited in its README) and a fake OpenAI client, so they need no network access and cost nothing.
GitHub Actions runs the hooks and the tests (on Ubuntu and macOS) on every push to `main` and every pull request.

A monthly live check (`.github/workflows/live-check.yml`, also runnable from the Actions tab) searches ADS with the repository secret `ADS_TOKEN` (Settings > Secrets and variables > Actions).
If the secret `OPENAI_API_KEY` is also set, it classifies two reviewed papers too, for about a cent a month; use a key from a separate OpenAI project with a small monthly budget.
GitHub pauses scheduled workflows after 60 days without repository activity; re-enable it in the Actions tab.
