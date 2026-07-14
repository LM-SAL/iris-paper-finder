# IRIS Paper LLM

Find solar-physics papers through ADS, download their PDFs, select relevant
passages locally, and ask the OpenAI API for a schema-validated IRIS-use
classification. The only interface is the `iris-papers` command.

Created by Anthony R. Buonomo.

## Install and classify one paper

Python 3.13 or newer is required.

```bash
uv sync
uv run python scripts/setup_onnx.py
```

Set the API key and classify a local PDF:

```bash
export OPENAI_API_KEY=your-key
uv run iris-papers classify paper.pdf --output data/results/one-paper.jsonl
```

The command first extracts page-aware text, finds deterministic IRIS/passband
matches, and uses the local ONNX model to rank additional passages. Only the
selected labeled passages are sent to OpenAI. Results are appended immediately
to JSONL, and rerunning the same command skips a matching successful result.
A Markdown report is written beside the JSONL file.

## Search and process a publication year

Set an [ADS API token](https://ui.adsabs.harvard.edu/user/settings/token):

```bash
export ADS_TOKEN=your-token
```

Run each durable stage separately:

```bash
uv run iris-papers search --year 2025
uv run iris-papers download data/metadata/2025.jsonl
uv run iris-papers classify data/pdfs/2025 --output data/results/2025.jsonl
```

Or compose the same three functions plus report generation:

```bash
uv run iris-papers run --year 2025
```

The default files are:

- `data/metadata/2025.jsonl`: ADS bibcodes and open-access link metadata;
- `data/pdfs/2025/`: validated PDF cache, attempt history, and manual queue;
- `data/results/2025.jsonl`: checkpointed structured classifications; and
- `data/results/2025.md`: current summary report.

The final JSON printed by each command includes the relevant downloaded,
manual-download, positive, negative, uncertain, failed, and skipped counts.
Use `--limit` for a small trial. `--force` intentionally reruns ADS search or
successful classifications; it does not make the downloader overwrite a valid
existing PDF.

## Reviewed evaluation corpus

The checksum-bearing manifest contains 13 manually reviewed cases:

```bash
uv run iris-papers evaluate data/eval/reviewed_cases.jsonl \
  --output data/eval/classification_results.jsonl --prepare-pdfs
```

`--prepare-pdfs` acquires the three ignored publisher PDFs from explicit
open-access sources and verifies their frozen SHA-256 checksums. Publisher PDFs
remain local and are not committed. The other ten PDFs are tracked test data.
Evaluation uses the same extraction, retrieval, classification, checkpoint,
and report functions as ordinary runs. Rerun the same command after an
interruption to skip matching successful records and continue from the JSONL
checkpoint. This gold-corpus evaluation is separate from the offline developer
checks: it uses real PDFs and ONNX, and new classifications call OpenAI.

Useful controls shared by `classify`, `evaluate`, and `run` are:

```text
--model MODEL
--retrieval-mode auto|heuristic|all
--top-k N
--chunk-size N
--chunk-overlap N
--no-ocr
```

`auto` is the default: deterministic matches and neighboring chunks are ranked
first, then whole-paper ONNX retrieval fills any unused context slots. A paper
with no exact match is still searched across the whole paper; it is never
silently classified as negative.

Embedded PDF text needs no extra packages. To allow the optional OCR fallback,
install its Python dependencies and the system `pdftoppm`/Tesseract programs:

```bash
uv sync --extra ocr
```

## PDF download behavior

Downloads use ordinary sequential HTTP by default. A candidate is written only
after a successful status, plausible content type, PDF signature, readable
first page, and optional checksum validation. Writes are atomic, and existing
valid PDFs are preserved.

Every attempted URL is appended to `download_attempts.jsonl`. Unresolved papers
are deduplicated in `manual_downloads.jsonl` and retried on later runs. If direct
HTTP is insufficient, install and explicitly request the browser fallback:

```bash
uv sync --extra browser
uv run iris-papers download data/metadata/2025.jsonl --browser-fallback
```

Selenium is not imported or started during an ordinary download.

## Classification definition

An IRIS paper is positive when its authors either:

- analyze observational data from the Interface Region Imaging Spectrograph;
  or
- create or analyze a synthetic observable inside an IRIS spectrograph window
  or slit-jaw channel.

A synthetic observable counts even when the paper does not mention IRIS. The
result distinguishes an explicit relationship (`EXPLICIT`) from one inferred
only from wavelength coverage (`PASSBAND_ONLY`), and records mission mention
independently.

Merely noting that IRIS observed an event is not observational use. In
particular, data availability, a coincidental observation, or a slit missing
the relevant place or time does not count when the authors do not analyze the
IRIS data. Citations, mission descriptions, and reviews of earlier work also do
not prove new data use.

Canonical coverage, following the
[IRIS instrument paper](https://doi.org/10.1007/s11207-014-0485-y), is:

- FUV1: 1331.7–1358.4 Å;
- FUV2: 1389.0–1407.0 Å;
- NUV: 2782.7–2835.1 Å; and
- slit-jaw channels: 1330, 1400, 2796, and 2832 Å.

The machine-readable values are in
[`src/paper_data_linking/iris.py`](src/paper_data_linking/iris.py). The manually
curated ADS IRIS library is append-only positive ground truth: membership is
positive, while absence is unlabeled rather than negative.

## Configuration and failure semantics

The CLI reads `OPENAI_API_KEY` and `ADS_TOKEN` from the environment or a local
`.env`. The OpenAI request has a 300-second timeout and at most two SDK retries.
The default model is the pinned `gpt-5-mini-2025-08-07`; use `--model` only for
an intentional experiment.

The classifier returns a strict Pydantic record. Extraction, retrieval, API,
schema, refusal, and evidence-validation problems are stored as
`PROCESSING_FAILED`; they are never converted to `NO`.

## Development checks

```bash
uv lock --check
uv run python -m compileall -q src scripts tests
uv run ruff check .
uv run ruff format --check .
uv run python -m paper_data_linking.models
uv run python -m paper_data_linking.classify
uv run python -m paper_data_linking.retrieval
uv run python scripts/setup_onnx.py --check
uv run python tests/test_download.py
uv run python tests/test_cli.py
```

The pipeline test uses real tracked scientific PDFs with a fake local ranker
and fake API response. It verifies selected page/chunk provenance, structured
classification, failure semantics, checkpointing, and resume without network
access or API charges. No Docker, Redis, browser, or local HTTP service is
needed for the default workflow.
