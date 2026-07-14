# IRIS Paper LLM roadmap

Last reconciled with the repository: 2026-07-14.

This is the active implementation plan. Completed experimental detail lives in
`data/eval/`; this file tracks the decisions that must remain true and the work
that is still actionable.

## Fixed product decisions

- Keep page-aware PDF extraction and local ONNX retrieval before the OpenAI API.
- Count every synthetic observable within an IRIS-observable passband or
  spectral line, even when the paper does not mention IRIS.
- Record synthetic connections as `EXPLICIT` or `PASSBAND_ONLY`, and record
  mission mention independently.
- Count observational IRIS use only when the paper's authors analyze the data.
  Data availability, a coincidental observation, or a slit that missed the
  relevant place or time does not count.
- Treat the manually curated ADS IRIS library as append-only positive ground
  truth. Absence from the library is unlabeled, not negative.
- Use schema-validated structured output. Never infer a negative result from a
  processing failure or parse prose with divider strings.
- Process OpenAI requests sequentially and checkpoint every paper to JSONL.
- Use direct HTTP for PDFs. Keep Selenium only as an explicit fallback pass.
- Make one Python CLI the only supported workflow.
- Remove the web application, Make, Docker, Compose, Redis, Celery, and Chroma.
- Keep the historical v2 prompt and results as frozen evaluation artifacts, not
  as a second live pipeline.
- Do not add Batch API support, a database, distributed workers, a plugin
  framework, or a corpus-wide vector store without measured need.

## Target workflow

```text
ADS search
  -> metadata and open-access links
  -> validated local PDF cache or manual-download queue
  -> page-aware text extraction
  -> deterministic exact matches and neighboring chunks
  -> local ONNX/NumPy ranking with whole-paper fallback
  -> top-20 labeled passages
  -> OpenAI structured classification
  -> checkpointed JSONL results
  -> JSON/CSV/Markdown evaluation report
```

The exact-match stage is a context selector, never a classifier. In `auto`
mode, ONNX fills unused context slots from the whole paper. No exact matches
must still produce whole-paper candidates rather than an automatic `NO`.

## Target repository structure

Keep the final package flat unless a file becomes genuinely hard to navigate:

```text
.
├── README.md
├── TODO.md
├── pyproject.toml
├── uv.lock
├── data/
│   ├── eval/                 # tracked labels, manifests, and reports
│   ├── metadata/             # generated ADS records
│   ├── pdfs/                 # ignored local cache
│   └── results/              # generated checkpointed output
├── models/
│   └── onnx/                 # ignored model cache, checksum-verified setup
├── src/paper_data_linking/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                # argparse and output only
│   ├── ads.py                # ADS search and link discovery
│   ├── download.py           # direct HTTP and optional browser fallback
│   ├── iris.py               # canonical IRIS observable definitions
│   ├── models.py             # shared Pydantic records
│   ├── retrieval.py          # extraction, chunks, and local ONNX ranking
│   ├── classify.py           # prompt, OpenAI call, and corpus checkpoints
│   └── evaluate.py           # reviewed-corpus reports
└── tests/
    └── test_pipeline.py      # small offline end-to-end check
```

This is a deletion target, not a request to create empty modules. Combine files
when doing so leaves a shorter, clearer live path.

## Current status

### Phases 0-4: complete

- [x] Frozen, versioned Phase 0 baseline and reviewed 13-paper corpus.
- [x] Append-only ADS-library evaluation semantics.
- [x] Strict Pydantic classification, provenance, evidence, and failure models.
- [x] Prompt `iris-v3.2` with the agreed observational and synthetic rules.
- [x] Page-aware chunks with stable IDs and exact evidence mapping.
- [x] Retrieval modes `auto`, `heuristic`, and `all`; top 20 is the accepted
  default.
- [x] Direct ONNX/tokenizer inference and NumPy cosine ranking.
- [x] Removed Chroma from the active path and dependencies.
- [x] Checksum-verified ONNX model setup with no silent analysis-time download.
- [x] Verified direct ranking against Chroma on all 13 papers: identical
  embeddings and top-20 ordering; maximum converted distance difference
  `2.384185791015625e-07`.
- [x] Verified structured classification on all 13 reviewed papers. The current
  retained report is 13/13 correct for overall, observational, synthetic,
  connection, and review-only fields.
- [x] Verified resume behavior skips all 13 matching successful records without
  an API call.

Evidence:

- `data/eval/phase3_retrieval_report.md`
- `data/eval/phase3_top20_results.md`
- `data/eval/phase4_report.md`
- `data/eval/phase4_top20_results.jsonl`

### Known migration gaps

- The web/LangChain fallback was removed after the CLI gate passed; Git history
  and the frozen v2 evaluation artifacts are the rollback path.
- A clean clone has 10 reviewed PDFs and acquires the other three from the
  checksum-bearing manifest without committing publisher files.
- The old Phase 0 freeze check reports expected implementation-hash drift after
  Phases 1-4. Do not rewrite the historical manifest to hide that drift.

## Phase 5: finish the reliable classification path

Already implemented:

- [x] Direct official OpenAI SDK call with the Pydantic response schema.
- [x] One paper per request, processed sequentially.
- [x] Append one result immediately after each paper.
- [x] Resume only when PDF hash, prompt, pipeline, model, retrieval mode, chunk
  settings, and top-k match a successful result.
- [x] Record request ID and token usage when available.
- [x] Record `PROCESSING_FAILED` with an error instead of converting failure to
  `NO`.
- [x] Validate model evidence against supplied chunk IDs and pages.
- [x] Keep Batch API support deferred.

Remaining:

- [x] Give the OpenAI client a 300-second timeout and at most two retries.
- [x] Make the pinned model snapshot the reproducible CLI default while keeping
  `--model` available for intentional experiments.
- [x] Run the existing classifier self-check and a one-case resume check.

Completed during web removal:

- [x] Delete `ChatOpenAI`, LangChain document/message wrappers,
  `answer_divider`, `json_divider`, and their dependencies once the fallback is
  retired.

Done when one classification function returns a validated `PaperResult`, has
bounded request behavior, and never requires prose parsing.

Implementation note (2026-07-14): the active path now defaults to
`gpt-5-mini-2025-08-07`, uses a 300-second request timeout with at most two SDK
retries, passes the offline classifier self-check, and skips an unchanged
successful case without an API call. Phase 8 removed the LangChain fallback.

## Phase 6: simplify ADS and PDF acquisition

- [x] Fix the URL fallback that assigns the return value of `list.append()`.
- [x] Keep ADS search and metadata behind ordinary callable services, with the
  token validated at the call or CLI boundary rather than during import.
- [x] Prefer ADS-provided, clearly open-access PDF links.
- [x] Replace the asynchronous downloader hierarchy with one direct HTTP path.
- [x] Download to a temporary file and validate:

  - [x] successful HTTP status;
  - [x] plausible content type;
  - [x] PDF signature;
  - [x] at least one readable page;
  - [x] expected checksum when one is supplied.
- [x] Rename atomically only after validation.
- [x] Never overwrite or delete an existing valid PDF.
- [x] Record attempted URLs, useful errors, and transient/permanent categories.
- [x] Retry transient failures on later runs and deduplicate failure records.
- [x] Write unresolved papers to a manual-download JSONL queue.
- [x] Do not reject a PDF merely because it has one page or no early `abstract`.
- [x] Replace NumPy used only for infinity in ADS pagination with `math.inf`.
- [x] Remove randomized browser headers and Selenium from the default path.
- [x] Add `--browser-fallback` as a separate pass over direct-download failures.
- [x] Keep one browser driver lifecycle and never reuse a closed driver.
- [x] Add one URL-fallback regression check and one local-file download smoke
  check; neither may use ADS or a publisher network.

Done when rerunning downloads touches only missing/retryable papers, direct
downloads never import Selenium, and every unresolved paper has a useful manual
queue record.

Implementation note (2026-07-14): `download.py` now owns sequential direct
downloads, structural/checksum validation, atomic replacement of invalid files,
attempt history, and a deduplicated manual queue. The old async hierarchy and
randomized headers were deleted. The legacy download script delegates to this
path until Phase 7 replaces Make. Its query/year options move into the unified
CLI rather than adding another temporary Make interface.

## Phase 7: add the sole supported CLI

Provide one stdlib `argparse` entry point:

```text
iris-papers search --year 2025
iris-papers download data/metadata/2025.jsonl
iris-papers classify data/pdfs/2025 --output data/results/2025.jsonl
iris-papers evaluate data/eval/reviewed_cases.jsonl
iris-papers run --year 2025
```

- [x] Add `src/paper_data_linking/cli.py` and a small `__main__.py` delegate.
- [x] Register `iris-papers = "paper_data_linking.cli:main"` in
  `pyproject.toml`.
- [x] Make commands call shared stage functions; do not duplicate pipeline
  logic inside argument handlers.
- [x] Preserve the current JSONL formats during migration.
- [x] Expose query/year, `--limit`, `--force`, `--model`,
  `--retrieval-mode auto|heuristic|all`, and `--browser-fallback` only where
  relevant.
- [x] Make `run` compose the same search, download, classify, and evaluate
  functions used by individual commands.
- [x] Print final classified-positive, negative, uncertain, failed, skipped,
  downloaded, and manual-download counts.
- [x] Use the checksum-bearing reviewed-case manifest to acquire the three
  missing evaluation PDFs. Do not commit another copy of publisher PDFs.
- [x] Prove a clean checkout can prepare and run all 13 reviewed cases.
- [x] Put the CLI installation and one-paper example first in `README.md`.

Done when one paper, a directory, and the 13-paper evaluation run without the
web app or Docker and resume from their durable JSONL artifacts.

Implementation note (2026-07-14): the CLI and compatibility script share one
set of stage functions. Offline tests exercised one real tracked PDF, directory
resume, reports, and checksum-aware acquisition. A fresh temporary cache
downloaded all three ignored reviewed PDFs with exact manifest hashes. The CLI
then consumed the retained Phase 4 JSONL, skipped all 13 matching successes
without an API client, and regenerated the 13/13 evaluation report.

## Phase 8: remove the web application and legacy runtime

Removal gate:

- [x] CLI exposes the prompt/model/retrieval choices people use.
- [x] CLI results retain classifications, evidence, provenance, errors, and
  review output.
- [x] Direct download and explicit browser fallback both work.
- [x] One stopped-and-resumed corpus run has been verified.

After the gate passes:

- [x] Move the frozen v2 prompt/config into evaluation history if necessary.
- [x] Delete `src/paper_data_linking/web_app/`.
- [x] Delete web-only Celery callbacks, HTML/highlight generation, rectangle
  helpers, and frontend serialization code.
- [x] Delete `Makefile`, `Dockerfile`, `docker-compose.yaml`, `.dockerignore`,
  `entrypoint.sh`, and `nginx/`.
- [x] Remove FastAPI/Uvicorn, Celery/Redis/Flower, SlowAPI, frontend, OCR-web,
  and LangChain dependencies that have no CLI caller.
- [x] Remove the legacy YAML-divider classification path.
- [x] Remove Docker, Nginx, browser UI, and Make instructions from `README.md`.
- [x] Generate `uv.lock` from the reduced `pyproject.toml` and document
  `uv sync` as the single installation path.
- [x] Confirm the default workflow needs no Docker, Redis, browser, or local
  HTTP server.

Git history and retained evaluation artifacts are the fallback after this
phase; dead executable web code is not.

Implementation note (2026-07-14): a live headless-Chrome check downloaded a
tracked PDF through the explicit browser fallback and matched its checksum.
The frozen v2 YAML moved under `data/eval/phase0_v2_2025/`; the web, container,
Make, and YAML/LangChain paths were deleted. A locked default `uv sync`
installed 39 packages and imported the CLI without any web, OCR, or Selenium
extras.

## Phase 9: delete remaining dead abstractions

- [ ] Follow every module from the CLI and delete files with no live caller.
- [ ] Remove unused parser and splitter variants.
- [ ] Remove the unused SOHO stepwise classifier.
- [ ] Remove factories and abstract base classes with one implementation.
- [ ] Replace LangChain `Document` with the existing `PaperChunk` or a plain
  local record.
- [ ] Keep one PDF reader and one junk/valid-PDF check.
- [ ] Remove unused settings, metadata models, constants, environment loading,
  and logging setup.
- [ ] Remove migration-only scripts whose evidence is already frozen under
  `data/eval/`; retain only scripts needed to reproduce recorded comparisons.
- [ ] Decide whether Python 3.13 is genuinely required; lower it only if useful
  and verified.

Done when the package can be understood by following the CLI into one pipeline,
and every retained module has a live caller.

## Phase 10: minimal offline verification and final documentation

- [ ] Add one end-to-end offline test using a fake embedder and fake OpenAI
  client.
- [ ] Verify selected chunks, page IDs, structured classification, and JSONL
  checkpointing in that test.
- [ ] Verify overall-label derivation and that processing failure is never
  serialized as `NO`.
- [ ] Keep gold-corpus evaluation separate because it uses real PDFs, ONNX, and
  optionally OpenAI.
- [ ] Run syntax, Ruff, formatting, classifier self-check, retrieval self-check,
  ONNX checksum check, and the offline test.
- [ ] Document setup, search/download, one-paper classification, corpus resume,
  evaluation, browser fallback, and the manual-download queue.
- [ ] Confirm every documented command is executable from a clean checkout.

## Next commits

The first three commits already exist:

1. [x] `test: add IRIS classification baseline` (`c0f344f`)
2. [x] `feat: add local IRIS classification pipeline` (`b03e2a3`)
3. [x] `docs: record retrieval evaluation` (`aa8f86e`)

Keep the remaining work independently reviewable:

4. [x] `fix: bound OpenAI requests` (`71dc258`)
5. [x] `refactor: simplify PDF acquisition` (`0832eec`)
6. [x] `feat: add iris-papers CLI` (`268112b`)
7. [ ] `refactor: remove web application`
8. [ ] `refactor: delete legacy pipeline code`
9. [ ] `test: add offline pipeline smoke test`
10. [ ] `docs: document the local workflow`

## Definition of complete

- A clean checkout installs with `uv sync` and prepares the checksum-verified
  local ONNX model explicitly.
- `iris-papers run` searches ADS, downloads what it can, records manual work,
  classifies each paper, checkpoints results, and resumes safely.
- Every positive or uncertain field cites a supplied page/chunk.
- The reviewed corpus remains reproducible and quality changes are reported.
- No active code imports Chroma, LangChain, FastAPI, Celery, Redis, or Selenium
  unless the explicit browser-download extra is requested.
- The repository has one documented interface and no web application,
  container stack, Make orchestration, or hidden task state.
