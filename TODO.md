# IRIS Paper LLM simplification plan

This document turns the code and prompt review into an implementation plan. It
is intentionally ordered so that classification behavior is measured before
components are replaced or deleted.

## Agreed direction

- Keep local PDF text extraction and local ONNX retrieval as the default
  pipeline. It reduces the text sent to the OpenAI API and keeps retrieval
  inspectable.
- Count every synthetic observable within an IRIS-observable passband or
  spectral line, even when the paper does not mention IRIS. Flag whether the
  connection is explicit or inferred from the passband/line.
- Treat the manually curated ADS IRIS library as immutable positive ground
  truth. Papers outside it remain unlabeled until reviewed.
- Replace brittle Markdown parsing with schema-validated structured output.
- Make one Python CLI the sole workflow interface.
- Remove the web application and its supporting services after the command-line
  workflow covers the required behavior.
- Remove Chroma's temporary per-PDF database if direct in-memory ranking with
  the existing ONNX embeddings is equivalent.
- Use sequential OpenAI requests with per-paper checkpoints and resumption.
- Use ordinary HTTP PDF downloads by default and retain Selenium only as an
  explicit retry fallback.
- Remove Make, Docker, and Compose rather than maintaining parallel ways to run
  the same local workflow.
- Keep OCR as a fallback for papers without usable embedded text.
- Preserve JSONL artifacts so interrupted runs can resume and results remain
  easy to inspect.

## Target workflow

```text
ADS query
  -> metadata and open-access links
  -> local PDF cache / manual-download queue
  -> page-aware text extraction
  -> paragraph or token chunks
  -> exact-match candidates plus neighboring chunks
  -> local ONNX embeddings and in-memory top-k ranking
  -> whole-paper ONNX fallback when exact matches are insufficient
  -> OpenAI structured classification
  -> versioned JSONL results
  -> JSON/CSV/Markdown evaluation report
```

The exact-match stage is a cheap selector, not a classifier or negative filter.
In the default `auto` mode, ONNX fills any unused context slots from the whole
paper, including when there are no exact matches. The OpenAI model receives the
selected passages with their page and chunk identifiers so every positive or
uncertain result can cite its evidence.

## Proposed final repository structure

This is a target, not a requirement to create every file immediately. Combine
files when that remains clearer.

```text
.
├── README.md
├── TODO.md
├── pyproject.toml
├── uv.lock
├── data/
│   ├── bibcodes/
│   ├── metadata/
│   ├── links/
│   ├── pdfs/
│   ├── results/
│   └── eval/
├── src/paper_data_linking/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── ads.py
│   ├── download.py
│   ├── evaluate.py
│   ├── models.py
│   ├── pipeline.py
│   └── classify.py
└── tests/
    └── test_pipeline.py
```

Expected responsibilities:

- `cli.py`: user-facing `search`, `download`, `classify`, `evaluate`, and `run`
  commands. It delegates rather than containing pipeline logic.
- `ads.py`: ADS query, library, metadata, and link discovery operations.
- `download.py`: ordinary HTTP downloading, validation, retry policy, and the
  optional browser fallback.
- `evaluate.py`: reports against a versioned ADS-library snapshot and reviewed
  labels.
- `models.py`: Pydantic records shared by the pipeline, result writer, and
  evaluator.
- `pipeline.py`: PDF extraction, chunking, exact-match selection, ONNX
  retrieval, and resumable directory processing. Split this file only if it
  becomes difficult to navigate.
- `classify.py`: the short versioned IRIS prompt and direct structured OpenAI
  call. The response schema stays in `models.py`, where it can be validated.

Do not create a module merely to match this diagram. Combine responsibilities
when the result is easier to follow.

## Phase 0: Freeze a trustworthy baseline

Do this before changing retrieval, prompts, or models.

- [x] Define a synthetic positive as any synthetic observable within an
  IRIS-observable passband or spectral line.
  - [x] Do not require the paper to mention IRIS.
  - [x] Distinguish an explicit IRIS connection from a connection inferred only
    from the passband or line.
  - [x] Define canonical spectrograph windows and slit-jaw channels once in
    `src/paper_data_linking/iris.py` from the IRIS instrument paper.
  - [x] Reuse those Python values in the Phase 2 prompt.
  - [x] Reuse those Python values in exact-match retrieval during Phase 3.
  - [x] Write the agreed definition in the README.
  - [x] Supersede the historical v2 prompt with the same definition in Phase 2;
    keep the v2 file frozen as a reproducible fallback.
- [x] Keep the current rule that instrument, calibration, or software papers
  without observational or synthetic data use are not positives.
- [x] Treat every paper in the manually curated ADS IRIS library as positive
  ground truth. The library is append-only: evaluation never removes a paper.
- [x] Snapshot the library bibcodes and snapshot date for every evaluation so
  an append-only library remains reproducible.
- [x] Create a versioned library snapshot plus
  `data/eval/reviewed_cases.jsonl`; do not duplicate all 756 library labels in
  another JSONL file.
  - [x] Include clear observational-data positives.
  - [x] Include clear synthetic-data positives.
  - [x] Include explicit synthetic connections.
  - [x] Add a reviewed `PASSBAND_ONLY` synthetic paper.
  - [x] Include papers that only cite or describe IRIS.
  - [x] Include unrelated uses of the acronym IRIS.
  - [x] Include a review-only paper.
  - [x] Search for a genuinely uncertain paper; do not invent one when every
    local candidate can be resolved from its full text.
  - [x] Include difficult spectral-line cases such as Si IV and Mg II without
    an explicit IRIS connection.
  - [x] Record the expected label and a short human evidence note.
- [x] Preserve every available v2 result and record absent results explicitly;
  do not spend API credits trying to recreate missing historical output.
- [x] Record baseline configuration:
  - [x] PDF checksum.
  - [x] prompt/config filename and checksum.
  - [x] model name, including that the historical run used an unpinned alias.
  - [x] chunk size, overlap, heuristic threshold/config checksum, and top-k.
  - [x] Record that selected chunk IDs and retrieval scores were not persisted
    and cannot be recovered from the saved output.
- [x] Report available confusion counts separately for observational,
  synthetic, review,
  and overall classification.
- [x] Evaluate library members as follows:
  - [x] predicted `YES`: recovered positive.
  - [x] predicted `NO`: false negative.
  - [x] predicted `UNCERTAIN`: unresolved positive.
  - [x] missing result: pipeline failure.
- [x] Treat papers outside the library as unlabeled, not negative. Queue
  predicted `YES` papers for manual review and possible addition to the library.

Acceptance criteria:

- A second person can read the gold JSONL and understand why each paper has its
  label.
- The evaluation report identifies the exact ADS-library snapshot used.
- A baseline run can be repeated without the web application.
- Later phases can show whether accuracy changed rather than relying on a few
  anecdotes.

## Phase 1: Define one structured result

- [x] Add Pydantic enums/models for:
  - [x] `YES`, `NO`, and `UNCERTAIN` decisions.
  - [x] observational IRIS data use.
  - [x] synthetic IRIS observable use.
  - [x] synthetic connection: `EXPLICIT`, `PASSBAND_ONLY`, `NOT_APPLICABLE`,
    or `UNCERTAIN`.
  - [x] whether the target IRIS mission is mentioned anywhere in the paper,
    independently of how the synthetic observable is connected to IRIS.
  - [x] review-only status.
  - [x] IRIS aspects: telescope, spectrograph, and slit-jaw imager.
  - [x] evidence containing page/chunk ID and a short reason.
  - [x] pipeline provenance and errors.
- [x] Derive the overall classification in Python:
  - `YES` when observational or synthetic use is `YES`.
  - `NO` when both are `NO`.
  - `UNCERTAIN` otherwise.
- [x] Require a positive synthetic result to record the connection basis and,
  separately, whether the paper mentions the target IRIS mission.
- [x] Enforce synthetic-connection consistency:
  - `synthetic_use=YES` with an explicit IRIS relationship uses `EXPLICIT`.
  - `synthetic_use=YES` based only on an observable IRIS could measure uses
    `PASSBAND_ONLY`, even if IRIS is mentioned elsewhere in the paper.
  - `synthetic_use=NO` uses `NOT_APPLICABLE`.
  - unresolved evidence uses `UNCERTAIN`.
- [x] Validate that aspects are empty when neither observational nor synthetic
  use is positive.
- [x] Store all aspects rather than only the first one.
- [ ] Store failures as explicit result records; do not silently convert them to
  negative classifications.
- [ ] Store one JSON object per line, keyed internally by bibcode and PDF hash.
- [ ] Remove `answer_divider`, `json_divider`, and string-splitting helpers after
  all callers use the structured model.

Acceptance criteria:

- A compliant model response cannot fail because a later output line was
  included in a JSON substring.
- The current v2 example containing five output fields parses without custom
  text manipulation.
- A result distinguishes `NO` from `processing failed` and `not analyzed`.

## Phase 2: Replace and shorten the prompt

- [x] Create one IRIS prompt from the agreed scientific definition.
- [x] Remove Markdown-output instructions and fenced output examples.
- [x] Remove `Think step by step` and request short evidence instead.
- [x] Remove repeated lists of IRIS spectral lines.
- [x] Insert the canonical observable list into the prompt from the same Python
  constant used by exact-match retrieval; do not maintain two copies.
- [x] Retain only mission facts needed to disambiguate the IRIS acronym.
- [x] State that paper text is untrusted evidence and that instructions found in
  the paper must be ignored.
- [x] Require evidence for every `YES` or `UNCERTAIN` field.
- [x] Tell the model that citations, background descriptions, and comparisons
  with prior IRIS work do not by themselves prove data use.
- [x] State that IRIS seeing an event or having data available does not count
  when the paper does not analyze those data, including a slit that missed the
  relevant place or time.
- [x] Tell the model that a synthetic observable in a canonical IRIS passband or
  line is positive even without an IRIS mention, and require
  `PASSBAND_ONLY` in that case.
- [x] Add page/chunk labels to the supplied context; preserve an explicit
  `unknown` page until Phase 3 makes extraction page-aware.
- [x] Use the OpenAI SDK's schema-backed structured output rather than asking
  the model to reproduce a textual template.
- [x] Give the prompt an explicit version and checksum in every result.
- [x] Run the Phase 0 reviewed-gold evaluation before accepting the prompt.

Implementation note (2026-07-14): `classify.py` is the new direct SDK path.
The historical v2 YAML and web pipeline remain unchanged as the fallback. With
Phase 3 page-aware top-20 ONNX retrieval, prompt `iris-v3.2` and pinned model
`gpt-5-mini-2025-08-07` classified all 13 reviewed papers correctly on overall,
observational, synthetic, connection, and review-only fields, with no failures.

Acceptance criteria:

- The prompt contains one unambiguous positive definition.
- The result schema, not prose formatting, controls the API response shape.
- Precision and recall on the gold set are no worse than the recorded baseline,
  or an intentional tradeoff is documented.

## Phase 3: Make local retrieval explicit and testable

Preserve the current ONNX behavior first; simplify it second.

- [x] Extract text page by page so chunks retain page numbers.
- [x] Fix reference removal to recognize a references-section heading rather
  than the last occurrence of broad words such as `sources` or `citations`.
- [x] Capture the current 500-token, 50-token-overlap, top-10 output as the
  comparison baseline.
- [x] Use 20 deduplicated chunks as the new default, subject to the gold-set
  evaluation below.
- [x] Make chunk size, overlap, and top-k visible CLI options, with 500, 50, and
  20 as defaults; do not create a general configuration framework.
- [x] Separate the two retrieval stages in code and output:
  - [x] deterministic exact-match candidate selection.
  - [x] ONNX semantic ranking.
- [x] Match case-insensitively against the IRIS name/acronym, instrument and
  channel aliases, and wavelengths in the canonical passband windows.
- [x] Remove generic selectors such as `solar chromosphere`, `transition
  region`, `UV imaging`, bare `spectrograph`, and similar domain-wide terms.
- [x] Avoid fuzzy matching for short scientific terms.
- [x] Add the immediate neighboring chunks around exact matches, then
  deduplicate overlaps before ranking.
- [x] Support three deliberately small retrieval modes:
  - [x] `--retrieval-mode auto` (default): rank exact matches and neighbors,
    then fill unused top-k slots from whole-paper ONNX ranking.
  - [x] `--retrieval-mode heuristic`: rank only exact matches and neighbors,
    with no global fill.
  - [x] `--retrieval-mode all`: rank the whole paper directly.
- [x] In `auto`, use whole-paper ranking for all top-k slots when there are no
  exact matches. A failed or empty selector must never imply a negative paper.
- [x] Record each selected chunk's reason as `heuristic_match`,
  `adjacent_context`, or `global_fallback`.
- [x] Record which chunks were excluded, selected, and sent to the API.
- [x] Compare at least:
  - [x] current top 10.
  - [x] deduplicated top 20.
  - [x] deduplicated top 30.
  - [x] all exact-match and neighboring chunks within a token ceiling.
  - [x] full locally extracted text as a diagnostic baseline.
- [x] Confirm that the selected context includes evidence from methods/results,
  not only introductions and citations.
- [x] Keep OCR fallback isolated so normal searchable PDFs do not import or run
  OCR machinery.

Implementation note (2026-07-14): the legacy artifact contains all 13 reviewed
papers and exposes one old heuristic early exit. The page-aware comparison is
deterministic across two complete runs; `auto` returned context for every paper.
The final structured-classification evaluation was correct on all 13 reviewed
papers and all evaluated component fields, so top 20 remains the default. See
`data/eval/phase3_retrieval_report.md` and
`data/eval/phase3_top20_results.md`.

Acceptance criteria:

- Retrieval is deterministic for a fixed PDF, model, and configuration.
- Every selected chunk maps back to a page and chunk position.
- No paper is classified `NO` merely because exact-match selection found
  nothing.
- The gold-set evaluation validates or changes the provisional top-20 default.

## Phase 4: Replace temporary Chroma collections with in-memory ranking

Do not mix this change with prompt changes.

- [x] Preserve Chroma while capturing baseline embeddings, similarity scores,
  ordering, and selected chunk IDs for representative papers.
- [x] Call the existing local ONNX embedding model directly for candidate chunks
  and the retrieval query.
- [x] Normalize vectors and rank with NumPy dot products/cosine similarity.
- [x] Keep stable chunk IDs rather than generating random UUIDs.
- [x] Verify that selected chunks and ordering match the Chroma baseline within
  expected floating-point behavior.
- [x] Run the Phase 0 evaluation before and after the replacement.
- [x] Remove collection-name cleaning and Chroma client lifecycle code.
- [x] Remove Chroma only after identifying the smallest supported way to run the
  existing ONNX model.
  - [x] Prefer the current model/tokenizer files and the minimum direct
    `onnxruntime` and tokenizer dependencies; declare every direct import.
  - [x] Do not add a larger embedding framework merely to delete Chroma.
  - [x] Keep the direct wrapper limited to tokenization, inference, pooling,
    normalization, and ranking; no embedding framework was added.
- [x] Replace `make onnx` with an explicit, checksum-verified model setup or a
  documented model cache step.

Implementation note (2026-07-14): the direct implementation produced identical
document/query embeddings and identical top-20 ordering for all 13 reviewed
papers. After converting Chroma's squared L2 scores to cosine distance, the
largest difference was `2.384185791015625e-07`. The full structured evaluation
remained 13/13 correct overall; one secondary connection field varied on the
first request and returned the expected value on one unchanged targeted repeat.
Both outcomes are retained. See `data/eval/phase4_report.md`.

Acceptance criteria:

- No vector database or per-PDF collection is created.
- Retrieval quality matches or improves on the gold set.
- The local model remains reproducible and does not download silently during an
  analysis run.

## Phase 5: Use the OpenAI SDK directly

- [ ] Replace `ChatOpenAI` and LangChain message/document wrappers with the
  official OpenAI Python SDK and the Phase 1 Pydantic schema.
- [x] Use ordinary sequential requests for both single-paper and corpus runs.
- [ ] Process one paper at a time and checkpoint its validated result before
  starting the next paper.
- [ ] Defer Batch API support; add it only when measured corpus cost or volume
  justifies another execution path.
- [ ] Pin a model snapshot for evaluation and reproducible production runs.
- [ ] Set request timeouts and bounded retries for transient API failures.
- [ ] Write the result after every paper so a stopped run resumes safely.
- [ ] Skip an existing successful result only when PDF hash, prompt version,
  model snapshot, and retrieval configuration all match.
- [ ] Record token usage and request IDs when the API provides them.
- [ ] Do not log API keys, full request headers, or unnecessary paper text.
- [ ] Remove LangChain dependencies after no imports remain.

Acceptance criteria:

- One function accepts selected chunks and returns a validated result model.
- No application code parses model prose.
- Directory processing resumes without re-running unchanged successful papers.
- A failure stops or records only the affected paper, not the completed corpus.

## Phase 6: Simplify ADS and PDF acquisition

- [ ] Fix the URL fallback that assigns the return value of `list.append()`.
- [ ] Prefer ADS-provided and clearly open-access PDF links.
- [ ] Replace the asynchronous downloader hierarchy with an ordinary HTTP
  session unless real concurrent downloading is added and measured.
- [x] Keep Selenium only as an explicit browser fallback.
- [ ] Remove Selenium and randomized browser headers from the default path.
- [ ] Make `--browser-fallback` a separate retry pass over direct-download
  failures, with browser dependencies documented as optional.
- [ ] Give the browser pass one obvious driver lifecycle; never reuse a driver
  after its context has closed.
- [ ] Put papers still blocked after the enabled download methods into a
  manual-download queue.
- [ ] Distinguish permanent failures from transient failures.
- [ ] Retry transient failures on later runs; do not permanently blacklist them
  after one request failure.
- [ ] Download to a temporary file, inspect HTTP status/content type, validate
  the PDF signature and a minimally readable page, then rename atomically.
- [ ] Never overwrite or delete an existing valid PDF.
- [ ] Record every attempted URL, failure category, and useful error message.
- [ ] Do not delete a PDF solely because it has one page or lacks the word
  `abstract` in its first two pages.
- [ ] Deduplicate failed-bibcode records.
- [ ] Replace NumPy's use as an infinity constant in ADS pagination with the
  standard library.
- [ ] Add progress and summaries without introducing another task system.

Acceptance criteria:

- A rerun downloads only missing or retryable papers.
- A failed automatic download produces a useful manual-work record.
- Direct downloads do not import or start Selenium.
- Browser fallback has one obvious resource lifecycle and no closed-driver
  reuse.

## Phase 7: Build the primary CLI and corpus workflow

- [ ] Provide one documented entry point, for example:

  ```text
  iris-papers search --year 2025
  iris-papers download data/metadata/2025.jsonl
  iris-papers classify data/pdfs/2025 --output data/results/2025.jsonl
  iris-papers evaluate data/eval/iris_gold.jsonl
  iris-papers run --year 2025
  ```

- [ ] Register `iris-papers = "paper_data_linking.cli:main"` in
  `pyproject.toml`.
- [ ] Reuse the existing JSONL artifacts during migration.
- [x] Make the Python CLI the sole orchestration layer; remove Make rather than
  maintaining two workflows.
- [ ] Keep query strings and year ranges as command arguments rather than source
  edits.
- [ ] Make `run` compose the same `search`, `download`, `classify`, and
  `evaluate` functions used by the individual commands.
- [ ] Make every stage read and update durable manifests so interrupted runs
  resume without hidden task state.
- [ ] Provide `--limit` and `--force` for small experiments and intentional
  reruns.
- [ ] Expose `--retrieval-mode auto|heuristic|all` and `--browser-fallback`
  without creating a general plugin/configuration layer.
- [ ] Print a final count of successful, negative, uncertain, failed, skipped,
  and manual-download papers.
- [ ] Update the README so the non-web workflow is the first and default usage.

Acceptance criteria:

- A new user can run one paper and a directory without Docker.
- The documented commands match executable entry points.
- Corpus processing writes durable output as each paper finishes.
- The end-to-end command and individual stage commands produce the same
  artifacts.

## Phase 8: Remove the web application

Delete instead of repairing web-only bugs that disappear with the web layer.

- [x] Confirm that nobody requires shared remote uploads or browser PDF
  highlighting.
- [ ] Confirm that the CLI exposes the prompt selection and result information
  people actually use.
- [ ] Remove `src/paper_data_linking/web_app/`.
- [ ] Remove web-only code from processing modules:
  - [ ] Celery progress callbacks and stage-message formatting.
  - [ ] HTML generation and raw highlight spans.
  - [ ] rectangle/highlight helpers.
  - [ ] frontend serialization wrappers.
- [x] Decide that no CLI container is retained without a demonstrated
  deployment requirement.
- [ ] Remove `Makefile`.
- [ ] Remove `docker-compose.yaml`, `Dockerfile`, `.dockerignore`, `nginx/`, and
  `entrypoint.sh`.
- [ ] Remove web-only dependencies:
  - [ ] FastAPI and Uvicorn.
  - [ ] Celery and Redis.
  - [ ] Flower.
  - [ ] SlowAPI.
  - [ ] Jinja2/frontend extras.
  - [ ] PDF.js, D3, jQuery, Bootstrap, Axios, Marked, and JSON viewer assets.
- [ ] Remove the no-op upload endpoint and polling client.
- [ ] Remove dead time-range visualization code.
- [ ] Replace the hand-pinned `requirements.txt` with `uv.lock` generated from
  the reduced `pyproject.toml`; document `uv sync` as the single installation
  path.
- [ ] Remove README instructions for Docker, Nginx, and the browser UI.
- [ ] Provide a small CLI-generated JSON, CSV, or Markdown review report instead
  of preserving a server for result viewing.
- [ ] Preserve screenshots or historical documentation only if they have
  research-report value; do not retain executable dead code for history.

Acceptance criteria:

- The default workflow requires no Docker, Redis, browser, or local HTTP
  request; only the explicit Selenium fallback requires a browser.
- No web-only dependency remains in the lock file.
- The same PDFs can be analyzed and evaluated through the CLI.
- There is one documented installation path and one workflow interface.

## Phase 9: Delete remaining dead abstractions

Perform this after the primary pipeline works without web imports.

- [ ] Remove unused `parsers.py` functions or merge the one retained PDF reader
  into `pipeline.py`.
- [ ] Remove unused `UnstructuredSplitter` and `PyMuPDFSplitter` variants.
- [ ] Remove the unused SOHO stepwise classifier.
- [ ] Remove splitter/embedder/plugin factories with only one implementation.
- [ ] Remove abstract base classes with one concrete implementation and no
  external extension requirement.
- [ ] Remove duplicate junk-PDF detection implementations.
- [ ] Remove unused settings such as `DATA_DIR_NAME`, `CHROMEDRIVER_PATH`, and
  web database paths.
- [ ] Remove unused metadata models and constants.
- [ ] Consolidate duplicate environment loading and logging configuration.
- [ ] Lower the minimum Python version if no Python 3.13-only behavior is used
  and broader installation support is useful.

Acceptance criteria:

- Every retained module has a live caller.
- The package can be understood by following the CLI into one pipeline.
- No new framework or plugin system replaces the deleted one.

## Phase 10: Minimal verification suite

- [ ] Add one small pipeline test using a fake embedder and fake OpenAI client.
- [ ] Verify that the highest-ranked chunks, page identifiers, and structured
  result survive the full local flow.
- [ ] Add a regression test for the URL fallback.
- [ ] Add a regression test for overall-label derivation.
- [ ] Add a regression test showing a processing failure is not saved as `NO`.
- [ ] Run the gold-set evaluator as a separate, explicit quality check rather
  than making network/model calls part of normal unit tests.
- [ ] Keep syntax, Ruff, and formatting checks.
- [ ] Add a smoke command that classifies one local text fixture without
  Docker.

Acceptance criteria:

- A code regression fails locally without calling ADS or OpenAI.
- A prompt, model, or retrieval change produces an evaluation comparison before
  it is accepted.

## Suggested implementation/commit order

Keep commits independently reviewable and avoid mixing behavioral changes with
deletions.

1. `test: add IRIS classification baseline`
2. `feat: add structured classification result`
3. `refactor: call OpenAI with structured output`
4. `refactor: make retrieval page-aware`
5. `refactor: rank ONNX embeddings in memory`
6. `fix: simplify PDF acquisition failures`
7. `feat: add local classification CLI`
8. `refactor: remove web application`
9. `refactor: delete unused pipeline code`
10. `docs: document the local corpus workflow`

## Resolved decision log

| Decision | Agreed outcome |
| --- | --- |
| What qualifies as synthetic IRIS data? | Every synthetic observable within an IRIS-observable passband/line is positive. Record `EXPLICIT` or `PASSBAND_ONLY`. |
| Is the ADS IRIS library ground truth? | Yes. It is immutable positive ground truth and append-only; absence remains unlabeled. |
| Default retrieval count | Top 20, deduplicated; accepted after comparing 10/20/30 and passing the reviewed corpus. |
| Keep heuristic filtering? | Use deterministic exact matches plus neighbors, with whole-paper ONNX fill in default `auto` mode. |
| Remove Chroma? | Done. Direct ONNX/NumPy ranking passed embedding, ordering, and classification checks. |
| Sequential or Batch API? | Sequential per-paper requests with immediate checkpoints. Defer Batch API support. |
| Keep Selenium fallback? | Yes, only as an explicit retry pass after direct HTTP failures. |
| Keep Make? | No. The Python CLI is the sole workflow interface. |
| Keep any web UI? | No. Remove it after CLI parity is verified. |
| Keep Docker for reproducibility? | No. Use `pyproject.toml` plus `uv.lock`; add deployment packaging only for a real deployment. |

## Deferred unless evidence requires it

- Persistent vector search across the entire paper corpus.
- OpenAI Batch API execution.
- A plugin framework for multiple missions or classifiers.
- A relational database.
- Distributed workers or a task queue.
- Browser PDF highlighting.
- A remote multi-user service.
- Docker or other deployment packaging.
- Fine-tuning.
- Automated publisher-login or bot-protection bypasses.

Add these only when a demonstrated workflow cannot be handled by the local CLI,
JSONL state, and current retrieval/classification pipeline.
