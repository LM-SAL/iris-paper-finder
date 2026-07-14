# Phase 3 retrieval comparison

This comparison uses the 13 reviewed PDFs and makes no OpenAI requests.

## Frozen legacy behavior

- Configuration: 500-token chunks, 50-token overlap, Chroma's
  all-MiniLM-L6-v2 ONNX embedder, fuzzy heuristic filter, top 10.
- The legacy splitter produced 476 chunks and returned 99 selected chunks.
- `future_iris_cross_reference` returned no chunks because its heuristic filter
  found no match. The old pipeline therefore skipped semantic retrieval.
- Exact legacy positions, distances, text hashes, and excerpts are stored in
  `phase3_legacy_retrieval.json`.

## Page-aware `auto` retrieval

- Page-aware extraction produced 1,383 stable chunks. Normal PDFs used embedded
  text; OCR was not imported or run.
- The selector uses only IRIS names/instrument aliases, SJI channels, and
  wavelengths inside the canonical IRIS spectrograph windows. It does not use
  fuzzy matching or generic solar-physics phrases.
- One paper, `citation_only_psp_sources`, had no deterministic match. `auto`
  correctly supplied all 20 slots from whole-paper ONNX ranking.
- Top 10, 20, and 30 selected 130, 260, and 383 chunks respectively. The final
  paper has only 23 chunks, which explains the three missing top-30 slots.
- The top-20 contexts contain 89 exact matches, 84 neighboring chunks, and 87
  whole-paper fallback chunks.
- Every top-10 context contains at least two chunks with methods, observations,
  data, analysis, or results language. Manual inspection confirmed that the
  known observational and synthetic positives include methods/results evidence,
  while the slit-miss negative retains the sentence explaining that the IRIS
  slit passed through only after the eruption ended.
- A full second run reproduced the stored chunk IDs, ordering, distances, and
  excerpts exactly.

## Decision

Keep top 20 as the default. It adds methods/results coverage and
protects no-match papers without sending the full 8,000-99,000-token local text.

The final structured run used prompt `iris-v3.2` and pinned model
`gpt-5-mini-2025-08-07`. It classified all 13 reviewed papers correctly for
overall, observational use, synthetic use, synthetic connection, and
review-only status, with no processing failures. The accepted run used 85,963
input tokens and 18,277 output tokens. Top 20 is therefore retained as the
default for the reviewed corpus; broader library evaluation can still revise it.

Two reviewed component labels were corrected before the final run. The IRIS
mosaic paper creates mock SUIT Mg II k images, and the Bifrost comparison paper
creates synthetic Mg II k and Si IV observables. Both count as synthetic under
the agreed rule that every synthetic observable inside IRIS coverage is
positive. Their overall positive labels did not change.
