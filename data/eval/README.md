# Evaluation data

`reviewed_cases.jsonl` is the active 13-paper gold corpus. It contains the
manually assigned labels, evidence notes, PDF locations, and checksums used by
`iris-papers evaluate`.

`ads_iris_library_2026-01-07.txt` preserves the append-only ADS IRIS library
snapshot used as positive ground truth. Absence from that library is unlabeled,
never negative. `outside_library_review_queue.txt` retains the three unresolved
candidates. The removed v2 runtime and migration evidence remain available in
Git history.

The accepted structured evaluation is `reference_results.jsonl` with its
human-readable decision record in `evaluation_report.md`. It preserves the full
schema-validated results, request provenance, and observed model variance.

The frozen Chroma ranking baseline remains as a full-corpus regression check
for the direct ONNX/NumPy implementation:

```bash
uv run python scripts/check_chroma_ranking.py
```

The check requires identical selected-chunk order and cosine distance within
tolerance. Bitwise embedding hashes remain diagnostic because floating-point
output can vary across ONNX platforms.

Run or resume the gold corpus with:

```bash
uv run iris-papers evaluate data/eval/reviewed_cases.jsonl \
  --output data/eval/classification_results.jsonl
```
