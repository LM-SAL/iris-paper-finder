# Legacy v2 baseline report

Created 2026-07-14 from the saved 2025 run. No ADS or OpenAI
request was made while creating this snapshot.

## Corpus

- ADS library snapshot: 756 positive bibcodes
- 2025 search candidates: 123
- Saved classifications: 117
- Missing results/download failures: 6/6
- Predictions: 45 YES, 67 NO,
  5 UNCERTAIN

## Curated ADS-library positives in candidate scope

| Outcome | Count |
| --- | ---: |
| Expected positives | 45 |
| Recovered positive | 41 |
| False negative | 3 |
| Unresolved positive | 0 |
| Pipeline failure | 1 |

There were 4 YES predictions outside
the ADS library. 1 is now a
reviewed negative, while 3 remain
queued and are not counted as false positives.

## Manually reviewed cases

These rows compare the saved *overall* v2 prediction within each scientific
case category. Component-level v2 predictions were not stored reliably.

| Category | Cases | With baseline | Correct | Incorrect | Missing baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| observational | 3 | 3 | 3 | 0 | 0 |
| synthetic | 5 | 5 | 3 | 2 | 0 |
| review_only | 1 | 1 | 0 | 1 | 0 |
| overall | 13 | 9 | 5 | 4 | 4 |

Pending scope review: None

## Known limitations

- The saved result writer discarded all but the first aspect, and the v2 parser failed to preserve aspects.
- Selected chunk IDs, similarity scores, token usage, and request IDs were not persisted and cannot be reconstructed from saved output.
- Existing results did not persist per-paper prompt/model provenance; the v2 config is the reported run configuration, not cryptographically linked to each response.
- gpt-5-mini was recorded as a moving alias rather than a pinned model snapshot.
- The ADS library supplies positive labels only; candidates outside it are unlabeled until reviewed.
- No genuinely uncertain paper was found in the local corpus; full-text review resolved every candidate considered.
