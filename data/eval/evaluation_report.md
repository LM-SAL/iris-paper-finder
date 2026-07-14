# Direct ONNX evaluation

## Equivalence

The frozen Chroma 1.3.5 baseline and the direct ONNX/NumPy implementation were
compared on all 13 reviewed PDFs:

- document embeddings: 13/13 byte-for-byte identical;
- query embeddings: 13/13 byte-for-byte identical;
- selected top-20 chunk IDs and ordering: 13/13 identical;
- largest distance difference after converting Chroma squared L2 to cosine
  distance: `2.384185791015625e-07`.

For normalized vectors, Chroma's squared L2 distance is twice the cosine
distance used by the new implementation. This changes only the displayed score
scale, not ranking.

## Classification

The unchanged `iris-v3.2` prompt, pinned `gpt-5-mini-2025-08-07` model, and
identical retrieved contexts produced 13/13 correct overall, observational,
synthetic, and review-only labels on the first full run. One connection label
varied from `EXPLICIT` to `PASSBAND_ONLY`; a single unchanged targeted repeat
returned `EXPLICIT`. Both requests remain in the JSONL artifact to preserve the
observed model variance. The current report is 13/13 on every evaluated field.

The full run used 85,963 input tokens and 16,559 output tokens. The targeted
repeat brought the total to 91,698 input and 18,579 output tokens across 14
unique request IDs.

## Decision

Use direct all-MiniLM-L6-v2 ONNX inference and in-memory NumPy cosine ranking.
Chroma is no longer an application dependency and no collection is created.
`uv run python scripts/setup_onnx.py` explicitly downloads and verifies the
pinned model; analysis never downloads model files.
