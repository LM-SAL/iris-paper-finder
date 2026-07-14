"""Compare direct ONNX/NumPy ranking with the frozen Chroma baseline."""

import hashlib
import json
from pathlib import Path

import numpy as np

from iris_paper_llm.models import RetrievalMode
from iris_paper_llm.retrieval import IRIS_RETRIEVAL_QUERY, ONNXEmbedder, chunk_pdf, retrieve_chunks

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/eval/reviewed_cases.jsonl"
BASELINE_PATH = ROOT / "data/eval/chroma_ranking_baseline.json"
DISTANCE_TOLERANCE = 2e-5


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


def build_comparison() -> dict:
    cases = [json.loads(line) for line in CASES_PATH.read_text().splitlines()]
    baseline = {paper["id"]: paper for paper in json.loads(BASELINE_PATH.read_text())["papers"]}
    embedder = ONNXEmbedder()
    papers = []
    for case in cases:
        chunks, _used_ocr = chunk_pdf(ROOT / case["path"])
        result = retrieve_chunks(chunks, mode=RetrievalMode.AUTO, embedder=embedder)
        old = baseline[case["id"]]
        old_selected = old["selected"]
        selected_ids = [chunk.chunk_id for chunk in result.selected]
        old_ids = [chunk["chunk_id"] for chunk in old_selected]
        deltas = [
            abs(chunk.distance - previous["distance"] / 2.0)
            for chunk, previous in zip(result.selected, old_selected, strict=True)
        ]
        papers.append(
            {
                "id": case["id"],
                "document_embeddings_exact": _hash(embedder.embeddings) == old["document_embeddings_sha256"],
                "query_embedding_exact": _hash(embedder.model.encode([IRIS_RETRIEVAL_QUERY])[0])
                == old["query_embedding_sha256"],
                "selected_order_exact": selected_ids == old_ids,
                "max_cosine_distance_delta": max(deltas, default=0.0),
                "selected_chunk_ids": selected_ids,
            }
        )
    passed = all(
        paper["selected_order_exact"] and paper["max_cosine_distance_delta"] <= DISTANCE_TOLERANCE for paper in papers
    )
    return {
        "schema_version": 1,
        "baseline": str(BASELINE_PATH.relative_to(ROOT)),
        "backend": "direct all-MiniLM-L6-v2 ONNX plus NumPy cosine distance",
        "distance_relation": "Chroma squared L2 / 2 equals cosine distance for normalized vectors",
        "distance_tolerance": DISTANCE_TOLERANCE,
        "passed": passed,
        "papers": papers,
    }


def main() -> None:
    comparison = build_comparison()
    if not comparison["passed"]:
        msg = "Direct ONNX ranking differs from the Chroma baseline"
        raise SystemExit(msg)
    print("Direct ONNX ranking comparison passed.")


if __name__ == "__main__":
    main()
