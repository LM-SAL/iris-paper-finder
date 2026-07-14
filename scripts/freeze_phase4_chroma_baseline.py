"""Freeze Chroma embeddings and ranking before the Phase 4 replacement.

Run this only in the legacy Docker image, which contains Chroma 1.3.5.
"""

# ruff: noqa: T201

import json
import hashlib
import argparse
from pathlib import Path

import numpy as np
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from paper_data_linking.retrieval import IRIS_RETRIEVAL_QUERY, chunk_pdf, select_candidates

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/eval/reviewed_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "data/eval/phase4_chroma_baseline.json"


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


def _rank_chroma(chunks, embeddings, query_embedding, collection_name: str) -> list[dict]:
    reasons = select_candidates(chunks)
    collection = chromadb.Client().create_collection(collection_name)
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=embeddings,
        metadatas=[{"is_candidate": int(chunk.chunk_id in reasons)} for chunk in chunks],
    )
    selected = []
    if reasons:
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(20, len(reasons)),
            where={"is_candidate": 1},
        )
        selected.extend(zip(result["ids"][0], result["distances"][0], strict=True))
    if len(selected) < min(20, len(chunks)):
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(len(chunks), 20 + len(selected)),
        )
        selected_ids = {chunk_id for chunk_id, _distance in selected}
        for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True):
            if chunk_id not in selected_ids:
                selected.append((chunk_id, distance))
                selected_ids.add(chunk_id)
            if len(selected) == min(20, len(chunks)):
                break
    return [{"chunk_id": chunk_id, "distance": distance} for chunk_id, distance in selected]


def build_baseline(limit: int | None = None) -> dict:
    cases = [json.loads(line) for line in CASES_PATH.read_text().splitlines()]
    if limit is not None:
        cases = cases[:limit]
    model = DefaultEmbeddingFunction()
    papers = []
    for case in cases:
        chunks, _used_ocr = chunk_pdf(ROOT / case["path"])
        document_embeddings = np.asarray(model([chunk.text for chunk in chunks]), dtype=np.float32)
        query_embedding = np.asarray(model([IRIS_RETRIEVAL_QUERY])[0], dtype=np.float32)
        papers.append(
            {
                "id": case["id"],
                "chunk_count": len(chunks),
                "document_embeddings_sha256": _hash(document_embeddings),
                "query_embedding_sha256": _hash(query_embedding),
                "document_norm_range": [
                    float(np.linalg.norm(document_embeddings, axis=1).min()),
                    float(np.linalg.norm(document_embeddings, axis=1).max()),
                ],
                "query_norm": float(np.linalg.norm(query_embedding)),
                "selected": _rank_chroma(
                    chunks,
                    document_embeddings,
                    query_embedding,
                    f"phase4-{hashlib.sha256(case['id'].encode()).hexdigest()[:16]}",
                ),
            }
        )
    return {
        "schema_version": 1,
        "backend": "Chroma 1.3.5 DefaultEmbeddingFunction",
        "model": "all-MiniLM-L6-v2 ONNX",
        "papers": papers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build_baseline(args.limit), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text() != rendered:
            msg = f"Phase 4 Chroma baseline is stale: {args.output}"
            raise SystemExit(msg)
        print("Phase 4 Chroma baseline is current.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
