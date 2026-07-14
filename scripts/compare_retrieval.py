"""Compare Phase 3 retrieval at top-k 10, 20, and 30 on reviewed PDFs."""

# ruff: noqa: T201

from __future__ import annotations

import re
import json
import hashlib
import argparse
from pathlib import Path

import tiktoken

from paper_data_linking.models import RetrievalMode, SelectionReason
from paper_data_linking.retrieval import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_pdf,
    retrieve_chunks,
    select_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/eval/reviewed_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "data/eval/phase3_retrieval_comparison.json"
TOP_K_VALUES = (10, 20, 30)
TOKEN_CEILING = 10_000
METHOD_RESULTS_TERM = re.compile(r"\b(?:methodology|methods?|observations?|data|analysis|results?)\b", re.IGNORECASE)


def load_cases() -> list[dict]:
    cases = []
    for line in CASES_PATH.read_text().splitlines():
        case = json.loads(line)
        if (ROOT / case["path"]).is_file():
            cases.append(case)
    return cases


def summarize_selected(result) -> list[dict]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "page": chunk.page,
            "distance": chunk.distance,
            "reason": chunk.reason,
            "text_sha256": hashlib.sha256(chunk.text.encode()).hexdigest(),
            "text_excerpt": " ".join(chunk.text.split())[:240],
        }
        for chunk in result.selected
    ]


def compare_paper(case: dict, encoding) -> dict:
    chunks, used_ocr = chunk_pdf(ROOT / case["path"])
    reasons = select_candidates(chunks)
    exact_neighbor_ids = set(reasons)
    ceiling_ids = []
    ceiling_tokens = 0
    for chunk in chunks:
        if chunk.chunk_id not in exact_neighbor_ids:
            continue
        token_count = len(encoding.encode(chunk.text, disallowed_special=()))
        if ceiling_tokens + token_count > TOKEN_CEILING:
            break
        ceiling_ids.append(chunk.chunk_id)
        ceiling_tokens += token_count

    top_k = {}
    for count in TOP_K_VALUES:
        result = retrieve_chunks(chunks, mode=RetrievalMode.AUTO, top_k=count)
        top_k[str(count)] = {
            "selected": summarize_selected(result),
            "method_or_results_chunk_ids": [
                chunk.chunk_id for chunk in result.selected if METHOD_RESULTS_TERM.search(chunk.text)
            ],
        }

    full_text = "\n\n".join(chunk.text for chunk in chunks)
    return {
        "id": case["id"],
        "bibcode": case["bibcode"],
        "path": case["path"],
        "expected_overall": case["expected_overall"],
        "used_ocr": used_ocr,
        "chunk_count": len(chunks),
        "page_count": len({chunk.page for chunk in chunks}),
        "exact_match_chunk_ids": [
            chunk.chunk_id for chunk in chunks if reasons.get(chunk.chunk_id) == SelectionReason.HEURISTIC_MATCH
        ],
        "adjacent_chunk_ids": [
            chunk.chunk_id for chunk in chunks if reasons.get(chunk.chunk_id) == SelectionReason.ADJACENT_CONTEXT
        ],
        "exact_neighbor_token_ceiling": {
            "ceiling": TOKEN_CEILING,
            "tokens": ceiling_tokens,
            "chunk_ids": ceiling_ids,
        },
        "full_text_diagnostic": {
            "tokens": len(encoding.encode(full_text, disallowed_special=())),
            "sha256": hashlib.sha256(full_text.encode()).hexdigest(),
        },
        "top_k": top_k,
    }


def build_comparison(limit: int | None = None) -> dict:
    cases = load_cases()
    if limit is not None:
        cases = cases[:limit]
    encoding = tiktoken.get_encoding("gpt2")
    return {
        "schema_version": 1,
        "comparison_id": "phase3_page_aware_auto",
        "configuration": {
            "chunk_size_tokens": DEFAULT_CHUNK_SIZE,
            "chunk_overlap_tokens": DEFAULT_CHUNK_OVERLAP,
            "top_k_values": TOP_K_VALUES,
            "exact_neighbor_token_ceiling": TOKEN_CEILING,
            "retrieval_mode": RetrievalMode.AUTO,
            "legacy_baseline": "data/eval/phase3_legacy_retrieval.json",
        },
        "papers": [compare_paper(case, encoding) for case in cases],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build_comparison(args.limit), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text() != rendered:
            msg = f"Retrieval comparison is stale: {args.output}"
            raise SystemExit(msg)
        print("Phase 3 retrieval comparison is current.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
