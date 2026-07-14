"""Page-aware IRIS candidate selection and local ONNX retrieval."""

# ruff: noqa: S101

from __future__ import annotations

import re
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from paper_data_linking.iris import IRIS_SLIT_JAW_CHANNELS_ANGSTROM, IRIS_SPECTROGRAPH_WINDOWS_ANGSTROM
from paper_data_linking.models import (
    PaperChunk,
    RetrievalMode,
    RetrievalResult,
    RetrievedChunk,
    SelectionReason,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 20
IRIS_RETRIEVAL_QUERY = (
    "Does this paper use observational IRIS data or create synthetic observables "
    "within an IRIS spectrograph window or slit-jaw channel?"
)

REFERENCE_HEADING = re.compile(
    r"(?m)^[ \t]*(?:\d+(?:\.\d+)*[.)]?[ \t]+)?"
    r"(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|WORKS CITED|Works Cited|"
    r"LITERATURE CITED|Literature Cited|REFERENCE LIST|Reference List)\b"
)
IRIS_NAME = re.compile(r"\b(?:Interface Region Imaging Spectrograph|IRIS)\b", re.IGNORECASE)
IRIS_INSTRUMENT = re.compile(
    r"\b(?:slit[- ]jaw(?: imager)?|SJI|IRIS[ -](?:spectrograph|telescope))\b",
    re.IGNORECASE,
)
WAVELENGTH = re.compile(
    r"(?<!\d)(\d{4}(?:\.\d+)?)\s*(?:Å|Å|A\b|angstroms?\b)",
    re.IGNORECASE,
)
SJI_CHANNEL = re.compile(
    rf"\bSJI[ -]?(?:{'|'.join(map(str, IRIS_SLIT_JAW_CHANNELS_ANGSTROM))})\b",
    re.IGNORECASE,
)


@dataclass
class _EmbeddingDocument:
    page_content: str
    metadata: dict[str, str | int]


def _usable_text(pages: Iterable[str]) -> bool:
    return sum(character.isalpha() for page in pages for character in page) >= 100


def _ocr_pages(path: Path) -> list[str]:
    from pdf2image import convert_from_path  # noqa: PLC0415
    from pytesseract import image_to_string  # noqa: PLC0415

    return [image_to_string(image) for image in convert_from_path(path)]


def extract_pdf_pages(path: Path | str, *, ocr_fallback: bool = True) -> tuple[list[str], bool]:
    """Extract each PDF page, importing OCR dependencies only when needed."""
    import fitz  # noqa: PLC0415

    path = Path(path)
    with fitz.open(path) as document:
        pages = [page.get_text(sort=True) for page in document]
    if _usable_text(pages):
        return pages, False
    if not ocr_fallback:
        msg = f"No usable embedded text in {path}"
        raise ValueError(msg)
    pages = _ocr_pages(path)
    if not _usable_text(pages):
        msg = f"OCR produced no usable text for {path}"
        raise ValueError(msg)
    return pages, True


def remove_reference_section(pages: list[str]) -> list[str]:
    """Drop text from the last standalone references heading onward."""
    headings = [
        (page_index, match.start())
        for page_index, page in enumerate(pages)
        for match in REFERENCE_HEADING.finditer(page)
    ]
    if not headings:
        return pages
    page_index, offset = headings[-1]
    prefix = pages[page_index][:offset].rstrip()
    return pages[:page_index] + ([prefix] if prefix else [])


def chunk_pages(
    pages: list[str],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[PaperChunk]:
    """Create stable token chunks without crossing page boundaries."""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        msg = "chunk_size must be positive and chunk_overlap must be smaller"
        raise ValueError(msg)
    import tiktoken  # noqa: PLC0415

    encoding = tiktoken.get_encoding("gpt2")
    chunks = []
    step = chunk_size - chunk_overlap
    for page_number, text in enumerate(pages, start=1):
        tokens = encoding.encode(text, disallowed_special=())
        for page_chunk_index, start in enumerate(range(0, len(tokens), step)):
            chunk_tokens = tokens[start : start + chunk_size]
            chunk_text = encoding.decode(chunk_tokens).strip()
            if chunk_text:
                chunks.append(
                    PaperChunk(
                        page=page_number,
                        chunk_id=f"p{page_number:04d}-c{page_chunk_index:04d}",
                        text=chunk_text,
                    )
                )
            if start + chunk_size >= len(tokens):
                break
    return chunks


def chunk_pdf(
    path: Path | str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ocr_fallback: bool = True,
) -> tuple[list[PaperChunk], bool]:
    pages, used_ocr = extract_pdf_pages(path, ocr_fallback=ocr_fallback)
    pages = remove_reference_section(pages)
    return chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap), used_ocr


def is_exact_iris_match(text: str) -> bool:
    if IRIS_NAME.search(text) or IRIS_INSTRUMENT.search(text) or SJI_CHANNEL.search(text):
        return True
    for match in WAVELENGTH.finditer(text):
        wavelength = float(match.group(1))
        if any(lower <= wavelength <= upper for lower, upper in IRIS_SPECTROGRAPH_WINDOWS_ANGSTROM.values()):
            return True
    return False


def select_candidates(chunks: list[PaperChunk]) -> dict[str, SelectionReason]:
    exact_indices = {index for index, chunk in enumerate(chunks) if is_exact_iris_match(chunk.text)}
    reasons = {chunks[index].chunk_id: SelectionReason.HEURISTIC_MATCH for index in exact_indices}
    for index in exact_indices:
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(chunks):
                reasons.setdefault(chunks[neighbor].chunk_id, SelectionReason.ADJACENT_CONTEXT)
    return reasons


def _query(embedder, *, n_results: int, where: dict | None = None) -> list[tuple[str, float]]:
    if n_results == 0:
        return []
    kwargs = {"where": where} if where is not None else None
    docs, distances = embedder.get_relevant_docs(IRIS_RETRIEVAL_QUERY, kwargs, n_results=n_results)
    return [(str(doc.metadata["chunk_id"]), float(distance)) for doc, distance in zip(docs, distances, strict=True)]


def retrieve_chunks(
    chunks: list[PaperChunk],
    *,
    mode: RetrievalMode = RetrievalMode.AUTO,
    top_k: int = DEFAULT_TOP_K,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedder=None,
) -> RetrievalResult:
    """Rank chunks locally, prioritizing deterministic candidates in auto mode."""
    if not chunks:
        msg = "At least one chunk is required"
        raise ValueError(msg)
    if top_k <= 0:
        msg = "top_k must be positive"
        raise ValueError(msg)
    mode = RetrievalMode(mode)
    reasons = select_candidates(chunks)
    if embedder is None:
        from paper_data_linking.process.embedders import ONNXEmbedder  # noqa: PLC0415

        embedder = ONNXEmbedder()

    documents = [
        _EmbeddingDocument(
            page_content=chunk.text,
            metadata={
                "page": chunk.page or -1,
                "position": position,
                "chunk_id": chunk.chunk_id,
                "is_candidate": int(chunk.chunk_id in reasons),
            },
        )
        for position, chunk in enumerate(chunks)
    ]
    embedder.create_embeddings(documents)

    selected: list[tuple[str, float, SelectionReason]] = []
    candidate_count = len(reasons)
    if mode in {RetrievalMode.AUTO, RetrievalMode.HEURISTIC} and candidate_count:
        candidate_ranked = _query(
            embedder,
            n_results=min(top_k, candidate_count),
            where={"is_candidate": 1},
        )
        selected.extend((chunk_id, distance, reasons[chunk_id]) for chunk_id, distance in candidate_ranked)

    if mode == RetrievalMode.ALL or (mode == RetrievalMode.AUTO and len(selected) < min(top_k, len(chunks))):
        global_ranked = _query(
            embedder,
            n_results=min(len(chunks), top_k + len(selected)),
        )
        selected_ids = {chunk_id for chunk_id, _distance, _reason in selected}
        for chunk_id, distance in global_ranked:
            if chunk_id not in selected_ids:
                selected.append((chunk_id, distance, SelectionReason.GLOBAL_FALLBACK))
                selected_ids.add(chunk_id)
            if len(selected) == min(top_k, len(chunks)):
                break

    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    retrieved = [
        RetrievedChunk(
            **chunks_by_id[chunk_id].model_dump(),
            distance=distance,
            reason=reason,
        )
        for chunk_id, distance, reason in selected
    ]
    sent_ids = [chunk.chunk_id for chunk in retrieved]
    exact_ids = [chunk.chunk_id for chunk in chunks if reasons.get(chunk.chunk_id) == SelectionReason.HEURISTIC_MATCH]
    adjacent_ids = [
        chunk.chunk_id for chunk in chunks if reasons.get(chunk.chunk_id) == SelectionReason.ADJACENT_CONTEXT
    ]
    return RetrievalResult(
        mode=mode,
        query=IRIS_RETRIEVAL_QUERY,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        exact_match_chunk_ids=exact_ids,
        adjacent_chunk_ids=adjacent_ids,
        excluded_chunk_ids=[chunk.chunk_id for chunk in chunks if chunk.chunk_id not in sent_ids],
        sent_chunk_ids=sent_ids,
        selected=retrieved,
    )


def _self_check() -> None:
    class FakeEmbedder:
        def create_embeddings(self, docs) -> None:
            self.docs = docs

        def get_relevant_docs(self, _query, kwargs=None, n_results=10):
            docs = self.docs
            if kwargs and kwargs.get("where"):
                docs = [doc for doc in docs if doc.metadata["is_candidate"] == 1]
            docs = sorted(docs, key=lambda doc: doc.metadata["position"], reverse=True)[:n_results]
            return docs, [float(index) / 10 for index in range(len(docs))]

    chunks = [
        PaperChunk(page=1, chunk_id="p1-c0", text="Introduction without a mission name."),
        PaperChunk(page=1, chunk_id="p1-c1", text="We analyze IRIS observations."),
        PaperChunk(page=2, chunk_id="p2-c0", text="The result is shown here."),
        PaperChunk(page=2, chunk_id="p2-c1", text="Unrelated appendix."),
    ]
    result = retrieve_chunks(chunks, top_k=4, embedder=FakeEmbedder())
    assert result.sent_chunk_ids == ["p2-c0", "p1-c1", "p1-c0", "p2-c1"]
    assert result.selected[-1].reason == SelectionReason.GLOBAL_FALLBACK
    assert is_exact_iris_match("Synthetic Si IV at 1402.8 Angstrom")
    assert not is_exact_iris_match("Synthetic Fe XII at 195 Angstrom")
    assert remove_reference_section(["Methods\nsources of data", "Results", "References\nCitation"])[-1] == "Results"
    assert remove_reference_section(["Introduction\nreferences therein", "Results"])[-1] == "Results"

    no_match = retrieve_chunks(
        [PaperChunk(page=1, chunk_id="only", text="No relevant term here.")],
        embedder=FakeEmbedder(),
    )
    assert no_match.sent_chunk_ids == ["only"]

    heuristic = retrieve_chunks(
        chunks,
        mode=RetrievalMode.HEURISTIC,
        top_k=4,
        embedder=FakeEmbedder(),
    )
    assert len(heuristic.selected) == 3
    assert SelectionReason.GLOBAL_FALLBACK not in {chunk.reason for chunk in heuristic.selected}

    all_chunks = retrieve_chunks(
        chunks,
        mode=RetrievalMode.ALL,
        top_k=2,
        embedder=FakeEmbedder(),
    )
    assert all(chunk.reason == SelectionReason.GLOBAL_FALLBACK for chunk in all_chunks.selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, nargs="?")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--retrieval-mode", choices=RetrievalMode, default=RetrievalMode.AUTO, type=RetrievalMode)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if args.pdf is None:
        parser.error("pdf is required unless --self-check is used")
    chunks, used_ocr = chunk_pdf(
        args.pdf,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        ocr_fallback=not args.no_ocr,
    )
    result = retrieve_chunks(
        chunks,
        mode=args.retrieval_mode,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(  # noqa: T201
        json.dumps(
            {"pdf": str(args.pdf), "used_ocr": used_ocr, "retrieval": result.model_dump(mode="json")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
