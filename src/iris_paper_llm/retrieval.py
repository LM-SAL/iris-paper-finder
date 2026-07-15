"""Page-aware IRIS candidate selection and local ONNX retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from iris_paper_llm.iris import IRIS_SLIT_JAW_CHANNELS_ANGSTROM, IRIS_SPECTROGRAPH_WINDOWS_ANGSTROM
from iris_paper_llm.models import (
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
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models/onnx"
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


class _MiniLMModel:
    """Local all-MiniLM-L6-v2 inference matching the frozen Chroma baseline."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        model_dir = Path(model_dir)
        model_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        missing = [str(path) for path in (model_path, tokenizer_path) if not path.is_file()]
        if missing:
            msg = f"Missing ONNX model files: {', '.join(missing)}. Run `python scripts/setup_onnx.py`."
            raise FileNotFoundError(msg)

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        batches = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer.encode_batch(texts[start : start + batch_size])
            input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
            attention_mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
            output = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": np.zeros_like(input_ids),
                },
            )[0]
            mask = attention_mask[..., None]
            embeddings = (output * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            batches.append((embeddings / np.clip(norms, 1e-12, None)).astype(np.float32))
        return np.concatenate(batches)


@cache
def _load_model(model_dir: str) -> _MiniLMModel:
    return _MiniLMModel(model_dir)


class ONNXEmbedder:
    """Keep documents and normalized embeddings in memory for one paper."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        self.model = _load_model(str(Path(model_dir).resolve()))
        self.docs = []
        self.embeddings = np.empty((0, 384), dtype=np.float32)

    def create_embeddings(self, docs: list) -> None:
        self.docs = list(docs)
        self.embeddings = self.model.encode([doc.page_content for doc in self.docs])

    def get_relevant_docs(self, query: str, kwargs: dict | None = None, n_results: int = 10):
        if n_results <= 0:
            return [], []
        kwargs = kwargs or {}
        unsupported = set(kwargs) - {"where"}
        if unsupported:
            msg = f"Unsupported ranking options: {sorted(unsupported)}"
            raise ValueError(msg)
        where = kwargs.get("where", {})
        indices = [
            index
            for index, doc in enumerate(self.docs)
            if all(doc.metadata.get(key) == value for key, value in where.items())
        ]
        if not indices:
            return [], []
        query_embedding = self.model.encode([query])[0]
        distances = 1.0 - self.embeddings[indices] @ query_embedding
        order = np.argsort(distances, kind="stable")[:n_results]
        return (
            [self.docs[indices[index]] for index in order],
            [float(distances[index]) for index in order],
        )


def _usable_text(pages: Iterable[str]) -> bool:
    return sum(character.isalpha() for page in pages for character in page) >= 100


def _ocr_pages(path: Path) -> list[str]:
    from pdf2image import convert_from_path
    from pytesseract import image_to_string

    return [image_to_string(image) for image in convert_from_path(path)]


def extract_pdf_pages(path: Path | str, *, ocr_fallback: bool = True) -> tuple[list[str], bool]:
    """Extract each PDF page, importing OCR dependencies only when needed."""
    import fitz

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
    import tiktoken

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
    exact_ids = [chunk.chunk_id for chunk in chunks if reasons.get(chunk.chunk_id) == SelectionReason.HEURISTIC_MATCH]
    adjacent_ids = [
        chunk.chunk_id for chunk in chunks if reasons.get(chunk.chunk_id) == SelectionReason.ADJACENT_CONTEXT
    ]
    return RetrievalResult(
        exact_match_chunk_ids=exact_ids,
        adjacent_chunk_ids=adjacent_ids,
        selected=retrieved,
    )
