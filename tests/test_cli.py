from __future__ import annotations

# ruff: noqa: S101

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import fitz

from paper_data_linking.ads import iris_query
from paper_data_linking.classify import DEFAULT_MODEL, classify_paths
from paper_data_linking.evaluate import prepare_case_pdfs, write_report
from paper_data_linking.models import (
    Decision,
    Evidence,
    IRISAspect,
    IRISClassification,
    RetrievalMode,
    SyntheticConnection,
)


def paper_pdf(label: str) -> bytes:
    with fitz.open() as document:
        page = document.new_page()
        page.insert_textbox(
            fitz.Rect(72, 72, 500, 700),
            f"{label}. We analyze IRIS observations. " * 20,
        )
        return document.tobytes()


class FakeResponse:
    status_code = 200
    url = "https://example.test/paper.pdf"

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.headers = {"content-type": "application/pdf"}


class FakeSession:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def get(self, *_args, **_kwargs) -> FakeResponse:
        return FakeResponse(self.content)


class FakeEmbedder:
    def create_embeddings(self, docs: list) -> None:
        self.docs = docs

    def get_relevant_docs(self, _query: str, kwargs: dict | None = None, n_results: int = 10):
        docs = self.docs
        if kwargs and kwargs.get("where"):
            docs = [doc for doc in docs if doc.metadata["is_candidate"] == 1]
        docs = docs[:n_results]
        return docs, [float(index) for index in range(len(docs))]


class CharacterEncoding:
    @staticmethod
    def encode(text: str, **_kwargs: object) -> list[int]:
        return [ord(character) for character in text]

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class FakeResponses:
    def parse(self, **kwargs: object) -> object:
        context = kwargs["input"][1]["content"]
        page = int(context.split("page=", 1)[1].split(" ", 1)[0])
        chunk_id = context.split("chunk_id=", 1)[1].split("]", 1)[0]
        classification = IRISClassification(
            observational_use=Decision.YES,
            synthetic_use=Decision.NO,
            synthetic_connection=SyntheticConnection.NOT_APPLICABLE,
            review_only=Decision.NO,
            iris_mission_mentioned=True,
            aspects=[IRISAspect.TELESCOPE],
            observational_evidence=[Evidence(page=page, chunk_id=chunk_id, reason="IRIS data are analyzed.")],
            synthetic_evidence=[],
            review_evidence=[],
        )
        return SimpleNamespace(
            id="response-test",
            model="resolved-test-model",
            output=[],
            output_parsed=classification,
            status="completed",
            usage=None,
        )


def test_one_pdf_and_directory_resume() -> None:
    client = SimpleNamespace(responses=FakeResponses())
    repository = Path(__file__).resolve().parents[1]
    first = repository / "test_pdfs/positive/iris_obs_paper.pdf"
    second = repository / "test_pdfs/negative/talks_about_iris_only.pdf"
    with TemporaryDirectory() as directory:
        root = Path(directory)
        pdfs = root / "pdfs"
        pdfs.mkdir()
        shutil.copyfile(first, pdfs / first.name)
        shutil.copyfile(second, pdfs / second.name)
        output = root / "results.jsonl"

        with patch("tiktoken.get_encoding", return_value=CharacterEncoding()):
            first_summary = classify_paths(first, output, client=client, embedder=FakeEmbedder())
            directory_summary = classify_paths(pdfs, output, client=client, embedder=FakeEmbedder())

        assert first_summary == {"positive": 1, "negative": 0, "uncertain": 0, "failed": 0, "skipped": 0}
        assert directory_summary == {"positive": 1, "negative": 0, "uncertain": 0, "failed": 0, "skipped": 1}
        assert len(output.read_text().splitlines()) == 2
        report = write_report(
            output,
            model=DEFAULT_MODEL,
            retrieval_mode=RetrievalMode.AUTO,
            top_k=20,
        )
        assert report["positive"] == 2
        assert output.with_suffix(".md").is_file()


def test_reviewed_pdf_preparation_uses_manifest_checksum() -> None:
    content = paper_pdf("Reviewed")
    checksum = hashlib.sha256(content).hexdigest()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        cases = root / "cases.jsonl"
        case = {
            "id": "reviewed",
            "path": "pdfs/TEST.pdf",
            "pdf_sha256": checksum,
            "pdf_links": ["https://example.test/paper.pdf"],
            "bibcode": "TEST",
        }
        cases.write_text(json.dumps(case) + "\n")
        summary = prepare_case_pdfs(cases, base_dir=root, session=FakeSession(content))
        assert summary == {"downloaded": 1, "skipped": 0, "failed": 0, "manual_queue": 0}
        assert hashlib.sha256((root / "pdfs/TEST.pdf").read_bytes()).hexdigest() == checksum


def test_standard_query_is_scoped_to_year() -> None:
    query = iris_query(2025)
    assert "2014SoPh..289.2733D" in query
    assert "pubdate:[2025-01 TO 2025-12]" in query


if __name__ == "__main__":
    test_one_pdf_and_directory_resume()
    test_reviewed_pdf_preparation_uses_manifest_checksum()
    test_standard_query_is_scoped_to_year()
