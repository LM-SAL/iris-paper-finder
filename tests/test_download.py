from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pymupdf
import pytest
import requests

from iris_paper_llm.download import REQUEST_DELAY_SECONDS, candidate_pdf_urls, download_records, validate_pdf
from iris_paper_llm.evaluate import prepare_case_pdfs
from iris_paper_llm.jsonl import read_jsonl

ABSTRACT = (
    "We analyze spectral observations of chromospheric heating above sunspots with the Interface Region Imaging "
    "Spectrograph, comparing magnesium profiles against radiative transfer models, quantifying velocity "
    "oscillations, turbulent broadening, magnetic reconnection signatures and transition region brightenings."
)


def one_page_pdf(text: str = "") -> bytes:
    with pymupdf.open() as document:
        document.new_page().insert_textbox(pymupdf.Rect(72, 72, 500, 700), text)
        return document.tobytes()


class FakeResponse:
    def __init__(self, content: bytes = b"", status_code: int = 200, headers: dict | None = None) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/pdf"}
        self.url = "https://example.test/paper.pdf"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(self.status_code)

    def json(self) -> object:
        return json.loads(self.content)


class FakeSession:
    """Serves `responses` by URL and `content` as a PDF for any other URL."""

    def __init__(self, content: bytes, *, responses: dict[str, FakeResponse] | None = None) -> None:
        self.content = content
        self.responses = responses or {}
        self.calls = 0
        self.urls = []

    def get(self, url: str, **_kwargs: object) -> FakeResponse:
        self.calls += 1
        self.urls.append(url)
        return self.responses.get(url) or FakeResponse(self.content)


def test_download_is_atomic_validated_and_resumable() -> None:
    content = one_page_pdf()
    session = FakeSession(content)
    record = {"bibcode": "test-paper", "pdf_links": ["https://example.test/paper.pdf"]}
    with TemporaryDirectory() as directory:
        output = Path(directory)
        first = download_records([record], output, session=session)
        validate_pdf((output / "test-paper.pdf").read_bytes())
        assert first == {"downloaded": 1, "skipped": 0, "failed": 0, "manual_queue": 0}
        assert read_jsonl(output / "download_attempts.jsonl")[0]["status"] == "downloaded"

        second = download_records([record], output, session=session)
        assert second == {"downloaded": 0, "skipped": 1, "failed": 0, "manual_queue": 0}
        assert session.calls == 1


def test_candidate_url_order_and_doi_rules() -> None:
    record = {
        "bibcode": "2024JAHH...27....3E",
        "links_data": [
            {"access": "open", "type": "electr", "url": "https://doi.org/10.1007%2Fs11207-014-0485-y"},
            {"access": "open", "type": "gif", "url": "http://articles.adsabs.harvard.edu/full/2024JAHH...27....3E"},
            {"access": "open", "type": "pdf", "url": "http://academic.oup.test/paper.pdf"},
            {"access": "open", "type": "pdf", "url": "https://academic.oup.test/paper.pdf"},
            {"access": "open", "type": "electr", "url": "https://doi.org/10.3389%2Ffspas.2022.820116"},
            {"access": "open", "type": "electr", "url": "https://doi.org/10.1186%2Fs40562-018-0103-1"},
            {"access": "open", "type": "electr", "url": "https://doi.org/10.1038%2Fs41550-024-02321-9"},
            {"access": "open", "type": "electr", "url": "https://doi.org/10.3847%2F1538-4357%2Fad95f7"},
            {"access": "", "type": "pdf", "url": "https://closed.test/paper.pdf"},
            {"access": "open", "type": "preprint", "url": "http://arxiv.org/abs/2501.01234"},
        ],
    }
    assert candidate_pdf_urls(record) == [
        "https://arxiv.org/pdf/2501.01234",
        "https://academic.oup.test/paper.pdf",
        "https://link.springer.com/content/pdf/10.1007/s11207-014-0485-y.pdf",
        "https://www.frontiersin.org/articles/10.3389/fspas.2022.820116/pdf",
        "https://link.springer.com/content/pdf/10.1186/s40562-018-0103-1.pdf",
        "https://www.nature.com/articles/s41550-024-02321-9.pdf",
        "https://articles.adsabs.harvard.edu/pdf/2024JAHH...27....3E",
    ]


def test_requests_are_spaced() -> None:
    missing = "https://example.test/missing.pdf"
    session = FakeSession(one_page_pdf(), responses={missing: FakeResponse(status_code=404)})
    records = [
        {"bibcode": "first", "pdf_links": [missing, "https://example.test/paper.pdf"]},
        {"bibcode": "second", "pdf_links": ["https://example.test/paper.pdf"]},
    ]
    with TemporaryDirectory() as directory, patch("iris_paper_llm.download.sleep") as sleep:
        summary = download_records(records, Path(directory), session=session)
    assert summary == {"downloaded": 2, "skipped": 0, "failed": 0, "manual_queue": 0}
    assert session.calls == 3
    assert [call.args for call in sleep.call_args_list] == [(REQUEST_DELAY_SECONDS,)] * 2


def test_manual_queue_is_deduplicated() -> None:
    session = FakeSession(one_page_pdf())
    with TemporaryDirectory() as directory:
        output = Path(directory)
        record = {"bibcode": "missing-paper", "links_data": []}
        expected = {"downloaded": 0, "skipped": 0, "failed": 1, "manual_queue": 1}
        assert download_records([record], output, session=session) == expected
        assert download_records([record], output, session=session) == expected
        queue = read_jsonl(output / "manual_downloads.jsonl")
        assert len(queue) == 1
        assert queue[0]["category"] == "permanent"
        assert queue[0]["errors"] == ["no open-access PDF URL was found in ADS (no DOI for OpenAlex)"]
        assert session.calls == 0


@pytest.mark.parametrize(
    "pmc_location",
    [
        "https://www.ncbi.nlm.nih.gov/pmc/articles/123",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
        "PMC123",
    ],
)
def test_openalex_copies_follow_failed_ads_links(pmc_location: str) -> None:
    publisher = "https://publisher.test/paper.pdf"
    openalex = "https://api.openalex.org/works/doi:10.1234/paper"
    repository = "https://repository.test/paper.pdf"
    pmc = "https://pmc-oa-opendata.s3.amazonaws.com/PMC123.1/PMC123.1.pdf"
    work = {
        "ids": {"pmcid": pmc_location if pmc_location == "PMC123" else None},
        "locations": [
            {"is_oa": True, "pdf_url": "https://doi.org/10.1234/paper"},  # can be a publisher placeholder PDF
            {"is_oa": True, "pdf_url": "http://publisher.test/paper.pdf"},  # already tried
            {"is_oa": False, "pdf_url": "https://closed.test/paper.pdf"},
            {"is_oa": True, "pdf_url": repository},
            {"is_oa": True, "pdf_url": None, "landing_page_url": pmc_location if "://" in pmc_location else None},
        ],
    }
    cloudflare = {"content-type": "text/html", "server": "cloudflare"}
    challenge = FakeResponse(b"<title>Just a moment...</title>", 403, cloudflare)
    session = FakeSession(
        one_page_pdf(),
        responses={
            publisher: challenge,
            openalex: FakeResponse(json.dumps(work).encode(), headers={"content-type": "application/json"}),
            repository: FakeResponse(status_code=404),
        },
    )
    record = {"bibcode": "oa-paper", "doi": ["10.1234/paper"], "pdf_links": [publisher]}
    with TemporaryDirectory() as directory, patch("iris_paper_llm.download.sleep"):
        output = Path(directory)
        assert download_records([record], output, session=session)["downloaded"] == 1
        attempts = read_jsonl(output / "download_attempts.jsonl")
    assert session.urls == [publisher, openalex, repository, pmc]
    assert [attempt["method"] for attempt in attempts] == ["http", "openalex", "openalex"]
    assert attempts[0]["error"] == "HTTP 403 (bot protection: Cloudflare)"


def test_failed_openalex_lookup_is_transient_and_unknown_doi_is_not() -> None:
    throttled = "https://api.openalex.org/works/doi:10.1234/throttled"
    unknown = "https://api.openalex.org/works/doi:10.1234/unknown"
    session = FakeSession(
        b"", responses={throttled: FakeResponse(status_code=429), unknown: FakeResponse(status_code=404)}
    )
    records = [
        {"bibcode": "throttled", "doi": ["10.1234/throttled"]},
        {"bibcode": "unknown", "doi": ["10.1234/unknown"]},
    ]
    with TemporaryDirectory() as directory, patch("iris_paper_llm.download.sleep"):
        output = Path(directory)
        download_records(records, output, session=session)
        (attempt,) = read_jsonl(output / "download_attempts.jsonl")
        throttled_entry, unknown_entry = read_jsonl(output / "manual_downloads.jsonl")
    assert (attempt["url"], attempt["category"]) == (throttled, "transient")
    assert throttled_entry["category"] == "transient"
    assert unknown_entry["category"] == "permanent"
    assert unknown_entry["errors"] == ["no open-access PDF URL was found in ADS or OpenAlex"]


def test_bot_protection_is_queued_as_blocked_and_skips_its_host() -> None:
    url = "https://www.aanda.test/paper.pdf"
    blocked = FakeResponse(b"<html></html>", 403, {"content-type": "text/html", "x-datadome": "protected"})
    session = FakeSession(b"", responses={url: blocked})
    records = [{"bibcode": "blocked", "pdf_links": [url]}, {"bibcode": "same-host", "pdf_links": [f"{url}?v=2"]}]
    with TemporaryDirectory() as directory:
        output = Path(directory)
        download_records(records, output, session=session)
        first, second = read_jsonl(output / "manual_downloads.jsonl")
    assert session.urls == [url]
    assert first["category"] == second["category"] == "blocked"
    assert first["errors"] == ["HTTP 403 (bot protection: DataDome)"]
    assert second["errors"] == ["skipped: www.aanda.test already blocked a request in this run"]


def test_pdf_of_another_paper_or_without_text_is_rejected() -> None:
    wrong = "https://example.test/wrong.pdf"
    scan = "https://example.test/scan.pdf"
    right = one_page_pdf(f"Title. {ABSTRACT}")
    responses = {
        wrong: FakeResponse(one_page_pdf("Galaxy clusters at high redshift.")),
        scan: FakeResponse(one_page_pdf()),
    }
    session = FakeSession(right, responses=responses)
    record = {"bibcode": "paper", "abstract": ABSTRACT, "pdf_links": [wrong, scan, "https://example.test/right.pdf"]}
    with TemporaryDirectory() as directory, patch("iris_paper_llm.download.sleep"):
        output = Path(directory)
        assert download_records([record], output, session=session)["downloaded"] == 1
        assert (output / "paper.pdf").read_bytes() == right
        attempts = read_jsonl(output / "download_attempts.jsonl")
    assert [attempt["error"] for attempt in attempts] == [
        "PDF text does not match the ADS abstract (overlap 0.00)",
        "PDF has no extractable text",
        None,
    ]


@pytest.mark.parametrize("text", ["Galaxy clusters at high redshift.", ""])
def test_existing_wrong_or_scanned_pdf_is_preserved_and_replaced(text: str) -> None:
    existing = one_page_pdf(text)
    right = one_page_pdf(ABSTRACT)
    session = FakeSession(right)
    record = {"bibcode": "paper", "abstract": ABSTRACT}
    with TemporaryDirectory() as directory:
        output = Path(directory)
        target = output / "paper.pdf"
        target.write_bytes(existing)
        assert download_records([record], output, session=session) == {
            "downloaded": 0,
            "skipped": 0,
            "failed": 1,
            "manual_queue": 1,
        }
        assert not target.exists()
        (rejected,) = output.glob("*.rejected")
        assert rejected.read_bytes() == existing
        assert session.calls == 0

        record["pdf_links"] = ["https://example.test/paper.pdf"]
        assert download_records([record], output, session=session)["downloaded"] == 1
        assert target.read_bytes() == right
        assert download_records([record], output, session=session)["skipped"] == 1
        assert session.calls == 1
        assert read_jsonl(output / "manual_downloads.jsonl") == []


def test_reviewed_pdf_preparation_uses_manifest_checksum() -> None:
    content = one_page_pdf()
    checksum = hashlib.sha256(content).hexdigest()
    with TemporaryDirectory() as directory:
        target = Path(directory) / "pdfs/TEST.pdf"
        cases = Path(directory) / "cases.jsonl"
        links = ["https://example.test/paper.pdf"]
        case = {"id": "reviewed", "path": str(target), "pdf_sha256": checksum, "pdf_links": links, "bibcode": "TEST"}
        cases.write_text(json.dumps(case) + "\n")
        summary = prepare_case_pdfs(cases, session=FakeSession(content))
        assert summary == {"downloaded": 1, "skipped": 0, "failed": 0, "manual_queue": 0}
        assert hashlib.sha256(target.read_bytes()).hexdigest() == checksum

        conflict = one_page_pdf("A different PDF that must be preserved.")
        target.write_bytes(conflict)
        summary = prepare_case_pdfs(cases, session=FakeSession(content))
        assert summary == {"downloaded": 0, "skipped": 0, "failed": 1, "manual_queue": 1}
        assert target.read_bytes() == conflict
