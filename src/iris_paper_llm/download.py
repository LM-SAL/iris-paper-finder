"""Validated, resumable PDF downloads from ADS links, with open copies found through OpenAlex."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote, urlparse

import pymupdf
import requests

from iris_paper_llm.jsonl import append_jsonl, atomic_write, read_jsonl, write_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

DEFAULT_TIMEOUT_SECONDS = 60.0
REQUEST_DELAY_SECONDS = 3.0
DEFAULT_HEADERS = {
    "User-Agent": "iris-papers/0.2 (+https://github.com/LM-SAL/iris-paper-finder)",
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
}
TRANSIENT_HTTP_STATUS = {408, 425, 429}
OPENALEX_WORK_URL = "https://api.openalex.org/works/doi:{doi}"
PMC_PDF_URL = "https://pmc-oa-opendata.s3.amazonaws.com/{pmcid}.1/{pmcid}.1.pdf"
# The wrong-paper check needs this many distinct abstract words of 5+ ASCII letters (so it skips Chinese abstracts).
# Of 1069 correct PDFs the first 3 pages held at least 0.64 of them (a preprint with a rewritten abstract);
# a same-title conference paper that ADS merged into the record held 0.46.
MIN_ABSTRACT_WORDS = 20
MIN_ABSTRACT_OVERLAP = 0.6
logger = logging.getLogger(__name__)


# Publishers whose open PDFs sit at a URL derived from the DOI.
DOI_PDF_URLS = {
    "10.1007/": "https://link.springer.com/content/pdf/{doi}.pdf",
    "10.1186/": "https://link.springer.com/content/pdf/{doi}.pdf",
    "10.1038/": "https://www.nature.com/articles/{suffix}.pdf",
    "10.3389/": "https://www.frontiersin.org/articles/{doi}/pdf",
}


def candidate_pdf_urls(record: Mapping[str, Any]) -> list[str]:
    """Return open arXiv links, publisher PDF links, DOI-derived PDF URLs, then the ADS scan, once each (https kept)."""
    arxiv = []
    publisher = [str(url) for url in record.get("pdf_links", []) or []]
    from_doi = []
    scans = []
    for link in record.get("links_data", []) or []:
        if not isinstance(link, dict) or link.get("access") != "open":
            continue
        url = str(link.get("url", ""))
        link_type = link.get("type")
        if "//articles.adsabs.harvard.edu/" in url:
            scans.append(f"https://articles.adsabs.harvard.edu/pdf/{record['bibcode']}")
        elif link_type == "preprint" and "arxiv.org/" in url:
            arxiv.append(url.replace("http://", "https://").replace("/abs/", "/pdf/"))
        elif link_type == "pdf":
            publisher.append(url)
        elif link_type == "electr":
            doi = unquote(url).split("doi.org/", 1)[-1]
            from_doi.extend(
                template.format(doi=doi, suffix=doi.split("/", 1)[-1])
                for prefix, template in DOI_PDF_URLS.items()
                if doi.startswith(prefix)
            )
    unique = {}
    for url in arxiv + publisher + from_doi + scans:
        key = url.split("://", 1)[-1]
        if url and (key not in unique or url.startswith("https://")):
            unique[key] = url
    return list(unique.values())


def _doi(record: Mapping[str, Any]) -> str | None:
    """Return the record's first DOI, else the DOI of an electr link."""
    if dois := record.get("doi"):
        return str(dois[0])
    for link in record.get("links_data", []) or []:
        url = unquote(str(link.get("url", ""))) if isinstance(link, dict) and link.get("type") == "electr" else ""
        if "doi.org/" in url:
            return url.split("doi.org/", 1)[1]
    return None


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", unicodedata.normalize("NFKC", text).lower()))


def abstract_overlap(abstract: str, text: str) -> float | None:
    """Return the share of the abstract's long words found in the text, or None if it has too few to judge."""
    words = {word for word in _words(abstract) if len(word) >= 5}
    if len(words) < MIN_ABSTRACT_WORDS:
        return None
    return len(words & _words(text)) / len(words)


def validate_pdf(content: bytes, expected_sha256: str | None = None, abstract: str | None = None) -> None:
    """Reject non-PDF, unreadable, empty, or checksum-mismatched content.

    With an abstract, also reject first pages that have no text or text that does not match it.
    """
    if not content.startswith(b"%PDF-"):
        msg = "content does not start with a PDF signature"
        raise ValueError(msg)
    if expected_sha256 and hashlib.sha256(content).hexdigest() != expected_sha256:
        msg = "PDF checksum does not match the expected SHA-256"
        raise ValueError(msg)
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            page_count = document.page_count
            if page_count:
                document.load_page(0)
            text = " ".join(page.get_text() for page in document.pages(stop=3)) if abstract else ""
    except Exception as error:
        msg = "PDF cannot be opened or its first page cannot be read"
        raise ValueError(msg) from error
    if page_count < 1:
        msg = "PDF has no pages"
        raise ValueError(msg)
    if abstract and not text.strip():
        msg = "PDF has no extractable text"
        raise ValueError(msg)
    overlap = abstract_overlap(abstract, text) if abstract else None
    if overlap is not None and overlap < MIN_ABSTRACT_OVERLAP:
        msg = f"PDF text does not match the ADS abstract (overlap {overlap:.2f})"
        raise ValueError(msg)


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _attempt(
    *,
    bibcode: str,
    url: str,
    method: str,
    status: str,
    category: str | None = None,
    error: str | None = None,
    final_url: str | None = None,
) -> dict:
    return {
        "timestamp": _now(),
        "bibcode": bibcode,
        "url": url,
        "final_url": final_url,
        "method": method,
        "status": status,
        "category": category,
        "error": error,
    }


@dataclass
class _Fetcher:
    """One run's HTTP state; consecutive requests are REQUEST_DELAY_SECONDS apart."""

    session: requests.Session
    timeout: float
    requested: bool = False
    blocked_hosts: set[str] = field(default_factory=set)

    def get(self, url: str) -> requests.Response:
        if self.requested:
            sleep(REQUEST_DELAY_SECONDS)
        self.requested = True
        return self.session.get(url, headers=DEFAULT_HEADERS, timeout=self.timeout)


def _bot_protection(response: requests.Response) -> str | None:
    headers = response.headers
    server = headers.get("server", "").lower()
    body = response.content
    if "x-datadome" in headers or b"captcha-delivery" in body:
        return "DataDome"
    # Every response through Cloudflare (doi.org too) has its server and cf-ray headers, so those count only on a 403.
    challenge = "cf-mitigated" in headers or b"<title>Just a moment" in body
    if challenge or (server == "cloudflare" and response.status_code == 403):
        return "Cloudflare"
    # The Akamai "Access Denied" page links errors.edgesuite.net; S3 also answers "Access Denied".
    if server.startswith("akamaighost") or b"edgesuite" in body:
        return "Akamai"
    return None


def _response_failure(response: requests.Response) -> tuple[str, str] | None:
    """Return (category, error) for a response that is not a PDF, else None."""
    status = response.status_code
    content_type = response.headers.get("content-type", "").lower()
    pdf_type = not content_type or "pdf" in content_type or "octet-stream" in content_type
    if 200 <= status < 300:
        if pdf_type or response.content.startswith(b"%PDF-"):
            return None
        error = f"unexpected content type: {content_type}"
    else:
        error = f"HTTP {status}"
    if protection := _bot_protection(response):
        return "blocked", f"{error} (bot protection: {protection})"
    if status in TRANSIENT_HTTP_STATUS or status >= 500:
        return "transient", error
    return "permanent", error


def _download_url(fetcher: _Fetcher, job: dict, url: str, method: str) -> tuple[bytes | None, dict]:
    """Attempt one direct HTTP download, unless its host blocked an earlier one, and return its durable record."""
    attempt = partial(_attempt, bibcode=job["bibcode"], url=url, method=method)
    host = urlparse(url).netloc
    if host in fetcher.blocked_hosts:
        return None, attempt(
            status="skipped", category="blocked", error=f"skipped: {host} already blocked a request in this run"
        )
    try:
        response = fetcher.get(url)
    except requests.RequestException as error:
        return None, attempt(status="failed", category="transient", error=str(error) or type(error).__name__)
    failure = _response_failure(response)
    if failure is None:
        try:
            validate_pdf(response.content, job["expected_sha256"], job["abstract"])
        except ValueError as error:
            failure = "permanent", str(error)
    if failure is not None:
        category, error = failure
        if category == "blocked":
            fetcher.blocked_hosts.add(host)
        return None, attempt(final_url=response.url, status="failed", category=category, error=error)
    return response.content, attempt(final_url=response.url, status="downloaded")


def _open_access_urls(fetcher: _Fetcher, job: dict) -> tuple[list[str], dict | None]:
    """Return the PDF URLs of OpenAlex's open locations for the DOI, then its PMC open-data copy.

    The second value records a failed lookup; OpenAlex not knowing the DOI (HTTP 404) is not a failure.
    """
    url = OPENALEX_WORK_URL.format(doi=quote(job["doi"]))
    try:
        response = fetcher.get(url)
        if response.status_code == 404:
            return [], None
        response.raise_for_status()
        work = response.json()
        locations = work.get("locations") or []
        # doi.org pdf_urls can resolve to a publisher placeholder PDF that validate_pdf accepts.
        urls = [
            location["pdf_url"]
            for location in locations
            if location.get("is_oa")
            and location.get("pdf_url")
            and urlparse(location["pdf_url"]).hostname not in {"doi.org", "dx.doi.org"}
        ]
        # OpenAlex responses currently carry the PMCID only in a location's landing page URL.
        pages = [(work.get("ids") or {}).get("pmcid"), *(location.get("landing_page_url") for location in locations)]
        pmcids = [
            match[1]
            for page in pages
            if (match := re.search(r"(?:/(?:pmc/)?articles/(?:PMC)?|^PMC)(\d+)(?:[/?#]|$)", str(page)))
        ]
    except (requests.RequestException, ValueError, AttributeError, TypeError) as error:
        # ponytail: every failure but a 404 counts as transient (timeouts, 429, 5xx); split out persistent 4xx if seen.
        error_text = f"OpenAlex lookup failed: {str(error) or type(error).__name__}"
        return [], _attempt(
            bibcode=job["bibcode"], url=url, method="openalex", status="failed", category="transient", error=error_text
        )
    if pmcids:
        urls.append(PMC_PDF_URL.format(pmcid=f"PMC{pmcids[0]}"))
    return list(dict.fromkeys(urls)), None


def _save_pdf(job: dict, content: bytes, attempts_path: Path) -> bool:
    try:
        atomic_write(job["target"], content)
    except OSError as error:
        attempt = _attempt(
            bibcode=job["bibcode"],
            url="",
            method="write",
            status="failed",
            category="transient",
            error=str(error) or type(error).__name__,
        )
        job["attempts"].append(attempt)
        append_jsonl(attempts_path, attempt)
        return False
    return True


def _try_urls(fetcher: _Fetcher, job: dict, urls: list[str], method: str, attempts_path: Path) -> bool:
    for url in urls:
        content, attempt = _download_url(fetcher, job, url, method)
        job["attempts"].append(attempt)
        append_jsonl(attempts_path, attempt)
        if content is not None and _save_pdf(job, content, attempts_path):
            return True
    return False


def _download_job(fetcher: _Fetcher, job: dict, attempts_path: Path) -> bool:
    """Try the ADS candidates, then the open copies OpenAlex lists for the DOI; return whether a PDF was saved."""
    if _try_urls(fetcher, job, job["urls"], "http", attempts_path):
        return True
    if not job["doi"]:
        return False
    urls, failure = _open_access_urls(fetcher, job)
    if failure is not None:
        job["attempts"].append(failure)
        append_jsonl(attempts_path, failure)
    tried = {url.split("://", 1)[-1] for url in job["urls"]}
    extra = [url for url in urls if url.split("://", 1)[-1] not in tried]
    job["urls"] += extra
    return _try_urls(fetcher, job, extra, "openalex", attempts_path)


def _job(record: Mapping[str, Any], output_dir: Path) -> dict:
    bibcode = str(record["bibcode"])
    return {
        "bibcode": bibcode,
        "target": output_dir / f"{bibcode}.pdf",
        "urls": candidate_pdf_urls(record),
        "expected_sha256": str(record.get("pdf_sha256") or "") or None,
        "doi": _doi(record),
        "abstract": str(record.get("abstract") or "") or None,
        "attempts": [],
    }


def _existing_pdf(job: dict, attempts_path: Path) -> str | None:
    """Validate cached PDFs; preserve rejected copies outside PDF discovery before trying a replacement."""
    if not job["target"].is_file():
        return None
    existing = job["target"].read_bytes()
    try:
        validate_pdf(existing, abstract=job["abstract"])
    except ValueError as error:
        rejected = job["target"].with_suffix(f".{hashlib.sha256(existing).hexdigest()}.rejected")
        job["target"].rename(rejected)
        attempt = _attempt(
            bibcode=job["bibcode"],
            url="",
            method="existing",
            status="failed",
            category="permanent",
            error=f"{error}; file preserved as {rejected.name}",
        )
        job["attempts"].append(attempt)
        append_jsonl(attempts_path, attempt)
        return None
    if job["expected_sha256"] and hashlib.sha256(existing).hexdigest() != job["expected_sha256"]:
        return "conflict"
    return "valid"


def _manual_entry(job: dict) -> dict:
    attempts = job["attempts"]
    errors = [attempt["error"] for attempt in attempts if attempt["error"]]
    if not attempts:
        searched = "ADS or OpenAlex" if job["doi"] else "ADS (no DOI for OpenAlex)"
        errors.append(f"no open-access PDF URL was found in {searched}")
    categories = {attempt["category"] for attempt in attempts}
    # A rerun may still fetch a transient failure, so that outranks needing a browser.
    category = next((category for category in ("transient", "blocked") if category in categories), "permanent")
    return {
        "bibcode": job["bibcode"],
        "target": str(job["target"]),
        "urls": job["urls"],
        "category": category,
        "errors": errors,
        "updated_at": _now(),
    }


def missing_pdfs(pdf_dir: Path) -> list[dict]:
    """The papers in `pdf_dir`'s manual download queue whose PDF is not yet at its target path."""
    queue = pdf_dir / "manual_downloads.jsonl"
    if not queue.is_file():
        return []
    return [entry for entry in read_jsonl(queue) if not Path(entry["target"]).is_file()]


def download_records(
    records: Iterable[Mapping[str, object]],
    output_dir: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> dict[str, int]:
    """Download missing PDFs and checkpoint attempts and unresolved work."""
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = output_dir / "download_attempts.jsonl"
    manual_path = output_dir / "manual_downloads.jsonl"
    manual = {record["bibcode"]: record for record in read_jsonl(manual_path)} if manual_path.is_file() else {}
    records = list(records)
    jobs: list[dict] = []
    skipped = conflicts = 0
    for index, record in enumerate(records, start=1):
        job = _job(record, output_dir)
        logger.info("[%d/%d] prepare %s", index, len(records), job["bibcode"])
        existing = _existing_pdf(job, attempts_path)
        if existing is None:
            jobs.append(job)
        elif existing == "valid":
            manual.pop(job["bibcode"], None)
            skipped += 1
        else:
            attempt = _attempt(
                bibcode=job["bibcode"],
                url="",
                method="existing",
                status="failed",
                category="permanent",
                error="existing valid PDF conflicts with the expected SHA-256; file was preserved",
            )
            job["attempts"].append(attempt)
            append_jsonl(attempts_path, attempt)
            manual[job["bibcode"]] = _manual_entry(job)
            conflicts += 1

    own_session = session is None
    fetcher = _Fetcher(session or requests.Session(), timeout)
    downloaded = 0
    try:
        for index, job in enumerate(jobs, start=1):
            logger.info("[%d/%d] download %s", index, len(jobs), job["bibcode"])
            if _download_job(fetcher, job, attempts_path):
                manual.pop(job["bibcode"], None)
                downloaded += 1
            else:
                manual[job["bibcode"]] = _manual_entry(job)
    finally:
        if own_session:
            fetcher.session.close()

    write_jsonl(manual_path, (manual[bibcode] for bibcode in sorted(manual)))
    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": len(jobs) - downloaded + conflicts,
        "manual_queue": len(manual),
    }
