"""Validated, resumable PDF downloads with an optional browser fallback."""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import monotonic, sleep
from typing import TYPE_CHECKING

import fitz
import requests

from iris_paper_llm.jsonl import append_jsonl, read_jsonl, write_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_BROWSER_WAIT_SECONDS = 15.0
USER_AGENT = "iris-papers/0.1"
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
}
TRANSIENT_HTTP_STATUS = {408, 425, 429}
logger = logging.getLogger(__name__)


def candidate_pdf_urls(record: Mapping[str, object]) -> list[str]:
    """Return explicit PDF and arXiv links in stable ADS order."""
    urls = [str(url) for url in record.get("pdf_links", []) or []]
    for link in record.get("links_data", []) or []:
        if not isinstance(link, dict) or link.get("access") != "open":
            continue
        url = str(link.get("url", ""))
        link_type = link.get("type")
        if link_type == "pdf":
            urls.append(url)
        elif link_type == "preprint" and "arxiv.org/" in url:
            urls.append(url.replace("http://", "https://").replace("/abs/", "/pdf/"))
    return list(dict.fromkeys(url for url in urls if url))


def validate_pdf(content: bytes, expected_sha256: str | None = None) -> None:
    """Reject non-PDF, unreadable, empty, or checksum-mismatched content."""
    if not content.startswith(b"%PDF-"):
        msg = "content does not start with a PDF signature"
        raise ValueError(msg)
    if expected_sha256 and hashlib.sha256(content).hexdigest() != expected_sha256:
        msg = "PDF checksum does not match the expected SHA-256"
        raise ValueError(msg)
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            page_count = document.page_count
            if page_count:
                document.load_page(0)
    except ValueError:
        raise
    except Exception as error:
        msg = "PDF cannot be opened or its first page cannot be read"
        raise ValueError(msg) from error
    if page_count < 1:
        msg = "PDF has no pages"
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


def download_http(
    session: requests.Session,
    *,
    bibcode: str,
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    expected_sha256: str | None = None,
) -> tuple[bytes | None, dict]:
    """Attempt one direct HTTP download and return its durable record."""
    try:
        response = session.get(url, allow_redirects=True, headers=dict(headers), timeout=timeout)
        if not 200 <= response.status_code < 300:
            category = (
                "transient"
                if response.status_code in TRANSIENT_HTTP_STATUS or response.status_code >= 500
                else "permanent"
            )
            msg = f"HTTP {response.status_code}"
            return None, _attempt(
                bibcode=bibcode,
                url=url,
                final_url=response.url,
                method="http",
                status="failed",
                category=category,
                error=msg,
            )
        content_type = response.headers.get("content-type", "").lower()
        if (
            content_type
            and "pdf" not in content_type
            and "octet-stream" not in content_type
            and not response.content.startswith(b"%PDF-")
        ):
            msg = f"unexpected content type: {content_type}"
            return None, _attempt(
                bibcode=bibcode,
                url=url,
                final_url=response.url,
                method="http",
                status="failed",
                category="permanent",
                error=msg,
            )
        validate_pdf(response.content, expected_sha256)
        return response.content, _attempt(
            bibcode=bibcode,
            url=url,
            final_url=response.url,
            method="http",
            status="downloaded",
        )
    except requests.RequestException as error:
        return None, _attempt(
            bibcode=bibcode,
            url=url,
            method="http",
            status="failed",
            category="transient",
            error=str(error) or type(error).__name__,
        )
    except ValueError as error:
        return None, _attempt(
            bibcode=bibcode,
            url=url,
            method="http",
            status="failed",
            category="permanent",
            error=str(error),
        )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _save_pdf(job: dict, content: bytes, attempts_path: Path) -> bool:
    try:
        _atomic_write(job["target"], content)
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


def _load_manual_queue(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return {record["bibcode"]: record for record in read_jsonl(path)}


def _browser_fetch(driver: object, url: str, wait_seconds: float, expected_sha256: str | None) -> bytes:
    with TemporaryDirectory() as directory:
        driver.execute_cdp_cmd(
            "Page.setDownloadBehavior",
            {"behavior": "allow", "downloadPath": directory},
        )
        driver.get(url)
        deadline = monotonic() + wait_seconds
        while monotonic() < deadline:
            files = [path for path in Path(directory).iterdir() if path.is_file() and path.suffix != ".crdownload"]
            if files:
                content = max(files, key=lambda path: path.stat().st_size).read_bytes()
                validate_pdf(content, expected_sha256)
                return content
            sleep(0.25)
    msg = f"browser produced no PDF within {wait_seconds:g} seconds"
    raise TimeoutError(msg)


def browser_attempts(jobs: Iterable[dict], wait_seconds: float) -> Iterator[tuple[dict, bytes | None, list[dict]]]:
    """Try direct-download failures with one lazily created browser driver."""
    jobs = list(jobs)
    try:
        from selenium import webdriver
    except ImportError as error:
        for job in jobs:
            attempt = _attempt(
                bibcode=job["bibcode"],
                url=job["urls"][0],
                method="browser",
                status="failed",
                category="permanent",
                error=f"browser fallback is not installed: {error}",
            )
            yield job, None, [attempt]
        return

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option(
        "prefs",
        {
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,
        },
    )
    try:
        driver_context = webdriver.Chrome(options=options)
    except Exception as error:
        for job in jobs:
            attempt = _attempt(
                bibcode=job["bibcode"],
                url=job["urls"][0],
                method="browser",
                status="failed",
                category="permanent",
                error=f"browser setup failed: {error}",
            )
            yield job, None, [attempt]
        return

    with driver_context as driver:
        for job in jobs:
            attempts = []
            content = None
            for url in job["urls"]:
                try:
                    content = _browser_fetch(driver, url, wait_seconds, job["expected_sha256"])
                    attempts.append(
                        _attempt(
                            bibcode=job["bibcode"],
                            url=url,
                            method="browser",
                            status="downloaded",
                        )
                    )
                    break
                except Exception as error:
                    attempts.append(
                        _attempt(
                            bibcode=job["bibcode"],
                            url=url,
                            method="browser",
                            status="failed",
                            category="permanent",
                            error=str(error) or type(error).__name__,
                        )
                    )
            yield job, content, attempts


def download_records(
    records: Iterable[Mapping[str, object]],
    output_dir: Path,
    *,
    headers: Mapping[str, str] = DEFAULT_HEADERS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    browser_fallback: bool = False,
    browser_wait: float = DEFAULT_BROWSER_WAIT_SECONDS,
    session: requests.Session | None = None,
) -> dict[str, int]:
    """Download missing PDFs and checkpoint attempts and unresolved work."""
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = output_dir / "download_attempts.jsonl"
    manual_path = output_dir / "manual_downloads.jsonl"
    manual = _load_manual_queue(manual_path)
    records = list(records)
    jobs = []
    skipped = conflicts = 0

    for index, record in enumerate(records, start=1):
        bibcode = str(record["bibcode"])
        target = output_dir / f"{bibcode}.pdf"
        expected_sha256 = str(record.get("pdf_sha256") or "") or None
        logger.info("[%d/%d] prepare %s", index, len(records), bibcode)
        if target.is_file():
            existing = target.read_bytes()
            try:
                validate_pdf(existing)
            except ValueError:
                pass
            else:
                if expected_sha256 and hashlib.sha256(existing).hexdigest() != expected_sha256:
                    attempt = _attempt(
                        bibcode=bibcode,
                        url="",
                        method="existing",
                        status="failed",
                        category="permanent",
                        error="existing valid PDF conflicts with the expected SHA-256; file was preserved",
                    )
                    append_jsonl(attempts_path, attempt)
                    manual[bibcode] = {
                        "bibcode": bibcode,
                        "target": str(target),
                        "urls": candidate_pdf_urls(record),
                        "category": "permanent",
                        "errors": [attempt["error"]],
                        "updated_at": _now(),
                    }
                    conflicts += 1
                    continue
                manual.pop(bibcode, None)
                skipped += 1
                continue
        urls = candidate_pdf_urls(record)
        jobs.append(
            {
                "bibcode": bibcode,
                "target": target,
                "urls": urls,
                "expected_sha256": expected_sha256,
                "attempts": [],
            }
        )

    own_session = session is None
    session = session or requests.Session()
    downloaded = 0
    pending = []
    try:
        for index, job in enumerate(jobs, start=1):
            logger.info("[%d/%d] download %s", index, len(jobs), job["bibcode"])
            content = None
            for url in job["urls"]:
                content, attempt = download_http(
                    session,
                    bibcode=job["bibcode"],
                    url=url,
                    headers=headers,
                    timeout=timeout,
                    expected_sha256=job["expected_sha256"],
                )
                job["attempts"].append(attempt)
                append_jsonl(attempts_path, attempt)
                if content is not None:
                    if _save_pdf(job, content, attempts_path):
                        manual.pop(job["bibcode"], None)
                        downloaded += 1
                        break
                    content = None
            if content is None:
                pending.append(job)
    finally:
        if own_session:
            session.close()

    if browser_fallback and pending:
        no_url_jobs = [job for job in pending if not job["urls"]]
        still_pending = []
        for job, content, attempts in browser_attempts((job for job in pending if job["urls"]), browser_wait):
            for attempt in attempts:
                job["attempts"].append(attempt)
                append_jsonl(attempts_path, attempt)
            if content is None:
                still_pending.append(job)
            elif _save_pdf(job, content, attempts_path):
                manual.pop(job["bibcode"], None)
                downloaded += 1
            else:
                still_pending.append(job)
        pending = no_url_jobs + still_pending

    for job in pending:
        attempts = job["attempts"]
        errors = [attempt["error"] for attempt in attempts if attempt["error"]]
        if not job["urls"]:
            errors.append("no explicit open-access PDF or arXiv URL was supplied by ADS")
        category = "transient" if any(attempt["category"] == "transient" for attempt in attempts) else "permanent"
        manual[job["bibcode"]] = {
            "bibcode": job["bibcode"],
            "target": str(job["target"]),
            "urls": job["urls"],
            "category": category,
            "errors": errors,
            "updated_at": _now(),
        }

    write_jsonl(manual_path, (manual[bibcode] for bibcode in sorted(manual)))
    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": len(pending) + conflicts,
        "manual_queue": len(manual),
    }
