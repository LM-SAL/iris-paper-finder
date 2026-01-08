import json
import asyncio
import argparse
from typing import Any
from pathlib import Path

import aiofiles
from tqdm.asyncio import tqdm

from paper_data_linking import logger, utils
from paper_data_linking.data.downloaders import AsyncRequestsDownloader, AsyncSeleniumDownloader, Downloader
from paper_data_linking.data.models import BasicMetadataRecord


class AsyncPdfDownloadService:
    def __init__(self, downloaders: list[Downloader]) -> None:
        self.downloaders = downloaders

    async def download_pdf(self, urls: list[str]) -> bytes:
        errors = []
        for url in urls:
            for downloader in self.downloaders:
                try:
                    async with downloader:
                        logger.info(f"Trying {downloader.__class__.__name__} for {url}")
                        return await downloader.download(url)
                except Exception as e:
                    logger.info(f"Downloader {downloader.__class__.__name__} failed for {url} due to {e}")
                    errors.append((downloader.__class__.__name__, url, str(e)))
        msg = f"All download attempts failed. Errors: {errors}"
        raise Exception(msg)


async def read_dict_records(input_path: Path) -> list[dict[str, Any]]:
    async with aiofiles.open(input_path) as f:
        return [json.loads(line) async for line in f]


async def main():
    parser = argparse.ArgumentParser(description="Download PDFs using various methods.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the JSONL file containing bibcode URLs.",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save downloaded PDFs.",
        type=Path,
    )
    parser.add_argument(
        "--headers",
        default=None,
        help="Optional path to a JSON file containing request headers.",
        type=Path,
    )
    args = parser.parse_args()
    failed_bibcodes_file = Path(args.output_dir) / "failed_bibcodes.txt"
    failed_bibcodes = set()
    if failed_bibcodes_file.exists():
        with failed_bibcodes_file.open("r") as f:
            failed_bibcodes = {line.strip() for line in f.readlines()}
    if args.headers:
        with args.headers.open("r") as f:
            headers = json.load(f)
    else:
        headers = None
    if args.output_dir.exists():
        logger.warning(f"Output directory {args.output_dir} already exists.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    request_downloader = AsyncRequestsDownloader(headers)
    selenium_downloader = AsyncSeleniumDownloader()
    pdf_service = AsyncPdfDownloadService([request_downloader, selenium_downloader])
    input_path = Path(args.input)
    dict_records = await read_dict_records(input_path)
    all_records = [BasicMetadataRecord.from_dict(doc) for doc in dict_records]
    # Exclude records for which we already have pdfs
    records = [
        r for r in all_records if not (output_dir / f"{r.bibcode}.pdf").exists() and r.bibcode not in failed_bibcodes
    ]
    records = sorted(records, key=lambda r: r.bibcode)
    logger.info(f"Downloading PDFs for {len(records)} records.")
    success_count = 0
    total_count = 0
    pbar = tqdm(records, desc=f"Success: {success_count} | Ratio: NaN")
    for r in pbar:
        try:
            logger.info(f"Downloading PDF for {r.bibcode} using {r.pdf_links}")
            content = await pdf_service.download_pdf(r.pdf_links)
            if len(content) <= 8 * 1024:
                msg = f"Downloaded PDF is too small for {r.pdf_links}"
                logger.info(msg)
                failed_bibcodes.add(r.bibcode)
            else:
                await utils.write_pdf(content, output_dir, r.bibcode)
                success_count += 1
        except Exception as e:
            logger.info(f"Failed to get PDF for {r.bibcode} due to {e}")
            await utils.append_failed_bibcode(failed_bibcodes_file, r.bibcode)
        finally:
            total_count += 1
            pbar.set_description(f"Success: {success_count} | Ratio: {success_count / total_count:.2f}")
    utils.validate_pdfs(output_dir)


if __name__ == "__main__":
    asyncio.run(main())
