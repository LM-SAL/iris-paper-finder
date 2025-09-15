import argparse
import asyncio
import json
from pathlib import Path
from typing import List

from tqdm.asyncio import tqdm

import paper_data_linking.utils as utils
from paper_data_linking.data.downloaders import (
    Downloader,
    AsyncRequestsDownloader,
    AsyncSeleniumDownloader,
)
from paper_data_linking import logger
from paper_data_linking.data.models import MetadataRecord


class AsyncPdfDownloadService:
    def __init__(self, downloaders: List[Downloader]):
        self.downloaders = downloaders

    async def download_pdf(self, urls: List[str]) -> bytes:
        errors = []
        for url in urls:
            for downloader in self.downloaders:
                try:
                    async with downloader:
                        logger.debug(f"Trying {downloader.__class__.__name__} for {url}")
                        file = await downloader.download(url)
                        return file
                except Exception as e:
                    logger.debug(
                        f"Downloader {downloader.__class__.__name__} failed for {url} due to {e}"
                    )
                    errors.append((downloader.__class__.__name__, url, str(e)))
        raise Exception(f"All download attempts failed. Errors: {errors}")


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
        with failed_bibcodes_file.open('r') as f:
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
    with input_path.open("r") as f:
        dict_records = [json.loads(line) for line in f]
    all_records = [MetadataRecord.from_dict(doc) for doc in dict_records]
    # Exclude records for which we already have pdfs
    records = [
        r for r in all_records
        if not (output_dir / f"{r.bibcode}.pdf").exists()
        and r.bibcode not in failed_bibcodes
    ]
    records = sorted(records, key=lambda r: r.bibcode)
    logger.info(f"Downloading PDFs for {len(records)} records.")
    success_count = 0
    total_count = 0
    pbar = tqdm(records, desc=f"Success: {success_count} | Ratio: NaN")
    async for r in pbar:
        try:
            logger.info(f"Downloading PDF for {r.bibcode} using {r.pdf_links}")
            content = await pdf_service.download_pdf(r.pdf_links)
            if len(content) <= 8 * 1024:
                raise Exception(f"Downloaded PDF is too small for {r.pdf_links}")
            utils.write_pdf(content, output_dir, r.bibcode)
            success_count += 1
        except Exception as e:
            logger.info(f"Failed to get PDF for {r.bibcode} due to {e}")
            with open(failed_bibcodes_file, 'a') as f:
                f.write(f"{r.bibcode}\n")
        finally:
            total_count += 1
            pbar.set_description(f"Success: {success_count} | Ratio: {success_count / total_count:.2f}")
    utils.validate_pdfs(output_dir)

if __name__ == "__main__":
    asyncio.run(main())
