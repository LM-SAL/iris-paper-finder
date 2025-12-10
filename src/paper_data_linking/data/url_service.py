import re
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from paper_data_linking import logger
from paper_data_linking.data.headers import build_headers
from paper_data_linking.data.models import BasicMetadataRecord

HEADERS = build_headers()


class URLTransformationService:
    def transform(self, urls):
        transformed_urls = []
        for url in urls:
            transformed_url = self._transform_single_url(url)
            transformed_urls.append(transformed_url)
        return self._prioritize_urls(list(set(transformed_urls)))

    def _transform_single_url(self, url):  # NOQA: PLR0911
        # Transformations for various URL types go here:
        if "doi.org" in url:
            return self._transform_doi_url(url)
        if "arxiv" in url:
            return self._transform_arxiv_url(url)
        if "articles.adsabs.harvard.edu/full/" in url:
            return self._transform_ads_url(url)
        if "link.springer.com/article/" in url:
            return self._transform_springer_url(url)
        if (
            "earth-planets-space.springeropen.com/articles/" in url
            or "geoscienceletters.springeropen.com/articles/" in url
        ):
            return self._transform_springer_open_url(url)
        if "copernicus.org/articles/" in url:
            return self._transform_copernicus_url(url)
        if "www.aanda.org" in url:
            return self._transform_aanda_url(url)
        if "agupubs.onlinelibrary.wiley.com/doi/abs/" in url:
            return self._transform_wiley_agu_url(url)
        if "onlinelibrary.wiley.com" in url:
            return self._transform_wiley_url(url)
        if "degruyter.com" in url:
            return self._transform_degruyter_url(url)
        return url

    def _transform_doi_url(self, url):
        try:
            response = requests.get(url, allow_redirects=True, timeout=30, headers=HEADERS)
            response.raise_for_status()
            return self._transform_single_url(response.url)
        except requests.exceptions.RequestException:
            return url

    def _transform_arxiv_url(self, url):
        pdf_url = url.replace("abs", "pdf")
        return pdf_url.replace("arxiv.org", "export.arxiv.org")

    def _transform_ads_url(self, url):
        return f"{url}?defaultprint=YES" if "?defaultprint=YES" not in url else url

    def _transform_springer_url(self, url):
        match = re.search(r"10\.\d{4}/\S+", url)
        if match:
            doi = match.group(0)
            return f"https://link.springer.com/content/pdf/{doi}.pdf?pdf=button"
        return url

    def _transform_springer_open_url(self, url):
        match = re.search(r"10\.\d{4}/\S+", url)
        if match:
            doi = match.group(0)
            base_url = url.split("/articles/")[0]
            return f"{base_url}/counter/pdf/{doi}.pdf"
        return url

    def _transform_copernicus_url(self, url):
        if url.endswith(".html"):
            return url[:-5] + ".pdf"
        article_parts = url.split("/")[-4:]
        article_id = "-".join(article_parts)[:-1]
        return f"{url}angeo-{article_id}.pdf"

    def _transform_aanda_url(self, url):
        return self._get_aanda_pdf_url(url) or url

    def _transform_wiley_agu_url(self, url):
        return url.replace("/abs/", "/epdf/")

    def _transform_wiley_url(self, url):
        match = re.search(r"10\.\d{4}/\S+", url)
        if match:
            doi = match.group(0)
            return f"https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/{doi}"
        return url

    def _transform_degruyter_url(self, url):
        return url.replace("html", "pdf")

    def _prioritize_urls(self, urls):
        def url_priority(url):
            if "arxiv.org" in url:
                return 0
            if "articles.adsabs.harvard.edu/full/" in url:
                return 1
            if "link.springer.com/article/" in url or "copernicus.org/articles/" in url:
                return 2
            return 10

        return sorted(urls, key=url_priority)

    def _get_aanda_pdf_url(self, url):
        logger.debug(f"Getting pdf url for {url}")
        try:
            response = requests.get(url, allow_redirects=True, timeout=30, headers=HEADERS)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                pdf_link = soup.select_one(".article_doc > ul:nth-child(1) > li:nth-child(3) > a:nth-child(1)")
                if pdf_link:
                    return f"https://www.aanda.org{pdf_link['href']}"
        except Exception as e:
            logger.warning(f"Failed to get pdf url because {e}")
        return None


class MetadataTransformer:
    def __init__(self, transformation_service) -> None:
        self.transformation_service = transformation_service

    def _filter_open_access(self, links_data):
        return [link["url"] for link in links_data if link["access"] == "open"]

    def _omit_unwanted_urls(self, urls):
        omit_urls = ["hcvalidate.perfdrive.com"]
        return [url for url in urls if not any(ou in url for ou in omit_urls)]

    def _process_line(self, line: str) -> str:
        """
        Process one JSONL line and return the output JSON string (without
        trailing newline).
        """
        if not line.strip():
            return ""
        record_dict = json.loads(line)
        record = BasicMetadataRecord.from_dict(record_dict)
        transformed_urls = []
        try:
            response = requests.get(
                f"https://ui.adsabs.harvard.edu/link_gateway/{record.bibcode}/ESOURCE",
                allow_redirects=True,
                timeout=30,
                headers=HEADERS,
            )
            response.raise_for_status()
            nice_data = BeautifulSoup(response.text, "html.parser")
            for link in nice_data.find_all("a", href=True):
                href = str(link["href"])
                if "pdf" in href.lower() and href not in transformed_urls:
                    transformed_urls.insert(0, href)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get transformed URLs for {record.bibcode} because {e}")
            logger.warning("Falling back to links_data processing.")
            open_access_urls = self._filter_open_access(record.links_data)
            open_access_urls = self._omit_unwanted_urls(open_access_urls)
            transformed_urls = transformed_urls.append(self.transformation_service.transform(open_access_urls))
        record.pdf_links = transformed_urls
        return json.dumps(record.to_dict())

    def export_transformed_links(self, input_file, output_file, max_workers):
        logger.info(f"Transforming URLs in {input_file} and saving to {output_file}")
        input_path = Path(input_file)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            input_path.open("r", encoding="utf-8") as infile,
            output_path.open("w", encoding="utf-8") as outfile,
            ThreadPoolExecutor(max_workers=max_workers) as executor,
        ):
            for out in tqdm(
                executor.map(self._process_line, infile),
                desc="Processing records",
            ):
                if not out:
                    continue
                outfile.write(out + "\n")
        logger.info("For any records that failed to process, no links were added.")
        logger.info("You will want to work out the PDF links for these manually and update the output file.")


def main():
    parser = argparse.ArgumentParser(description="Transform URLs in metadata records.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file with metadata records.")
    parser.add_argument(
        "output_file", type=str, help="Path to the output JSONL file to store metadata with links to pdfs."
    )
    parser.add_argument(
        "--max-workers", type=int, default=8, help="Maximum number of worker threads to use for processing."
    )
    args = parser.parse_args()
    transformation_service = URLTransformationService()
    transformer = MetadataTransformer(transformation_service)
    transformer.export_transformed_links(args.input_file, args.output_file, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
