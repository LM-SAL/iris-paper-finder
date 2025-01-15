import json
import argparse
import logging
import re

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from paper_data_linking.data.models import MetadataRecord
from paper_data_linking.log_config import logger

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

class UrlTransformationService:

    def __init__(self):
        pass

    def transform(self, urls):
        transformed_urls = [self._transform_single_url(url) for url in urls]
        return self._prioritize_urls(list(set(transformed_urls)))

    def _transform_single_url(self, url):
        # Transformations for various URL types go here:

        if "arxiv" in url:
            return self._transform_arxiv_url(url)

        if "articles.adsabs.harvard.edu/full/" in url:
            return self._transform_ads_url(url)

        if "link.springer.com/article/" in url:
            return self._transform_springer_url(url)

        if ("earth-planets-space.springeropen.com/articles/" in url or
            "geoscienceletters.springeropen.com/articles/" in url):
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

        return url  # default behavior if no transformation rule matches

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
        return url.replace('html', 'pdf')

    def _prioritize_urls(self, urls):
        def url_priority(url):
            if "arxiv.org" in url:
                return 0
            elif "articles.adsabs.harvard.edu/full/" in url:
                return 1
            elif "link.springer.com/article/" in url or "copernicus.org/articles/" in url:
                return 2
            else:
                return 10
        return sorted(urls, key=url_priority)

    def _get_aanda_pdf_url(self, url):
        logger.debug(f"Getting pdf url for {url}")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                pdf_link = soup.select_one('.article_doc > ul:nth-child(1) > li:nth-child(3) > a:nth-child(1)')
                if pdf_link:
                    return f"https://www.aanda.org{pdf_link['href']}"
        except Exception as e:
            LOG.warning(f"Failed to get pdf url because {e}")
        return None


class MetadataTransformer:

    def __init__(self, transformation_service):
        self.transformation_service = transformation_service

    def _filter_open_access(self, links_data):
        return [link['url'] for link in links_data if link['access'] == 'open']

    def _omit_unwanted_urls(self, urls):
        omit_urls = ["hcvalidate.perfdrive.com"]
        return [url for url in urls if not any(ou in url for ou in omit_urls)]

    def export_transformed_links(self, input_file, output_file):
        logger.info(f"Transforming URLs in {input_file} and saving to {output_file}")
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in tqdm(infile):
                record_dict = json.loads(line)
                record = MetadataRecord.from_dict(record_dict)

                # Filter based on access level and omit unwanted URLs
                urls = self._filter_open_access(record.links_data)
                urls = self._omit_unwanted_urls(urls)

                # Transform the URLs
                transformed_urls = self.transformation_service.transform(urls)
                record.pdf_links = transformed_urls
                outfile.write(json.dumps(record.to_dict()) + '\n')


# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform URLs in metadata records.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file with metadata records.")
    parser.add_argument("output_file", type=str, help="Path to the output JSONL file to store metadata with links to pdfs.")

    args = parser.parse_args()

    transformation_service = UrlTransformationService()
    transformer = MetadataTransformer(transformation_service)
    transformer.export_transformed_links(args.input_file, args.output_file)