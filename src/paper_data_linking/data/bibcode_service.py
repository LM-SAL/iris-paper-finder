import argparse
import math
from pathlib import Path

import ads
import requests

from paper_data_linking import logger
from paper_data_linking.settings import ADS_TOKEN


class BibcodeService:
    def __init__(self, api_token: str) -> None:
        """
        Initializes the BibcodeService.

        Args:
            api_token (str): API token for accessing ADS.
        """
        if not api_token:
            msg = "ADS API token is required"
            raise ValueError(msg)
        self.api_token = api_token
        ads.config.token = self.api_token

    def get_bibcodes_from_library(self, library_id: str) -> list[str]:
        """
        Fetches bibcodes from the ADS API.

        Args:
            library_id (str): Library ID for ADS.

        Returns
        -------
            List[str]: List of fetched bibcodes.
        """
        headers = {"Authorization": "Bearer " + self.api_token}
        rows = 2000
        start = 0
        num_found = math.inf
        bibcodes = []
        while start < num_found:
            url = f"https://api.adsabs.harvard.edu/v1/biblib/libraries/{library_id}?rows={rows}&start={start}"
            response = requests.get(url, headers=headers, timeout=360)
            response.raise_for_status()
            data = response.json()
            num_found = data["solr"]["response"]["numFound"]
            bibcodes_chunk = data["documents"]
            logger.info(f"Extracted: {start + len(bibcodes_chunk)}/{num_found}")
            start += rows
            bibcodes.extend(bibcodes_chunk)
        return bibcodes

    def get_bibcodes_by_query(self, query: str, rows: int = 2000) -> list[str]:
        """
        Fetches bibcodes from ADS based on a search query.

        Args:
            query (str): The search query for fetching bibcodes.
            rows (int, optional): Number of results to fetch. Defaults to 2000.

        Returns
        -------
            List[str]: List of fetched bibcodes.
        """
        search_query = ads.SearchQuery(q=query, rows=rows)
        return [paper.bibcode for paper in search_query]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch bibcodes from ADS and save them to a file.")
    parser.add_argument("--api-token", "--api_token", default=ADS_TOKEN, help="API token for accessing ADS.")
    parser.add_argument(
        "--output", required=True, help="Path to the output file where all bibcodes will be saved.", type=Path
    )
    parser.add_argument(
        "--num_records",
        type=int,
        default=2000,
        help="Desired number of records to fetch. Relevant mainly for search queries.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--library_id", help="Library ID for ADS.")
    group.add_argument("--query", help="Search query for fetching bibcodes.")
    args = parser.parse_args()
    if not args.api_token:
        parser.error("ADS_TOKEN or --api-token is required")
    bibcode_service = BibcodeService(
        api_token=args.api_token,
    )
    if args.library_id:
        logger.info(f"Fetching bibcodes from ADS based on the library ID: {args.library_id}")
        bibcodes = bibcode_service.get_bibcodes_from_library(library_id=args.library_id)
    else:
        logger.info(f"Fetching bibcodes from ADS based on the search query: {args.query}")
        bibcodes = bibcode_service.get_bibcodes_by_query(query=args.query, rows=args.num_records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w+") as f:
        for bibcode in bibcodes:
            f.write(f"{bibcode}\n")
    logger.info(f"Bibcodes saved to {out_path}")


if __name__ == "__main__":
    main()
