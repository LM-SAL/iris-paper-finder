import argparse
import json
import os
from pathlib import Path
from typing import List

import ads
import numpy as np
import requests
from dotenv import load_dotenv, find_dotenv

from paper_data_linking import logger
from paper_data_linking.settings import ADS_TOKEN

if ADS_TOKEN is None:
    raise Exception("Must set ADS_TOKEN. Add ADS_TOKEN to .env file.")


class BibcodeService:
    def __init__(self, api_token: str):
        """
        Initializes the BibcodeService.

        Args:
            api_token (str): API token for accessing ADS.
        """
        self.api_token = api_token
        self.start = 0
        ads.config.token = self.api_token

    def get_bibcodes_from_library(self, library_id: str) -> List[str]:
        """
        Fetches bibcodes from the ADS API.

        Args:
            library_id (str): Library ID for ADS.

        Returns:
            List[str]: List of fetched bibcodes.
        """
        headers = {"Authorization": "Bearer " + self.api_token}
        rows = 2000
        num_found = np.inf
        bibcodes = []
        while self.start <= num_found:
            url = f"https://api.adsabs.harvard.edu/v1/biblib/libraries/{library_id}?rows={rows}&start={self.start}"
            response = requests.get(url, headers=headers)
            data = json.loads(response.content)
            num_found = data["solr"]["response"]["numFound"]
            bibcodes_chunk = data["documents"]
            logger.info(f"Extracted: {self.start + len(bibcodes_chunk)}/{num_found}")
            self.start += rows  # Update start but don't save it as a token
            bibcodes.extend(bibcodes_chunk)
        return bibcodes

    def get_bibcodes_by_query(self, query: str, rows: int = 2000) -> List[str]:
        """
        Fetches bibcodes from ADS based on a search query.

        Args:
            query (str): The search query for fetching bibcodes.
            rows (int, optional): Number of results to fetch. Defaults to 2000.

        Returns:
            List[str]: List of fetched bibcodes.
        """
        search_query = ads.SearchQuery(q=query, rows=rows)
        return [paper.bibcode for paper in search_query]


def main():
    parser = argparse.ArgumentParser(description="Fetch bibcodes from ADS and save them to a file.")
    parser.add_argument("--api_token", default=ADS_TOKEN, help="API token for accessing ADS.")
    parser.add_argument("--output", help="Path to the output file where all bibcodes will be saved.", type=Path)
    parser.add_argument("--num_records", type=int, default=2000,
                        help="Desired number of records to fetch. Relevant mainly for search queries.",)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--library_id", help="Library ID for ADS.")
    group.add_argument("--query", help="Search query for fetching bibcodes.")
    args = parser.parse_args()
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
