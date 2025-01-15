import argparse
import json
import os
from typing import List

import requests
from dotenv import load_dotenv, find_dotenv

from paper_data_linking.data.models import MetadataRecord
from paper_data_linking.log_config import logger
from paper_data_linking.utils import load_bibcodes_list, to_jsonlines

load_dotenv(find_dotenv())


class MetadataService:
    ADS_TOKEN = os.getenv("ADS_TOKEN")

    def __init__(self):
        if not self.ADS_TOKEN:
            raise Exception("Must set ADS_TOKEN. Add ADS_TOKEN to .env file.")

    def get_metadata_for_bibcodes(self, bibcodes: List[str]):
        """Fetch metadata for a list of bibcodes."""
        logger.info(f"Trying to download {len(bibcodes)} metadata records.")

        # Define headers and other constants required for the API call
        headers = {
            'Content-Type': 'big-query/csv',
            'Authorization': 'Bearer ' + self.ADS_TOKEN,
        }
        fields = [
            "bibcode",
            "id",
            "title",
            "author",
            "year",
            "pub",
            "links_data",
            "pubdate",
            "doi",
            "doctype",
        ]
        fields_str = ",".join(fields)
        payload = "bibcode\n" + "\n".join(bibcodes)

        # Initialize counters for pagination
        start = 0
        num_found = len(bibcodes)
        docs = []

        while start < num_found:
            response = requests.post(
                f'https://api.adsabs.harvard.edu/v1/search/bigquery?q=*:*&fl={fields_str}&rows=2000&start={start}',
                headers=headers,
                data=payload,
            )
            data = json.loads(response.content)

            num_found = data['response']['numFound']
            if num_found != len(bibcodes):
                logger.warning(f"num_found ({num_found}) and len(bibcodes) ({len(bibcodes)}) are different.")

            inner_docs = data['response']['docs']
            num_downloaded = len(inner_docs)
            logger.info(f"{start + num_downloaded}/{num_found}")

            start = data['response']['start'] + num_downloaded
            docs.extend(inner_docs)  # Using extend for combining lists

        records = [MetadataRecord.from_dict(doc) for doc in docs]
        return records


def main(args):
    # Instantiate the MetadataService
    service = MetadataService()

    # Load bibcodes
    bibcodes = load_bibcodes_list(args.bibcodes_file)

    # Fetch metadata for bibcodes
    records = service.get_metadata_for_bibcodes(bibcodes)

    # Save metadata records to the specified output file
    docs = [record.to_dict() for record in records]
    outfile = to_jsonlines(docs, args.output_file)
    logger.info(f"Wrote metadata records to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch metadata for a list of bibcodes from ADS.")
    parser.add_argument('bibcodes_file', type=str, help="Path to the file containing bibcodes, one per line.")
    parser.add_argument('output_file', type=str, help="Path to the output file where metadata records will be saved.")
    args = parser.parse_args()

    main(args)