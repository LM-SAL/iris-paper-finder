import json
import argparse
from pathlib import Path

import requests

from paper_data_linking import logger
from paper_data_linking.data.models import MetadataRecord
from paper_data_linking.settings import ADS_TOKEN
from paper_data_linking.utils import load_bibcodes_list, to_jsonlines

if ADS_TOKEN is None:
    msg = "Must set ADS_TOKEN. Add ADS_TOKEN to .env file."
    raise Exception(msg)


class MetadataService:
    def get_metadata_for_bibcodes(self, bibcodes: list[str]):
        """
        Fetch metadata for a list of bibcodes.
        """
        logger.info(f"Trying to download {len(bibcodes)} metadata records.")
        headers = {
            "Content-Type": "big-query/csv",
            "Authorization": "Bearer " + ADS_TOKEN,
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
        start = 0
        num_found = len(bibcodes)
        docs = []
        while start < num_found:
            response = requests.post(
                f"https://api.adsabs.harvard.edu/v1/search/bigquery?q=*:*&fl={fields_str}&rows=2000&start={start}",
                headers=headers,
                data=payload,
                timeout=360,
            )
            data = json.loads(response.content)
            num_found = data["response"]["numFound"]
            if num_found != len(bibcodes):
                logger.warning(f"num_found ({num_found}) and len(bibcodes) ({len(bibcodes)}) are different.")
            inner_docs = data["response"]["docs"]
            num_downloaded = len(inner_docs)
            logger.info(f"{start + num_downloaded}/{num_found}")
            start = data["response"]["start"] + num_downloaded
            docs.extend(inner_docs)
        return [MetadataRecord.from_dict(doc) for doc in docs]


def main():
    parser = argparse.ArgumentParser(description="Fetch metadata for a list of bibcodes from ADS.")
    parser.add_argument("bibcodes_file", type=str, help="Path to the file containing bibcodes, one per line.")
    parser.add_argument("output_file", type=str, help="Path to the output file where metadata records will be saved.")
    args = parser.parse_args()
    service = MetadataService()
    bibcodes = load_bibcodes_list(args.bibcodes_file)
    records = service.get_metadata_for_bibcodes(bibcodes)
    docs = [record.to_dict() for record in records]
    output_path = Path(args.output_file)
    outfile = to_jsonlines(docs, output_path)
    logger.info(f"Wrote metadata records to {outfile}")


if __name__ == "__main__":
    main()
