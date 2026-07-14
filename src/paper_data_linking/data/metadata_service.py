import argparse
import json
from pathlib import Path

import requests

from paper_data_linking import logger
from paper_data_linking.data.models import BasicMetadataRecord
from paper_data_linking.settings import ADS_TOKEN


class MetadataService:
    def __init__(self, api_token: str) -> None:
        if not api_token:
            msg = "ADS API token is required"
            raise ValueError(msg)
        self.api_token = api_token

    def get_metadata_for_bibcodes(self, bibcodes: list[str]):
        """
        Fetch metadata for a list of bibcodes.
        """
        logger.info(f"Trying to download {len(bibcodes)} metadata records.")
        headers = {
            "Content-Type": "big-query/csv",
            "Authorization": "Bearer " + self.api_token,
        }
        fields = [
            "bibcode",
            "links_data",
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
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            num_found = data["response"]["numFound"]
            if num_found != len(bibcodes):
                logger.warning(f"num_found ({num_found}) and len(bibcodes) ({len(bibcodes)}) are different.")
            inner_docs = data["response"]["docs"]
            num_downloaded = len(inner_docs)
            if not num_downloaded and start < num_found:
                msg = f"ADS returned no metadata records at offset {start} of {num_found}"
                raise RuntimeError(msg)
            logger.info(f"{start + num_downloaded}/{num_found}")
            start = data["response"]["start"] + num_downloaded
            docs.extend(inner_docs)
        return [BasicMetadataRecord.from_dict(doc) for doc in docs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch metadata for a list of bibcodes from ADS.")
    parser.add_argument("bibcodes_file", type=str, help="Path to the file containing bibcodes, one per line.")
    parser.add_argument("output_file", type=str, help="Path to the output file where metadata records will be saved.")
    parser.add_argument("--api-token", default=ADS_TOKEN, help="API token for accessing ADS.")
    args = parser.parse_args()
    if not args.api_token:
        parser.error("ADS_TOKEN or --api-token is required")
    service = MetadataService(args.api_token)
    bibcodes = [line.strip() for line in Path(args.bibcodes_file).read_text().splitlines() if line.strip()]
    records = service.get_metadata_for_bibcodes(bibcodes)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
    logger.info(f"Wrote metadata records to {output_path}")


if __name__ == "__main__":
    main()
