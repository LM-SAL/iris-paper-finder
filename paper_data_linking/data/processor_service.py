import argparse
import csv
import json
import random
import fitz
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Union, Optional

from paper_data_linking.data.models import MetadataRecord
from paper_data_linking.log_config import logger


class ExternalDataLoader:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.data = self.load_data()

    def load_data(self) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        raise NotImplementedError(
            "This method should be implemented in the child class."
        )

    def get_data_for_bibcode(self, bibcode: str) -> Dict:
        return self.data.get(bibcode, {})


class SOHOInstrumentLoader(ExternalDataLoader):
    def load_data(self) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        data = {}
        with self.filepath.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                instruments = [inst.strip() for inst in row["INTRUMENT"].split(",")]
                # Yes, instrument is misspelled intentionally here
                data[row["BIBCODE"]] = {
                    "instruments": instruments,
                }
        return data


class ProcessorService:
    def __init__(
        self,
        topics: Dict[str, Dict[str, Union[str, Path]]],
        pdfs_base_path: Path,
        external_data_loader: Optional[ExternalDataLoader] = None,
        pubdate_cutoff: Optional[str] = None,
        sample_size: Optional[int] = None,
        enriched_only: bool = True,
    ):
        self.pdfs_base_path = pdfs_base_path
        self.topics = topics
        self.external_data_loader = external_data_loader
        self.pubdate_cutoff = (
            datetime.strptime(pubdate_cutoff, "%Y-%m-%d") if pubdate_cutoff else None
        )  # Convert string to datetime object
        logger.info(f"Removing publications from before: {self.pubdate_cutoff}")
        self.sample_size = sample_size
        self.enriched_only = enriched_only

    def load_jsonl(self, filepath: Path) -> List[MetadataRecord]:
        """Load JSONL data into a list."""
        with open(filepath, "r", encoding="utf-8") as f:
            dict_records = [json.loads(line) for line in f]
        records = [MetadataRecord.from_dict(doc) for doc in dict_records]
        return records

    def process_data(self, topic: str, path: Path, label: str) -> List[Dict]:
        """Process data for a specific topic."""
        records = self.load_jsonl(path)
        processed_records = []
        for record in records:
            if self.pubdate_cutoff and record.pubdate > self.pubdate_cutoff:
                continue  # Skip the record if it's after the specified cutoff date
            pr = {"bibcode": record.bibcode, "topic": topic, "label": label}
            if self.external_data_loader:
                pr.update(self.external_data_loader.get_data_for_bibcode(record.bibcode))
            pr["pdf_path"] = str(self.pdfs_base_path / topic / f"{record.bibcode}.pdf")
            if Path(pr["pdf_path"]).exists():
                if self.is_valid_pdf(pr["pdf_path"]):
                    processed_records.append(pr)
                else:
                    logger.warning(f"Invalid PDF for {record.bibcode} at {pr['pdf_path']}.")
            else:
                logger.warning(f"PDF not found for {record.bibcode} at {pr['pdf_path']}.")

        return processed_records

    def is_valid_pdf(self, file_path: str) -> bool:
        """Check if a file is a valid PDF."""
        try:
            with fitz.open(file_path) as doc:
                if doc.page_count > 0:
                    return True
                else:
                    return False
        except Exception as e:
            logger.error(f"Error opening PDF file {file_path}: {e}")
            return False

    def run(self, output_path: Path) -> None:
        """Run the processing and save the results."""
        all_processed_data = []
        positive_bibcodes = set()
        positive_data = []
        negative_data = []

        # First, process positive topics and collect their bibcodes
        for topic, data in self.topics.items():
            if data["label"] == "positive":
                records = self.process_data(topic, data["path"], data["label"])
                positive_data.extend(records)
                positive_bibcodes.update([record["bibcode"] for record in records])

        if enriched_only:
            logger.info(
                f"Filtering to only positive cases which have been enriched with more data."
            )
            positive_data = [
                record
                for record in positive_data
                if self.external_data_loader.get_data_for_bibcode(record["bibcode"])
            ]

        # Then, process negative topics, but exclude bibcodes found in positive topics
        # We do this because we are getting negative examples by just looking at topics
        # We get positive records by getting specific papers which have been manually labeled positive
        # The negative records have not been manually labeled negative, so we are not completely sure
        # This is why we filter out new papers, because we assume SMEs have not yet had time to label them
        for topic, data in self.topics.items():
            if data["label"] == "negative":
                records = self.process_data(topic, data["path"], data["label"])
                filtered_records = [
                    record
                    for record in records
                    if record["bibcode"] not in positive_bibcodes
                ]
                negative_data.extend(filtered_records)

        # If sample size is specified, subsample the data
        if self.sample_size:
            logger.info(f"Subsampling data to {self.sample_size} records.")
            half_sample = self.sample_size // 2
            positive_data = random.sample(
                positive_data, min(half_sample, len(positive_data))
            )
            negative_data = random.sample(
                negative_data, min(half_sample, len(negative_data))
            )

        all_processed_data.extend(positive_data)
        all_processed_data.extend(negative_data)

        random.shuffle(all_processed_data)

        # Save processed data
        logger.info(f"Saving processed data to {output_path}.")
        with output_path.open("w", encoding="utf-8") as f:
            for record in all_processed_data:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for ML validation.")
    parser.add_argument(
        "--csv-path", type=Path, required=False, help="Path to the CSV data file."
    )
    parser.add_argument(
        "--topics",
        type=json.loads,
        required=True,
        help="JSON dictionary mapping topic to its data path and label.",
    )
    parser.add_argument(
        "--pdfs-base-path",
        type=Path,
        default=Path("./data/raw/pdfs"),
        help="Base path where PDFs are stored.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save the processed data.",
    )
    parser.add_argument(
        "--pubdate-cutoff",
        type=str,
        help="Latest publication date in the format 'YYYY-MM-DD'. Articles published after this date will be excluded.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of records to randomly sample (evenly split between positive and negative).",
        default=None,
    )
    parser.add_argument(
        "--no-enriched-only",
        action="store_true",
        default=False,
        help="Do not limit to only positive cases which have been enriched with more data.",
    )

    args = parser.parse_args()
    enriched_only = not args.no_enriched_only

    if args.csv_path:
        external_loader = SOHOInstrumentLoader(args.csv_path)
    else:
        external_loader = None
    service = ProcessorService(
        topics=args.topics,
        external_data_loader=external_loader,
        pdfs_base_path=args.pdfs_base_path,
        pubdate_cutoff=args.pubdate_cutoff,
        sample_size=args.sample_size,
        enriched_only=enriched_only,
    )
    service.run(args.output_path)
