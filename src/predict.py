import argparse
import json
import logging
# from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from fitz.fitz import FileDataError
from uuid import uuid4

import pandas as pd
from tqdm import tqdm

from paper_data_linking.process.analyzer import get_analyzer
from paper_data_linking.utils import AnalysisResult

LOG = logging.getLogger(__name__)


class AnalyzerHandler:
    def __init__(self, out_path, config_path):
        self.out_path = out_path
        self.config_path = config_path
        self.processed_files = self.read_existing_results()

    def read_existing_results(self):
        if not self.out_path.exists():
            return set()

        df = pd.read_json(self.out_path, lines=True)
        return set(df["bibcode"])

    def process_file(self, file_path: Path, bibcode: str, my_uuid="0"):
        if bibcode in self.processed_files:
            return

        try:
            LOG.warning(f"Processing {bibcode}")
            analyzer = get_analyzer([self.config_path])
            record = analyzer.process(file_path)
            LOG.warning(f"Finished processing {bibcode}")
            result = AnalysisResult(uuid=my_uuid, bibcode=bibcode, record=record)
            LOG.warning("returning result")
            return result
        except (FileNotFoundError, FileDataError) as e:
            LOG.warning(f"Failed on {bibcode} due to error: {e}")
            return

    def write_result(self, file, result: AnalysisResult):
        """Write the result to the file."""
        file.write(result.json() + "\n")

    def process_files(self, df, n_workers=4, my_uuid="0"):
            if n_workers > 1:
                raise NotImplementedError("Multiprocessing not implemented yet.")
                # with ProcessPoolExecutor(max_workers=n_workers) as executor:
                #     futures = {
                #         executor.submit(
                #             self.process_file,
                #             row["pdf_path"],
                #             row["bibcode"],
                #             my_uuid,
                #         )
                #         for _, row in df.iterrows()
                #     }
                #
                #     for future in tqdm(as_completed(futures), total=len(df)):
                #         result: AnalysisResult = future.result()
                #         if result:
                #             with open(self.out_path, "a") as f0:
                #                 self.write_result(f0, result)
            else:
                # Processing in the main thread for a single worker
                for _, row in tqdm(df.iterrows(), total=len(df)):
                    if row['label'] == "positive":
                        LOG.warning(f"Processing {row['bibcode']}")
                    result = self.process_file(row["pdf_path"], row["bibcode"], my_uuid)
                    LOG.warning(f"Got result.")
                    if result:
                        with open(self.out_path, "a") as f0:
                            self.write_result(f0, result)


def main(infile: Path, out_results: Path, config: Path, n_workers=4):
    df = pd.read_json(infile, lines=True)
    handler = AnalyzerHandler(out_results, config)
    my_uuid = str(uuid4())
    handler.process_files(df, my_uuid=my_uuid, n_workers=n_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for content analysis.")
    parser.add_argument("i", help="input processed data jsonl", type=Path)
    parser.add_argument("o", help="output results jsonl", type=Path)
    parser.add_argument("c", help="config file", type=Path)
    parser.add_argument("--num-workers", help="number of workers", type=int, default=1)
    args = parser.parse_args()
    main(args.i, args.o, args.c, args.num_workers)
