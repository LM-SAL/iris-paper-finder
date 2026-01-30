import json
import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_bibcodes(file_path):
    """
    Load bibcodes from a file.
    """
    with open(file_path) as file:
        return [line.strip() for line in file]


def load_search_results(file_path):
    """
    Load search results from a JSON file.
    """
    with open(file_path) as file:
        return json.load(file)


def compare_keys(iris_bibcodes, search_results):
    """
    Compare bibcodes from the IRIS list with those in the search results.
    """
    iris_bibcodes = set(iris_bibcodes)
    true_bibcodes = set()
    false_bibcodes = set()
    uncertain_bibcodes = set()
    logger.info("Comparing IRIS bibcodes with search results...")
    logger.info("")
    print_per_bib = False
    for bibcode, result in search_results.items():
        iris_classification = result.get("iris_classification")
        if iris_classification:
            if "YES" in result.get("iris_classification"):
                if print_per_bib:
                    logger.info(f"Bibcode {bibcode} classified as YES.")
                true_bibcodes.add(bibcode)
            elif "UNCERTAIN" in result.get("iris_classification"):
                if print_per_bib:
                    logger.info(f"Bibcode {bibcode} classified as UNCERTAIN.")

                uncertain_bibcodes.add(bibcode)
            else:
                if print_per_bib:
                    logger.info(f"Bibcode {bibcode} classified as NO.")
                false_bibcodes.add(bibcode)
        else:
            logger.info(f"Bibcode {bibcode} has no iris_classification in search results.")
            logger.info("This bibcode will be added to the false.")
            false_bibcodes.add(bibcode)
    missing_positive_bibcodes = true_bibcodes - iris_bibcodes
    false_positive_bibcodes = false_bibcodes.intersection(iris_bibcodes)
    found_positive_bibcodes = true_bibcodes.intersection(iris_bibcodes)
    unavailable_bibcodes = iris_bibcodes - true_bibcodes - false_bibcodes
    return (
        true_bibcodes,
        missing_positive_bibcodes,
        false_bibcodes,
        false_positive_bibcodes,
        unavailable_bibcodes,
        found_positive_bibcodes,
        uncertain_bibcodes,
    )


if __name__ == "__main__":
    IRIS_LIST_BIBCODES_FILE = Path(__file__).parent.parent / "data" / "bibcodes" / "iris_library_bibcodes.txt"
    IRIS_SEARCH_RESULTS_FILE = "iris_search_results.json"
    IRIS_FAILED_DOWNLOADED = Path(__file__).parent.parent / "data" / "pdfs" / "iris_search" / "failed_bibcodes.txt"
    iris_library_bibcodes = load_bibcodes(IRIS_LIST_BIBCODES_FILE)
    failed_downloads = load_bibcodes(IRIS_FAILED_DOWNLOADED)
    if failed_downloads:
        logger.info(
            "Of the failed downloads, the number on the IRIS list is %d",
            len(set(failed_downloads).intersection(set(iris_library_bibcodes))),
        )
    search_results_bibcodes = load_search_results(IRIS_SEARCH_RESULTS_FILE)
    missing_bibcodes_from_library_based_on_query = set(iris_library_bibcodes) - set(search_results_bibcodes)
    logger.info(f"Missing bibcodes from IRIS list based on query: {missing_bibcodes_from_library_based_on_query}")
    logger.info(
        f"Number of missing bibcodes from IRIS list based on query: {len(missing_bibcodes_from_library_based_on_query)}"
    )
    (
        true_bibcodes,
        missing_positive_bibcodes,
        false_bibcodes,
        false_positive_bibcodes,
        unavailable_bibcodes,
        found_positive_bibcodes,
        uncertain_bibcodes,
    ) = compare_keys(iris_library_bibcodes, search_results_bibcodes)
    log_per_bib = False
    logger.info("In total, there are %d bibcodes classified by the LLM.", len(search_results_bibcodes))
    logger.info("In total, there are %d bibcodes in the IRIS list.", len(iris_library_bibcodes))
    logger.info("In total, there are %d bibcodes classified as positive by the LLM.", len(true_bibcodes))
    logger.info("In total, there are %d bibcodes classified as negative by the LLM.", len(false_bibcodes))
    logger.info("In total, there are %d bibcodes classified as uncertain by the LLM.", len(uncertain_bibcodes))
    logger.info("Comparison results:")
    if missing_positive_bibcodes:
        logger.info("Missing positive bibcodes:")
        for bibcode in sorted(missing_positive_bibcodes):
            logger.info(bibcode)
        logger.info(
            "There are %d classified positive bibcodes missing from the IRIS list.", len(missing_positive_bibcodes)
        )
    else:
        logger.info("No missing positive bibcodes.")
    logger.info("")
    logger.info("Comparison complete.")
