import json
import time
import logging
import traceback
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_paper(
    pdf_path: str,
    config_name: str = "iris_config.yaml",
    base_url: str = "http://localhost:80",
):
    """
    Analyze a paper using the paper-data-linking API.

    Args:
        pdf_path: Path to the PDF file
        config_name: Name of config file to use (e.g. "iris_config.yaml")
        base_url: Base URL of the API
    """
    logger.info("Starting paper analysis...")
    session = requests.Session()
    logger.info("Fetching available configs...")
    configs = session.get(f"{base_url}/api/get_configs")
    if not configs.ok:
        logger.error(f"Failed to get configs: {configs.status_code}")
        msg = f"Failed to get configs: {configs.status_code}"
        raise Exception(msg)
    available_configs = [c["filename"] for c in configs.json()["configs"]]
    if config_name not in available_configs:
        logger.error(f"Config {config_name} not found in available configs: {available_configs}")
        msg = f"Config {config_name} not found in available configs: {available_configs}"
        raise Exception(msg)
    logger.info(f"Using config: {config_name}")
    logger.info("Uploading the PDF file and starting analysis...")
    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f, "application/pdf")}
        data = {"config_file": config_name}
        response = session.post(f"{base_url}/api/highlight_pdf", files=files, data=data)
    if not response.ok:
        logger.error(f"Upload failed: {response.status_code}, {response.text}")
        msg = f"Upload failed: {response.status_code}, {response.text}"
        raise Exception(msg)
    task_id = response.json()["task_id"]
    logger.info(f"Started task with ID: {task_id}")
    while True:
        status = session.get(f"{base_url}/api/task/{task_id}")
        if not status.ok:
            logger.error(f"Status check failed: {status.status_code}")
            msg = f"Status check failed: {status.status_code}"
            raise Exception(msg)
        result = status.json()
        if result["task_status"] == "SUCCESS":
            logger.info("Analysis successfully completed.")
            return result["task_result"]
        if result["task_status"] == "FAILURE":
            logger.error("Analysis failed.")
            msg = "Analysis failed"
            raise Exception(msg)
        progress = (result.get("task_result") or {}).get("current", "unknown")
        logger.info(f"Task in progress: {progress}")
        time.sleep(5)


def summarize_results(result):
    """
    Summarizes the analysis results.

    Args:
        result: The JSON result from the analysis.
    """
    logger.info("Summarizing results...")
    result.get("analysis", "No analysis available.")
    summary = result["data"]
    logger.info(f"File Analyzed: {summary.get('filename', 'Unknown')}")
    logger.info(f"Task Timestamp: {summary.get('timestamp', 'Unknown')}")
    logger.info(f"Highlights Count: {len(result.get('highlights', []))}")
    results = summary.get("results", [])
    logger.info(f"Total Results Count: {len(results)}")
    for i, res in enumerate(results):
        analyzer = res.get("analyzer", "Unknown Analyzer")
        passed = res.get("passed_heuristic_filter", False)
        data = res.get("data", {})
        logger.info(f"\nResult {i + 1}:")
        logger.info(f"  Analyzer: {analyzer}")
        logger.info(f"  Passed Heuristic Filter: {passed}")
        logger.info(f"  IRIS Classification: {data.get('IRIS', 'Unknown')}")
        logger.info(f"  Aspects: {', '.join([a['label'] for a in data.get('aspects', [])])}")
    logger.info("Detailed analysis:")


def append_dict_to_json(file_path, new_data):
    """
    Appends a dictionary to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        new_data (dict): The dictionary to append.
    """
    try:
        with open(file_path, "r+") as file:
            existing_data = json.load(file)
            existing_data.update(new_data)
            file.seek(0)
            json.dump(existing_data, file, indent=4)
            file.truncate()
    except FileNotFoundError:
        with open(file_path, "w+") as file:
            json.dump(new_data, file, indent=4)


def save_results(result, results_file):
    """
    Saves the analysis results.

    Args:
        result: The JSON result from the analysis.
        results_file: The file to save the results to.
    """
    summary = result["data"]
    final_results = summary.get("results", [])[0]
    data = final_results.get("data")
    if data is None:
        logger.warning("No data found in the result.")
        logger.warning(f"Summary: {summary}")
        results_dict = {
            summary.get("filename").split(".pdf")[0]: {
                "passed_heuristic_filter": final_results.get("passed_heuristic_filter", ""),
                "iris_classification": "NO",
                "aspects": "None",
                "details": result.get("analysis", "No analysis available."),
            }
        }
    else:
        if data.get("aspects") == []:
            data["aspects"] = [{"label": "None"}]
        results_dict = {
            summary.get("filename").split(".pdf")[0]: {
                "passed_heuristic_filter": final_results.get("passed_heuristic_filter", ""),
                "iris_classification": data.get("IRIS", ""),
                "aspects": ", ".join([data.get("aspects")[0]["label"]]),
                "details": result.get("analysis", "No analysis available."),
            }
        }
    append_dict_to_json(results_file, results_dict)


if __name__ == "__main__":
    import argparse
    from glob import glob

    from tqdm import tqdm

    path = argparse.ArgumentParser()
    path.add_argument("--pdf_path", type=str, required=True, help="Path to the PDF directory")
    args = path.parse_args()
    pdf_path = args.pdf_path
    pdf_files = sorted(glob(f"{pdf_path}/*.pdf"))
    results_name = pdf_path.split("/")[-1]
    results_file = f"{results_name}_results.json"
    failed_files = []
    for pdf_file in tqdm(pdf_files):
        try:
            result = analyze_paper(
                pdf_file,
                config_name="iris_config_v2.yaml",
            )
            save_results(result, results_file=results_file)
        except Exception:
            logger.exception(f"An error occurred with {pdf_file}")
            logger.exception(traceback.format_exc())
            failed_files.append(pdf_file)
    if failed_files:
        logger.exception(f"Failed files: {failed_files}")
