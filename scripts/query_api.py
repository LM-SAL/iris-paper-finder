import logging
import os
import requests
import time
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def analyze_paper(
        pdf_path: str,
        config_name: str = "iris_config.yaml",
        base_url: str = "http://localhost:80",
        username: str = "your_username",
        password: str = "your_password"
):
    """
    Analyze a paper using the paper-data-linking API.

    Args:
        pdf_path: Path to the PDF file
        config_name: Name of config file to use (e.g. "iris_config.yaml")
        base_url: Base URL of the API
        username: Nginx auth username
        password: Nginx auth password
    """
    logging.info("Starting paper analysis...")

    # Create a session with authentication
    session = requests.Session()
    session.auth = (username, password)

    # Verify config is available
    logging.info("Fetching available configs...")
    configs = session.get(f"{base_url}/api/get_configs")
    if not configs.ok:
        logging.error(f"Failed to get configs: {configs.status_code}")
        raise Exception(f"Failed to get configs: {configs.status_code}")

    available_configs = [c['filename'] for c in configs.json()["configs"]]
    if config_name not in available_configs:
        logging.error(f"Config {config_name} not found in available configs: {available_configs}")
        raise Exception(f"Config {config_name} not found in available configs: {available_configs}")
    logging.info(f"Using config: {config_name}")

    # Open file and prepare multipart form data
    logging.info("Uploading the PDF file and starting analysis...")
    with open(pdf_path, 'rb') as f:
        files = {
            'file': (Path(pdf_path).name, f, 'application/pdf')
        }
        data = {
            'config_file': config_name
        }

        response = session.post(
            f"{base_url}/api/highlight_pdf",
            files=files,
            data=data
        )

    if not response.ok:
        logging.error(f"Upload failed: {response.status_code}, {response.text}")
        raise Exception(f"Upload failed: {response.status_code}, {response.text}")

    task_id = response.json()["task_id"]
    logging.info(f"Started task with ID: {task_id}")

    # Poll for results
    while True:
        status = session.get(f"{base_url}/api/task/{task_id}")
        if not status.ok:
            logging.error(f"Status check failed: {status.status_code}")
            raise Exception(f"Status check failed: {status.status_code}")

        result = status.json()
        if result["task_status"] == "SUCCESS":
            logging.info("Analysis successfully completed.")
            return result["task_result"]
        elif result["task_status"] == "FAILURE":
            logging.error("Analysis failed.")
            raise Exception("Analysis failed")

        progress = result.get("task_result", {}).get("current", "unknown")
        logging.info(f"Task in progress: {progress}")
        time.sleep(2)  # Wait before checking again


def summarize_results(result):
    """
    Summarizes the analysis results.

    Args:
        result: The JSON result from the analysis.
    """
    logging.info("Summarizing results...")
    analysis = result.get('analysis', 'No analysis available.')
    summary = result['data']

    logging.info(f"File Analyzed: {summary.get('filename', 'Unknown')}")
    logging.info(f"Task Timestamp: {summary.get('timestamp', 'Unknown')}")
    logging.info(f"Highlights Count: {len(result.get('highlights', []))}")

    results = summary.get('results', [])
    logging.info(f"Total Results Count: {len(results)}")
    for i, res in enumerate(results):
        analyzer = res.get('analyzer', 'Unknown Analyzer')
        passed = res.get('passed_heuristic_filter', False)
        data = res.get('data', {})
        logging.info(f"\nResult {i + 1}:")
        logging.info(f"  Analyzer: {analyzer}")
        logging.info(f"  Passed Heuristic Filter: {passed}")
        logging.info(f"  IRIS Classification: {data.get('IRIS', 'Unknown')}")
        logging.info(f"  Aspects: {', '.join([a['label'] for a in data.get('aspects', [])])}")

    logging.info("Detailed analysis:")
    print(analysis)


if __name__ == "__main__":
    # Example usage
    pdf_file = "Cho_2024_ApJ_975_33.pdf"
    try:
        result = analyze_paper(
            pdf_file,
            config_name="iris_config.yaml",
            username=os.getenv("USERNAME"),
            password=os.getenv("PASSWORD"),
            # fill in USERNAME and PASSWORD values in .env file in root of repo.
            # These values should match those set in Step 3 of the README.
        )

        summarize_results(result)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
