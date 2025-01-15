import json
from pathlib import Path

import pandas as pd
from scipy.stats import sem, t
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from paper_data_linking.utils import INSTRUMENT_ACRONYM_EXPANSIONS

DATA_DIR = Path(__file__).parent.parent / "data"

VALID_ACRONYMS = [acronym for acronym, _ in INSTRUMENT_ACRONYM_EXPANSIONS]


def read_results(file_path):
    results = []
    with open(file_path, "r") as file:
        for line in file:
            results.append(json.loads(line.strip()))
    return pd.DataFrame(results)


def calculate_confidence_interval(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    std_err = sem(y_pred == y_true)
    confidence_interval = t.interval(0.95, len(y_pred)-1, loc=accuracy, scale=std_err)
    return accuracy, confidence_interval


def classify_bibcodes(results):
    false_positives = results[(results.true_label == "NO") & (results.predicted_label == "YES")]["bibcode"].tolist()
    false_negatives = results[(results.true_label == "YES") & (results.predicted_label == "NO")]["bibcode"].tolist()
    return false_positives, false_negatives


def create_symlinks(bibcodes, data_dir, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    for bibcode in bibcodes:
        source = data_dir / "soho" / "pdf" / f"{bibcode}.pdf"
        target = target_dir / f"{bibcode}.pdf"
        if not source.exists():
            # source = data_dir / "non_soho" / "pdf" / f"{bibcode}.pdf"
            source = data_dir / "unsure_solar" / "pdf_old" / f"{bibcode}.pdf"
        if source.exists() and not target.exists():
            target.symlink_to(source)


def write_classification_report_to_file(report, filename="classification_report.txt"):
    with open(filename, "w") as f:
        f.write(report)


def evaluate_results(results, data_dir):
    # Filter out 'UNCERTAIN' labels
    results = results[results.predicted_label != "UNCERTAIN"]
    uncertain_count = len(results[results.predicted_label == "UNCERTAIN"])

    y_true = results["true_label"]
    y_pred = results["predicted_label"]

    accuracy, confidence_interval = calculate_confidence_interval(y_true, y_pred)

    report = classification_report(y_true, y_pred, target_names=["NO", "YES"])

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["NO", "YES"]).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    false_positives, false_negatives = classify_bibcodes(results)

    create_symlinks(false_positives, data_dir, Path("./false_positives"))
    create_symlinks(false_negatives, data_dir, Path("./false_negatives"))

    # Construct output text
    output = f"Accuracy: {accuracy}\n"
    output += f"Confidence interval: {confidence_interval}\n"
    output += f"TPR: {tpr}\n"
    output += f"FPR: {fpr}\n"
    output += f"Uncertain count: {uncertain_count}\n"
    output += f"\nClassification Report:\n"
    output += report

    write_classification_report_to_file(output)


if __name__ == "__main__":
    results = read_results("results.jsonl")
    evaluate_results(results, DATA_DIR)
