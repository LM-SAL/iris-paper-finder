import argparse
import seaborn as sns

from sklearn.utils import resample
from scipy import stats
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
import matplotlib
from sklearn.preprocessing import MultiLabelBinarizer

from paper_data_linking.utils import INSTRUMENT_ACRONYM_EXPANSIONS

VALID_ACRONYMS = [acronym for acronym, _ in INSTRUMENT_ACRONYM_EXPANSIONS]
matplotlib.rcParams.update({'font.size': 14})


def generate_sankey_text(df):
    # Initialize counters
    total = len(df)
    soho = len(df[df['soho_true'] == 1])
    not_soho = len(df[df['soho_true'] == 0])

    # For SOHO
    pred_soho = len(df[(df['soho_true'] == 1) & (df['soho_pred'] == 1)])
    not_pred_soho = len(df[(df['soho_true'] == 1) & (df['soho_pred'] == 0)])

    # For Not SOHO
    solar = len(df[(df['soho_true'] == 0) & (df['topic'] == 'solar')])
    cosmology = len(df[(df['soho_true'] == 0) & (df['topic'] == 'cosmology')])

    # For Solar
    solar_pred_soho = len(
        df[(df['soho_true'] == 0) & (df['topic'] == 'solar') & (df['soho_pred'] == 1)])
    solar_not_pred_soho = len(
        df[(df['soho_true'] == 0) & (df['topic'] == 'solar') & (df['soho_pred'] == 0)])

    # For Cosmology
    cosmo_pred_soho = len(df[(df['soho_true'] == 0) & (df['topic'] == 'cosmology') & (
                df['soho_pred'] == 1)])
    cosmo_not_pred_soho = len(df[(df['soho_true'] == 0) & (
                df['topic'] == 'cosmology') & (df['soho_pred'] == 0)])

    # Prepare the text for SankeyMATIC
    text = f'Total [{soho}] SOHO\nTotal [{not_soho}] Not_SOHO\n'
    text += f'SOHO [{pred_soho}] Predicted_SOHO\nSOHO [{not_pred_soho}] Not_Predicted_SOHO\n'
    text += f'Not_SOHO [{solar}] Solar\nNot_SOHO [{cosmology}] Cosmology\n'
    text += f'Solar [{solar_pred_soho}] Predicted_SOHO\nSolar [{solar_not_pred_soho}] Not_Predicted_SOHO\n'
    text += f'Cosmology [{cosmo_pred_soho}] Predicted_SOHO\nCosmology [{cosmo_not_pred_soho}] Not_Predicted_SOHO'
    # Build from string with https://sankeymatic.com/build/

    return text


def calculate_precision_recall(merged_df):
    # For precision and recall, we first need to binarize labels
    mlb = MultiLabelBinarizer()

    # Fitting on all unique instrument labels present in the data
    mlb.fit(
        merged_df["instruments_true"].tolist() + merged_df["instruments_pred"].tolist()
    )

    # Transforming the instrument labels
    y_true_bin = mlb.transform(merged_df["instruments_true"].tolist())
    y_pred_bin = mlb.transform(merged_df["instruments_pred"].tolist())

    try:
        # Calculate precision and recall for each class
        precision = precision_score(y_true_bin, y_pred_bin, average=None)
        recall = recall_score(y_true_bin, y_pred_bin, average=None)
        precision_dict = dict(zip(mlb.classes_, precision))
        recall_dict = dict(zip(mlb.classes_, recall))
        return precision_dict, recall_dict, mlb.classes_
    except ValueError as e:
        print(
            f"Couldn't calculate precision and recall due to the following issue: {e}"
        )
        return np.nan, np.nan, None


def write_report(report_text, filename):
    with open(filename, "w") as f:
        f.write(report_text)


def print_metrics(df, data_label, output_dir):
    report = []

    report.append(f"\nFor {data_label} data:\n")
    accuracy, y_true, y_pred = calculate_accuracy(df)
    margin_error = calculate_accuracy_confidence_interval(y_true, y_pred)
    report.append(f"Accuracy: {accuracy:.3f} ± {margin_error:.3f}\n")
    f1_micro, f1_macro = calculate_f1_scores(df)
    report.append(f"Micro-average F1 score: {f1_micro}\n")
    report.append(f"Macro-average F1 score: {f1_macro}\n")
    precision_dict, recall_dict, classes = calculate_precision_recall(df)
    report.append(f"Precision for each class: {precision_dict}\n")
    report.append(f"Recall for each class: {recall_dict}\n")
    report_text = "".join(report)

    write_report(report_text, f"{output_dir}/{data_label}_report.txt")
    plot_precision_recall(precision_dict, recall_dict, classes, data_label, true_labels=df["instruments_true"], output_dir=output_dir)

    sankey_txt = generate_sankey_text(df)
    write_report(sankey_txt, output_dir / f"{data_label}_sankey.txt")


def plot_precision_recall(precision_dict, recall_dict, classes, data_label, true_labels, output_dir):
    precision_values = list(precision_dict.values())
    recall_values = list(recall_dict.values())

    # Get counts of true labels for each instrument
    label_counts = true_labels.explode().value_counts().to_dict()

    # Convert counts to sizes for plotting
    sizes = np.array([label_counts.get(cls, 1) for cls in classes])
    sizes = sizes * 1.2

    plt.figure(figsize=(10, 6), dpi=200)
    sns.set_style("whitegrid")  # set seaborn style to 'whitegrid'
    plt.scatter(recall_values, precision_values, marker="o", s=sizes, alpha=0.5)  # Swapped recall and precision
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    for i, txt in enumerate(classes):
        plt.annotate(txt, (recall_values[i], precision_values[i]))  # Swapped recall and precision
    plt.title(f"Precision vs Recall for {data_label} data")
    plt.xlabel("Recall")  # Set x-label to "Recall"
    plt.ylabel("Precision")  # Set y-label to "Precision"
    plt.savefig(
        f"{output_dir}/{data_label}_precision_recall_plot.png"
    )  # Saving the figure
    plt.close()  # Closing the figure to prevent it from being displayed



def load_data(in_preds, in_labels):
    # Load both files into dataframes
    predictions_df = pd.read_json(in_preds, lines=True)
    labeled_df = pd.read_json(in_labels, lines=True)
    return predictions_df, labeled_df


def process_predictions_df(predictions_df):
    # Extract SOHO, instruments, ocr, and analyzer data from 'results'
    predictions_df["soho"] = predictions_df["results"].apply(
        lambda x: 1 if x["SOHO"] == "YES" else 0
    )
    predictions_df["instruments"] = predictions_df["results"].apply(
        lambda x: [
            i["instrument"]
            for i in x["instruments"]
            if i["instrument"] in VALID_ACRONYMS
        ]
    )
    predictions_df["ocr"] = predictions_df["results"].apply(lambda x: x["ocr"])
    predictions_df["analyzer"] = predictions_df["results"].apply(
        lambda x: x["analyzer"]
    )

    # Drop the 'results' column as it's no longer needed
    predictions_df.drop(columns=["results"], inplace=True)
    return predictions_df


def merge_dataframes(labeled_df, predictions_df):
    # Merge the dataframes
    merged_df = pd.merge(
        labeled_df, predictions_df, on="bibcode", suffixes=("_true", "_pred")
    )
    return merged_df


# def calculate_accuracy(merged_df):
#     # Calculate accuracy for the SOHO label
#     accuracy = accuracy_score(merged_df["soho_true"], merged_df["soho_pred"])
#     return accuracy

def calculate_accuracy(merged_df):
    # Calculate accuracy for the SOHO label
    accuracy = accuracy_score(merged_df["soho_true"], merged_df["soho_pred"])
    return accuracy, merged_df["soho_true"], merged_df["soho_pred"]


def calculate_accuracy_confidence_interval(y_true, y_pred, confidence=0.95):
    # Perform bootstrap resampling and calculate accuracies
    n_iterations = 1000  # Number of bootstrap samples to create
    n_size = int(len(y_true) * 0.50)  # Size of a bootstrap sample
    y_true = y_true.values
    y_pred = y_pred.values
    accuracies = list()
    for _ in range(n_iterations):
        indices = resample(np.arange(len(y_true)), n_samples=n_size)  # Resample indices
        score = accuracy_score(y_true[indices], y_pred[indices])  # Calculate accuracy
        accuracies.append(score)

    # Calculate the standard error (SE)
    se = np.std(accuracies)

    # Calculate the margin of error (ME)
    me = se * stats.norm.ppf((1 + confidence) / 2.)

    return me


def calculate_f1_scores(merged_df):
    # For F1-score, we first need to binarize labels
    mlb = MultiLabelBinarizer()

    # Fitting on all unique instrument labels present in the data
    mlb.fit(
        merged_df["instruments_true"].tolist() + merged_df["instruments_pred"].tolist()
    )

    # Transforming the instrument labels
    y_true_bin = mlb.transform(merged_df["instruments_true"].tolist())
    y_pred_bin = mlb.transform(merged_df["instruments_pred"].tolist())

    try:
        # Calculate micro-average F1 score
        f1_micro = f1_score(y_true_bin, y_pred_bin, average="micro")
        # Calculate macro-average F1 score
        f1_macro = f1_score(y_true_bin, y_pred_bin, average="macro")
        return f1_micro, f1_macro
    except ValueError as e:
        print(f"Couldn't calculate F1 scores due to the following issue: {e}")
        return np.nan, np.nan


def split_by_ocr(merged_df):
    ocr_df = merged_df[merged_df["ocr"] == 1]
    non_ocr_df = merged_df[merged_df["ocr"] == 0]
    return ocr_df, non_ocr_df


def split_by_analyzer(df):
    fuzzy_filter_df = df[df["analyzer"] == "fuzzy_filter"]
    soho_classifier_df = df[df["analyzer"] == "soho_classifier"]
    return fuzzy_filter_df, soho_classifier_df


def main(output_dir, processed_data_dir):

    in_preds = processed_data_dir / "results.jsonl"
    in_labels = processed_data_dir / "papers_labeled.jsonl"

    predictions_df, labeled_df = load_data(in_preds, in_labels)
    predictions_df = process_predictions_df(predictions_df)
    merged_df = merge_dataframes(labeled_df, predictions_df)

    ocr_df, non_ocr_df = split_by_ocr(merged_df)

    print_metrics(merged_df, "full", output_dir)
    print_metrics(ocr_df, "OCR", output_dir)
    print_metrics(non_ocr_df, "non-OCR", output_dir)

    fuzzy_filter_df, soho_classifier_df = split_by_analyzer(merged_df)
    print_metrics(fuzzy_filter_df, "fuzzy_filter", output_dir)
    print_metrics(soho_classifier_df, "soho_classifier", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions and generate report"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the reports and plots",
    )
    parser.add_argument(
        "--processed_data_dir",
        type=Path,
        required=True,
        help="Directory with processed data",
    )
    args = parser.parse_args()

    # PROCESSED_DATA_DIR = Path(args.processed_data_dir)

    main(args.output_dir, args.processed_data_dir)
