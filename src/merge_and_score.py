import json
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def read_jsonl(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            bibcode = entry['bibcode']
            data[bibcode] = entry
    return data


def merge_and_score(analysis_file, true_labels_file):
    # Read the files and create dictionaries
    analysis_results = read_jsonl(analysis_file)
    true_labels = read_jsonl(true_labels_file)

    # Lists to hold the predicted labels and true labels
    y_pred = []
    y_true = []

    # Merge the data based on bibcode and evaluate
    for bibcode, analysis in analysis_results.items():
        if bibcode in true_labels:
            # Convert "NO" and "YES" to binary values for comparison
            # predicted_label = 0 if analysis['data']['SOHO'] == 'NO' else 1
            if analysis['record']['records'][0]['data'] is None:
                predicted_label = 0
            else:
                predicted_label = 1 if analysis['record']['records'][0]['data']['WIND'] == 'YES' else 0
            true_label = 1 if true_labels[bibcode]['label'] == 'positive' else 0

            y_pred.append(predicted_label)
            y_true.append(true_label)

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    # Print out the scores
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Merge analysis results with true labels and score the performance.')

    # Add arguments
    parser.add_argument('analysis_file', type=str, help='The file path to the analysis results JSONL file.')
    parser.add_argument('true_labels_file', type=str, help='The file path to the true labels JSONL file.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the scoring function with provided arguments
    merge_and_score(args.analysis_file, args.true_labels_file)
