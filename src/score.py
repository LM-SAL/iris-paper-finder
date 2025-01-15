import pandas as pd
import argparse
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def extract_and_normalize(row):
    # Normalize the 'records' data
    records = pd.json_normalize(row['record'], 'records')

    # Check if 'records' is empty; if so, return None
    if records.empty:
        return None

    # Select the first record (as per your requirement)
    first_record = records.iloc[0]

    # Extract other needed fields
    row_data = {
        'uuid': row['uuid'],
        'bibcode': row['bibcode'],
        'docs': row['record'].get('docs'),
        'ocr_status': row['record'].get('ocr_status'),
        'analyzer': first_record.get('analyzer'),
        'passed_heuristic_filter': first_record.get('passed_heuristic_filter'),
        'data': first_record.get('data'),
        'WIND': first_record.get('data.WIND'),
        'instruments': first_record.get('data.instruments'),
        'analysis': first_record.get('analysis'),
        'relevant_indices': first_record.get('relevant_indices')
    }
    return row_data


def calculate_metrics(y_true, y_pred):
    """
    Calculates precision, recall, and accuracy based on true labels and predictions.

    :param y_true: Array-like, true labels.
    :param y_pred: Array-like, predicted labels.
    :return: Dictionary with precision, recall, accuracy, and F1 score.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': f1,
    }


def contains_omni(docs):
    if not isinstance(docs, list):
        return False
    for doc in docs:
        if 'page_content' in doc and " OMNI " in doc['page_content']:
            return True
    return False


def contains_string_in_docs(input_string, docs, relevant_indices=None):
    if not isinstance(docs, list):
        return False

    # If relevant_indices are provided, limit the search to those indices
    if relevant_indices is not None and isinstance(relevant_indices, list):
        docs_to_check = [docs[i] for i in relevant_indices if i < len(docs)]
    else:
        docs_to_check = docs

    for doc in docs_to_check:
        if 'page_content' in doc and input_string in doc['page_content']:
            return True
    return False


def check_string_in_docs(df, string_to_check, use_relevant_indices=False):
    # Apply the search function to the DataFrame
    df[f'{string_to_check.strip()}_in_docs'] = df.apply(
        lambda row: contains_string_in_docs(
            string_to_check, row['docs'],
            row['relevant_indices'] if use_relevant_indices else None
        ),
        axis=1
    )
    return df


def docs_to_text(docs):
    if not isinstance(docs, list):
        return ""
    text = ""
    for i, doc in enumerate(docs):
        if 'page_content' in doc:
            text += f"Excerpt {i + 1}"
            text += "\n\n"
            text += doc['page_content']
            text += "\n\n"
    return text


def create_subset_based_on_bibcodes(existing_file_path, bibcodes):
    # Load the existing file into a DataFrame
    existing_df = pd.read_json(existing_file_path, lines=True)

    # Filter the DataFrame to only include rows with bibcodes from `fn_df`
    subset_df = existing_df[existing_df['bibcode'].isin(bibcodes)]

    # Save the subset to a new file
    subset_file_path = existing_file_path.replace('.jsonl', '_subset.jsonl')
    subset_df.to_json(subset_file_path, orient='records', lines=True)

    print(f'Subset created and saved to {subset_file_path}')


def overwrite_analysis_data(original_analysis_df, additional_file_path):
    # Read the additional file
    additional_df = pd.read_json(additional_file_path, lines=True)

    # Use the additional file data to overwrite the original analysis data
    # We will identify records by 'bibcode' and overwrite the analysis data for those records
    combined_df = pd.concat([original_analysis_df, additional_df])
    combined_df = combined_df.drop_duplicates(subset='bibcode', keep='last')

    return combined_df


def main(analysis_file, true_labels_file, additional_file_path=None):
    # Read the analysis file
    analysis_df = pd.read_json(analysis_file, lines=True)

    # If an additional file is provided, overwrite the analysis data
    if additional_file_path:
        analysis_df = overwrite_analysis_data(analysis_df, additional_file_path)

    # Flatten the data
    flattened_data = [extract_and_normalize(row) for index, row in analysis_df.iterrows()]
    flattened_df = pd.DataFrame(flattened_data)
    flattened_df["text"] = flattened_df["docs"].apply(docs_to_text)

    # Read the true labels file
    true_labels_df = pd.read_json(true_labels_file, lines=True)

    # If an additional true labels file is provided, overwrite the true labels
    # Merge the dataframes on 'bibcode'
    merged_df = pd.merge(flattened_df, true_labels_df, on='bibcode', how='inner')

    # Rename columns
    merged_df = merged_df.rename(columns={'WIND': 'pred_wind', 'label': 'true_wind'})

    # Standardize formats and map to numeric values
    format_mapping = {'YES': 1, 'NO': 0, 'positive': 1, 'negative': 0}
    merged_df['pred_wind'] = merged_df['pred_wind'].map(format_mapping).fillna(0)
    merged_df['true_wind'] = merged_df['true_wind'].map(format_mapping)

    # merged_df['OMNI_in_docs'] = merged_df['docs'].apply(contains_omni)
    # remove rows where OMNI is in the docs

    # Define new column order
    new_column_order = [
        'uuid', 'bibcode', 'pred_wind', 'true_wind', 'instruments',
        'topic', 'ocr_status', 'passed_heuristic_filter', 'data',
        'analysis', 'text', 'docs', 'relevant_indices', 'analyzer', 'pdf_path'
    ]

    # Reorder the columns
    merged_df = merged_df[new_column_order]
    merged_df = check_string_in_docs(merged_df, "OMNI", use_relevant_indices=False)
    merged_df = check_string_in_docs(merged_df, " WIND ", use_relevant_indices=False)

    # Display the first few rows of the merged DataFrame
    # print(merged_df.head())
    fn_df = merged_df[merged_df['true_wind'] == 1]
    fn_df = fn_df[fn_df['pred_wind'] == 0]
    fn_df = fn_df[fn_df['OMNI_in_docs'] == False]

    # Extract bibcodes from the `fn_df` DataFrame
    fn_bibcodes = fn_df['bibcode'].tolist()
    # Call the subset creation function
    # create_subset_based_on_bibcodes(true_labels_file, fn_bibcodes)

    merged_df = merged_df[merged_df['OMNI_in_docs'] == False]
    # Extract the true and predicted labels
    y_true = merged_df['true_wind']
    y_pred = merged_df['pred_wind']

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)

    # Print the metrics
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Score: {metrics['f1_score']}")

    print('Done.')


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Merge analysis results with true labels and score the performance.')

    # Add arguments
    parser.add_argument('analysis_file', type=str, help='The file path to the analysis results JSONL file.')
    parser.add_argument('true_labels_file', type=str, help='The file path to the true labels JSONL file.')
    # Add an optional argument for the path to the additional file
    parser.add_argument('--additional_file', type=str, default=None,
                        help='Optional file path to an additional JSONL file to overwrite analysis data.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the scoring function with provided arguments
    main(args.analysis_file, args.true_labels_file, args.additional_file)
