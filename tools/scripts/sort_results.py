import argparse
import os
import sys

import pandas as pd


def sort_results(csv_path, output_path=None):
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Columns to sort by
    sort_cols = [
        "threads",
        "write_batch_size",
        "read_batch_size",
        "dataset_name",
        "with_lock",
        "use_node_lock",
        "insert_input_rate",
        "search_input_rate",
        "index_params",
        "query_params",
    ]

    missing_cols = [col for col in sort_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Warning: The following columns are missing from the CSV and will be ignored in sorting: {missing_cols}"
        )
        sort_cols = [col for col in sort_cols if col in df.columns]

    if not sort_cols:
        print("Error: No valid columns to sort by.")
        sys.exit(1)

    df_sorted = df.sort_values(by=sort_cols)

    if output_path is None:
        output_path = csv_path

    try:
        df_sorted.to_csv(output_path, index=False)
        print(f"Successfully sorted results and saved to: {output_path}")
    except Exception as e:
        print(f"Error writing CSV: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort benchmark results CSV.")
    parser.add_argument("csv_path", help="Path to the input CSV file.")
    parser.add_argument(
        "--output",
        help="Path to the output CSV file. If not provided, overwrites input.",
    )

    args = parser.parse_args()

    sort_results(args.csv_path, args.output)
