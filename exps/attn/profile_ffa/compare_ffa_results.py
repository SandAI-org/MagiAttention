# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys

import pandas as pd

# Define configurations for different test modes
# key_columns: Unique identifier columns to match the same test case in baseline and target files
# start_compare_column: The column from which to start calculating performance differences
CONFIGS = {
    "dense": {
        "key_columns": ["mask_type", "direction", "seqlen"],
        "start_compare_column": "tflops_per_sec",
    },
    "block_sparse": {
        "key_columns": ["seqlen", "sparsity_ratio", "block_size", "direction"],
        "start_compare_column": "latency_ms",
    },
}


def detect_mode(columns):
    """Detects the comparison mode based on the CSV file's columns."""
    # Create sets of the unique keys for each mode
    block_sparse_keys = set(CONFIGS["block_sparse"]["key_columns"])
    dense_keys = set(CONFIGS["dense"]["key_columns"])

    column_set = set(columns)

    # Check for the more specific key set first (block_sparse has more keys)
    if block_sparse_keys.issubset(column_set):
        return "block_sparse"
    elif dense_keys.issubset(column_set):
        return "dense"
    else:
        return None


def run_comparison(mode, baseline_path, target_path, output_path):
    """
    Compares performance metrics of two CSV files based on the specified mode
    and saves the results to an output file.

    Args:
        mode (str): The test mode ('dense' or 'block_sparse').
        baseline_path (str): Path to the baseline results CSV file.
        target_path (str): Path to the target results CSV file.
        output_path (str): Path for the output CSV file to save comparison results.
    """
    # 1. Select configuration based on the mode
    if mode not in CONFIGS:
        print(
            f"Error: Invalid mode '{mode}'. Please choose from {list(CONFIGS.keys())}."
        )
        sys.exit(1)

    config = CONFIGS[mode]
    key_columns = config["key_columns"]
    start_column = config["start_compare_column"]

    print(f"--- Running comparison in '{mode}' mode ---")
    print(f"Key columns for matching: {key_columns}")

    # 2. Read the CSV files
    try:
        df_baseline = pd.read_csv(baseline_path)
        df_target = pd.read_csv(target_path)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the files: {e}")
        sys.exit(1)

    # 3. Check if key columns exist
    for col in key_columns:
        if col not in df_baseline.columns or col not in df_target.columns:
            print(
                f"Error: Key column '{col}' required for mode '{mode}' not found in one of the files."
            )
            sys.exit(1)

    # 4. Merge the two DataFrames
    df_merged = pd.merge(
        df_baseline, df_target, on=key_columns, suffixes=("_baseline", "_target")
    )

    if df_merged.empty:
        print("Error: No matching rows found between the two files.")
        print(
            f"Please ensure that combinations of '{', '.join(key_columns)}' exist in both files."
        )
        sys.exit(1)

    # 5. Determine the columns to compare
    try:
        start_index = df_baseline.columns.get_loc(start_column)
        compare_columns = df_baseline.columns[start_index:].tolist()
    except KeyError:
        print(
            f"Error: Start compare column '{start_column}' not found in the baseline file."
        )
        sys.exit(1)

    # 6. Calculate differences and generate results
    result_df = df_merged[key_columns].copy()
    fluctuation_alerts = []

    for col in compare_columns:
        col_baseline = f"{col}_baseline"
        col_target = f"{col}_target"

        if col_baseline in df_merged and col_target in df_merged:
            # Add a small epsilon to avoid division by zero
            denominator = df_merged[col_baseline] + 1e-9
            improvement = (
                (df_merged[col_target] - df_merged[col_baseline]) / denominator
            ) * 100

            result_df[f"{col}_baseline"] = df_merged[col_baseline]
            result_df[f"{col}_target"] = df_merged[col_target]
            # The calculation is unified as (target-baseline)/baseline for user interpretation.
            result_df[f"{col}_improvement(%)"] = improvement

            if col == "tflops_per_sec":
                # Filter rows where the fluctuation exceeds 1.5%
                fluctuating_rows = df_merged[abs(improvement) > 1.5]
                if not fluctuating_rows.empty:
                    # Iterate over these rows to prepare the output message
                    for index, row in fluctuating_rows.iterrows():
                        case_identifier = {k: row[k] for k in key_columns}
                        alert = (
                            f"  - Case: {case_identifier}\n"
                            f"    tflops_per_sec Baseline: {row[col_baseline]:.4f}\n"
                            f"    tflops_per_sec Target: {row[col_target]:.4f}\n"
                            f"    Fluctuation: {improvement.loc[index]:+.2f}%"  # Use '+' to show the sign
                        )
                        fluctuation_alerts.append(alert)

    # 7. Save results to a file
    try:
        result_df.to_csv(output_path, index=False, float_format="%.4f")
        print("\n--- Performance Comparison Report ---")
        print(f"Baseline File: {baseline_path}")
        print(f"Target File:   {target_path}")
        print(f"Results successfully saved to: {output_path}")
    except IOError as e:
        print(f"Error: Could not write to file {output_path}. Reason: {e}")
        sys.exit(1)

    if fluctuation_alerts:
        print("\n" + "=" * 60)
        print(
            "⚠️  Detected TFLOPs performance fluctuation over 1.5% in the following cases:"
        )
        print("=" * 60)
        for alert in fluctuation_alerts:
            print(alert)
        print("=" * 60)
    else:
        print("\n✅ TFLOPs performance fluctuation is within 1.5% for all test cases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compares performance test CSV files by automatically detecting the mode ('dense' or 'block_sparse').",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("baseline", help="Path to the baseline results CSV file")
    parser.add_argument("target", help="Path to the target results CSV file")
    parser.add_argument(
        "output", help="Path for the output CSV file to save comparison results"
    )

    args = parser.parse_args()

    # --- Auto-detection Logic ---
    try:
        # Read only the header of the baseline file to determine the mode
        df_header = pd.read_csv(args.baseline, nrows=0)
        mode = detect_mode(df_header.columns)

        if mode is None:
            print(
                "Error: Could not automatically determine the comparison mode from the CSV headers."
            )
            print(f"File columns found: {list(df_header.columns)}")
            print(
                "Please ensure the CSV contains the key columns for either 'dense' or 'block_sparse' mode."
            )
            sys.exit(1)

        print(f"--- Automatically detected mode: '{mode}' ---")
        # Call the original function with the detected mode
        run_comparison(mode, args.baseline, args.target, args.output)

    except FileNotFoundError:
        print(f"Error: The baseline file was not found at '{args.baseline}'")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during mode detection: {e}")
        sys.exit(1)
