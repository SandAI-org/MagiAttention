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

import pandas as pd


def merge_multiple_sources_to_dst(
    src_paths: list[str],
    dst_path: str,
    output_path: str,
    extracted_columns: dict[str, list[str]] | None = None,
    order_columns: list[str] | None = None,
    join_key: str = "seqlen",
):
    """
    Merges unique columns from multiple source CSVs into one destination CSV.

    Args:
        src_paths (list[str]): List of paths to source CSV files.
        dst_path (str): Path to the destination (base) CSV file.
        output_path (str): Path where the merged CSV will be saved.
        extracted_columns (dict[str, list[str]] | None): Dictionary specifying which columns to extract from each source.
        order_columns (list[str] | None): List of strings to specify the preferred column order.
        join_key (str): The common column name used to align rows (default is 'seqlen').
    """
    # 1. Load the base destination CSV
    df_final = pd.read_csv(dst_path)
    print(f"Loaded base destination: {dst_path}")

    # 2. Iterate through each source and merge new columns
    for src_p in src_paths:
        df_src = pd.read_csv(src_p)

        # Extract new columns to add from source
        if extracted_columns is not None and src_p in extracted_columns:
            new_cols = extracted_columns[src_p]
        else:  # If not given, auto-detect new columns
            # Identify columns in current src that are NOT in the current merged result
            new_cols = [col for col in df_src.columns if col not in df_final.columns]

        if new_cols:
            print(f"Adding columns {new_cols} from {src_p}")
            # Subset src to include only join_key and the new unique columns
            src_subset = df_src[[join_key] + new_cols]

            # Left join ensures we keep all rows from the original destination table
            df_final = pd.merge(df_final, src_subset, on=join_key, how="left")
        else:
            print(f"No new columns found in {src_p}, skipping.")

    # 3. Handle column reordering
    if order_columns:
        final_column_sequence = []

        # Force the join_key to be the first column if it's not in the order list
        if join_key not in order_columns:
            final_column_sequence.append(join_key)

        final_column_sequence.extend(order_columns)

        # Filter: Keep only columns that actually exist in the merged dataframe
        existing_ordered_cols = [
            c for c in final_column_sequence if c in df_final.columns
        ]

        # Collect any remaining columns that were not specified in the custom order
        remaining_cols = [c for c in df_final.columns if c not in existing_ordered_cols]

        # Apply the final column order
        df_final = df_final[existing_ordered_cols + remaining_cols]

    # 4. Save the final merged table
    df_final.to_csv(output_path, index=False)
    print(f"\nAll sources merged successfully! Result saved to: {output_path}")


# --- Example Usage ---
if __name__ == "__main__":
    # List of your source files
    source_files = [
        "fa3_ffa.csv",
        "cudnn_fa4.csv",
    ]

    # The base destination file
    dst_path = "sdpa.csv"

    # The output file path
    output_path = "sdpa_fa3_fa4_ffa_cudnn.csv"

    # Preferred order for specific columns
    preferred_order = ["sdpa", "sdpa", "fa3", "fa4", "ffa", "cudnn"]

    merge_multiple_sources_to_dst(
        src_paths=source_files,
        dst_path=dst_path,
        output_path=output_path,
        order_columns=preferred_order,
        join_key="seqlen",
    )
