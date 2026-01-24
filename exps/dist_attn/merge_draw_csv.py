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

from pathlib import Path
from typing import Dict, List

import pandas as pd
from baselines.interface import AttnImpl

from magi_attention.benchmarking import Mark


def append_files_with_value_prefix(
    files_pack: List[str],
    expected_cols: List[str],
    prefix_list: List[str],
    output_path: str,
    short_for_xlables: Dict[str, str] | None = None,
):
    """
    Append rows from multiple CSV files in order.
    Add prefix to the VALUES of expected_cols[0] for each file.
    Optionally replace values using a mapping dictionary.

    Args:
        files_pack (list[str]): Ordered list of CSV files.
        expected_cols (list[str]): Columns to extract (same for all files).
                                   Prefix is applied to expected_cols[0].
        prefix_list (list[str]): Prefix per file (same length as files_pack).
        output_path (str): Output CSV path.
        short_for_xlables (dict[str,str] | None): Optional mapping for replacing values
                                                  after prefixing.
    """

    if len(files_pack) != len(prefix_list):
        raise ValueError("files_pack and prefix_list must have the same length")

    if not expected_cols:
        raise ValueError("expected_cols must not be empty")

    key_col = expected_cols[0]
    dfs = []

    for file_path, prefix in zip(files_pack, prefix_list):
        file_path = Path(file_path)  # type: ignore[assignment]
        if not file_path.exists():  # type: ignore[attr-defined]
            raise FileNotFoundError(file_path)

        df = pd.read_csv(file_path)

        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise KeyError(f"{file_path} missing columns: {missing}")

        df = df[expected_cols].copy()

        def add_prefix_and_replace(v: str) -> str:
            v_prefixed = f"{prefix}_{v}"
            if short_for_xlables and v in short_for_xlables:
                return f"{prefix}_{short_for_xlables[v]}"
            return v_prefixed

        df[key_col] = df[key_col].astype(str).apply(add_prefix_and_replace)

        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(output_path, index=False)

    print(f"Saved appended CSV to: {output_path}")


if __name__ == "__main__":
    # List of CSV files to be merged as a single file
    src_files_pack = [
        "./outputs_dcp_1_8/fwd/output-8-full-fwd.csv",
        "./outs/output-8-full-fwd.csv",
    ]
    # Columns to extract from CSV files during the merge
    expected_cols = ["baseline", "tflops-mean"]
    # Prefixes to prepend to the values of the first expected column for each file,
    # in order to disinguish between baselines.
    prefix_list = ["old", "test"]
    # Path to save the resulting merged CSV file
    output_path = "./merged.csv"

    short_for_xlables = {
        AttnImpl.ULYSSES.value: "a2a",
        AttnImpl.RING_P2P.value: "p2p",
        AttnImpl.RING_ALLGATHER.value: "ag",
        AttnImpl.USP.value: "usp",
        AttnImpl.LOONGTRAIN.value: "loongt",
        AttnImpl.MAGI_ATTENTION.value: "magi",
        AttnImpl.HYBRID_DCP.value: "dcp",
    }
    append_files_with_value_prefix(
        src_files_pack,
        expected_cols,
        prefix_list,
        output_path,
        short_for_xlables,
    )

    Mark.draw_from_csv(
        csv_path=output_path,
        perf_key="test",
        plot_name="merged-dist-attn-exps",
        line_arg="flops",
        save_path="./",
        ylabel="Throughout (TFLOPs/s)",
        x_int=False,
        x_log=False,
    )
