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

# usage example:  python tools/ptxas_analyzer.py install.log ${start_line} ${end_line} > ptxas_report.md

import re
import sys
from typing import Any, Dict, List

# --- Configuration Area: Threshold Settings --- #

# These thresholds define what is considered an "anomaly".
# Adjust these values based on your specific performance requirements.
THRESHOLDS = {
    "stack_frame": 1024,  # Bytes
    "spill_stores": 500,  # Bytes (Local memory traffic)
    "spill_loads": 500,  # Bytes (Local memory traffic)
    "registers": 128,  # Count (High usage limits occupancy)
    "barriers": 16,  # Count
    "compile_time": 1000.0,  # Milliseconds (ms)
}

# Display names for the report categories and their associated data keys.
CATEGORIES = [
    ("High Stack Frame Usage", "stack_frame", "bytes"),
    ("High Spill Stores", "spill_stores", "bytes"),
    ("High Spill Loads", "spill_loads", "bytes"),
    ("High Register Usage", "registers", ""),
    ("High Barrier Usage", "barriers", ""),
    ("Slow Compilation Time", "compile_time", "ms"),
]


def parse_ptxas_log(
    file_path: str, start_line: int, end_line: int
) -> List[Dict[str, Any]]:
    """
    Parses the ptxas info from the log file within the specified line range.
    Uses a simple state machine logic to group properties under their respective kernel names.
    """
    kernels = []
    current_kernel = None

    # Pre-compile Regex patterns for performance
    re_entry = re.compile(r"Compiling entry function '(.+?)' for '(.+?)'")
    re_props = re.compile(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
    )
    re_usage = re.compile(r"Used (\d+) registers, used (\d+) barriers")
    re_time = re.compile(r"Compile time = ([\d.]+) ms")

    try:
        with open(file_path, "r") as f:
            # Slicing lines (converts 1-based input to 0-based Python indexing)
            all_lines = f.readlines()
            target_lines = all_lines[start_line - 1 : end_line]

            for line in target_lines:
                line = line.strip()

                # Case: Start of a new kernel compilation block
                entry_match = re_entry.search(line)
                if entry_match:
                    if current_kernel:
                        kernels.append(current_kernel)
                    current_kernel = {
                        "name": entry_match.group(1),
                        "arch": entry_match.group(2),
                        "stack_frame": 0,
                        "spill_stores": 0,
                        "spill_loads": 0,
                        "registers": 0,
                        "barriers": 0,
                        "compile_time": 0.0,
                    }
                    continue

                if not current_kernel:
                    continue

                # Case: Extraction of Stack and Spill properties
                props_match = re_props.search(line)
                if props_match:
                    current_kernel["stack_frame"] = int(props_match.group(1))
                    current_kernel["spill_stores"] = int(props_match.group(2))
                    current_kernel["spill_loads"] = int(props_match.group(3))
                    continue

                # Case: Extraction of Register and Barrier usage
                usage_match = re_usage.search(line)
                if usage_match:
                    current_kernel["registers"] = int(usage_match.group(1))
                    current_kernel["barriers"] = int(usage_match.group(2))
                    continue

                # Case: Extraction of Compilation Time
                time_match = re_time.search(line)
                if time_match:
                    current_kernel["compile_time"] = float(time_match.group(1))
                    continue

            # Append the final kernel block processed
            if current_kernel:
                kernels.append(current_kernel)

    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    return kernels


def generate_report(kernels: List[Dict[str, Any]], start_line: int, end_line: int):
    """
    Generates a Markdown formatted report categorized by anomaly type.
    """
    report = []
    report.append("# PTXAS Compilation Anomaly Report")
    report.append(f"Analyzed lines: {start_line} to {end_line}\n")

    has_any_anomaly = False

    for title, key, unit in CATEGORIES:
        # Filter kernels exceeding the threshold
        anomaly_list = [k for k in kernels if k[key] > THRESHOLDS[key]]

        if anomaly_list:
            has_any_anomaly = True
            report.append(f"## {title} (Threshold: {THRESHOLDS[key]}{unit})")
            report.append("| Value | Kernel Function | Arch |")
            report.append("| :--- | :--- | :--- |")

            # Sort anomalies by value in descending order for prioritized debugging
            anomaly_list.sort(key=lambda x: x[key], reverse=True)

            for k in anomaly_list:
                val = f"{k[key]}{unit}"
                report.append(f"| **{val}** | `{k['name']}` | {k['arch']} |")
            report.append("")

    if not has_any_anomaly:
        report.append(
            "Result: No anomalies detected within the current threshold configurations."
        )

    return "\n".join(report)


if __name__ == "__main__":
    # Validate command line arguments
    if len(sys.argv) < 4:
        print("Usage: python ptxas_analyzer.py <log_file> <start_line> <end_line>")
        sys.exit(1)

    log_file_path = sys.argv[1]
    start_ln = int(sys.argv[2])
    end_ln = int(sys.argv[3])

    # Run parsing engine
    parsed_data = parse_ptxas_log(log_file_path, start_ln, end_ln)

    # Generate and print Markdown output
    final_markdown = generate_report(parsed_data, start_ln, end_ln)
    print(final_markdown)
