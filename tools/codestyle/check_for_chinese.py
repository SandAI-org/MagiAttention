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


import re
import sys

RED = "\033[91m"
RESET = "\033[0m"


def contains_chinese(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return lines


def check_for_chinese(file_path):
    lines = contains_chinese(file_path)
    chinese_lines = []
    for idx, line in enumerate(lines, start=1):
        matches = list(re.finditer(r"[\u4e00-\u9fff]", line))
        if matches:
            start = matches[0].start() + 1  # +1 to make it 1-based
            end = matches[-1].start() + 1  # +1 to make it 1-based
            chinese_lines.append((idx, start, end))
    return chinese_lines


if __name__ == "__main__":
    for file_path in sys.argv[1:]:
        chinese_lines = check_for_chinese(file_path)
        if chinese_lines:
            for line_number, start, end in chinese_lines:
                if start == end:
                    print(
                        f"{file_path}:{line_number}:{start}: {RED}error:{RESET} Contains Chinese characters"
                    )
                else:
                    print(
                        f"{file_path}:{line_number}:{start}-{end}: {RED}error:{RESET} Contains Chinese characters"
                    )
            sys.exit(1)
