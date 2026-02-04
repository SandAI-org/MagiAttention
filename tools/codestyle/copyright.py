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

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import datetime
import io
import os
import re
import sys

# global current year
NOW = int(os.getenv("MAGI_ATTENTION_COPYRIGHT_TEST_YEAR", datetime.datetime.now().year))

# initial copyright for starting year 2025
COPYRIGHT = """Copyright (c) 2025 SandAI. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

RE_ENCODE = re.compile(r"^[ \t\v]*#.*?coding[:=]", re.IGNORECASE)
RE_COPYRIGHT = re.compile(
    r".*Copyright \(c\) (\d{4})(?:-(\d{4}))? SandAI", re.IGNORECASE
)
RE_SHEBANG = re.compile(r"^[ \t\v]*#[ \t]?\!")


def _is_python_file(path) -> bool:
    lang_type = re.compile(r"\.(py|pyi|sh)$")
    if lang_type.search(path) is not None:
        return True
    return False


def _is_c_file(path) -> bool:
    lang_type = re.compile(r"\.(h|c|hpp|cc|cpp|cu|go|cuh|proto)$")
    if lang_type.search(path) is not None:
        return True
    return False


def _generate_copyright(path, comment_mark) -> list[str]:
    is_c_file = _is_c_file(path)

    if isinstance(comment_mark, tuple):
        start_mark, end_mark = comment_mark
    else:
        start_mark = comment_mark
        end_mark = comment_mark

    copyright = COPYRIGHT.split(os.linesep)
    if NOW == 2025:
        header = "Copyright (c) 2025 SandAI. All Rights Reserved."
    else:
        header = f"Copyright (c) 2025-{NOW} SandAI. All Rights Reserved."

    ans = []

    if is_c_file:
        ans.append(
            start_mark
            + (
                "/**********************************************************************************"
            )
            + os.linesep
        )
    ans.append(start_mark + (" * " if is_c_file else " ") + header + os.linesep)
    for idx, line in enumerate(copyright[1:]):
        if start_mark == "#":
            ans.append(comment_mark + " " + line.rstrip() + os.linesep)
        else:
            ans.append(start_mark + " * " + line.rstrip() + os.linesep)
    if is_c_file:
        ans.append(
            start_mark
            + (
                " *********************************************************************************/"
            )
            + os.linesep
        )
    if end_mark != "#":
        ans.append(end_mark)

    return ans


def _get_comment_mark(path) -> str | tuple[str, str]:
    if _is_python_file(path):
        return "#"
    elif _is_c_file(path):
        return "", ""
    else:
        raise RuntimeError(f"Unsupported file: {path}")


def _check_copyright(path):
    try:
        with open(path) as f:
            for line in f:
                match = RE_COPYRIGHT.search(line)
                if match:
                    year_str = match.group(1)
                    if NOW == 2025:
                        return 1 if year_str == "2025" else 2
                    else:
                        expected = f"2025-{NOW}"
                        return 1 if year_str == expected else 2
            return 0
    except Exception:
        return 0


def generate_copyright(path, comment_mark):
    original_contents = io.open(path, encoding="utf-8").readlines()
    head = original_contents[0:4]

    insert_line_no = 0
    for i, line in enumerate(head):
        if RE_ENCODE.search(line) or RE_SHEBANG.search(line):
            insert_line_no = i + 1

    copyright = _generate_copyright(path, comment_mark)
    if insert_line_no == 0:
        new_contents = copyright
        if len(original_contents) > 0 and len(original_contents[0].strip()) != 0:
            new_contents.append(os.linesep)
        new_contents.extend(original_contents)
    else:
        new_contents = original_contents[0:insert_line_no]
        new_contents.append(os.linesep)
        new_contents.extend(copyright)
        if (
            len(original_contents) > insert_line_no
            and len(original_contents[insert_line_no].strip()) != 0
        ):
            new_contents.append(os.linesep)
        new_contents.extend(original_contents[insert_line_no:])
    new_contents = "".join(new_contents)

    with io.open(path, "w") as output_file:
        output_file.write(new_contents)


def update_copyright_year_in_file(path):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except Exception:
        return

    new_lines = []
    updated = False
    for line in lines:
        if not updated:
            match = RE_COPYRIGHT.search(line)
            if match:
                if NOW == 2025:
                    new_year = "2025"
                else:
                    new_year = f"2025-{NOW}"
                new_line = re.sub(r"\d{4}(-\d{4})?", new_year, line, count=1)
                new_lines.append(new_line)
                updated = True
                continue
        new_lines.append(line)

    if updated:
        with open(path, "w") as f:
            f.writelines(new_lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Checker for copyright declaration.")
    parser.add_argument("filenames", nargs="*", help="Filenames to check")
    args = parser.parse_args(argv)

    for path in args.filenames:
        comment_mark = _get_comment_mark(path)
        if comment_mark is None:
            print("warning:Unsupported file", path, file=sys.stderr)
            continue

        status = _check_copyright(path)
        if status == 0:
            generate_copyright(path, comment_mark)
        elif status == 2:
            update_copyright_year_in_file(path)


if __name__ == "__main__":
    exit(main())
