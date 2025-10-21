#!/bin/bash

# Copyright (c) 2025 SandAI. All Rights Reserved.
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

# Define the path to the clang-format executable.
# You might need to adjust the version number (e.g., clang-format-15, clang-format-17)
# or just "clang-format" if it's a generic link.

LLVM_VERSION=${LLVM_VERSION:-21}

CLANG_FORMAT_BIN="clang-format-${LLVM_VERSION}"

INSTALL_CMD="bash scripts/install_clang_format.sh"

# Check if the clang-format executable is available in the system's PATH.
# 'command -v' attempts to locate the command.
# '&> /dev/null' redirects both stdout and stderr to null, suppressing output during the check.
if ! command -v "$CLANG_FORMAT_BIN" &> /dev/null
then
    echo "Error: '$CLANG_FORMAT_BIN' command not found."
    echo "This pre-commit hook requires '$CLANG_FORMAT_BIN' to be installed."
    echo "We recommend running '$INSTALL_CMD' to install it."
    exit 1 # Exit with a non-zero status code, signaling failure to pre-commit.
fi

# If clang-format is found, execute it with all arguments passed to this script.
# 'exec' replaces the current shell process with the clang-format process,
# meaning clang-format's exit status will become this script's exit status.
# '"$@"' expands to all positional parameters passed to this script, preserving whitespace.
exec "$CLANG_FORMAT_BIN" "$@"
