#!/bin/bash

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

# Exit immediately if a command exits with a non-zero status.
set -e

CLANG_FORMAT_VERSION=${1:-20.1.8}
if [[ ! "$CLANG_FORMAT_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: specify a full clang-format version, e.g. 20.1.8."
    exit 1
fi
LLVM_VERSION=${CLANG_FORMAT_VERSION%%.*}
LLVM_SH_URL="https://apt.llvm.org/llvm.sh"
LLVM_SH_SCRIPT=$(mktemp)
trap 'rm -f "$LLVM_SH_SCRIPT"' EXIT

echo "======================================================"
echo " Starting LLVM ${LLVM_VERSION} and clang-format ${CLANG_FORMAT_VERSION} installation "
echo "======================================================"

# Check for sudo permissions
if [ "$(id -u)" -ne 0 ]; then
    echo "This script requires root privileges. Please run with sudo."
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Install necessary dependencies
echo "Checking and installing essential dependencies (lsb-release, wget, software-properties-common, gnupg)..."
if command_exists apt; then
    apt update
    apt install -y lsb-release wget software-properties-common gnupg
elif command_exists dnf; then # For Fedora/RHEL-like systems (if we wanted to extend support)
    dnf install -y redhat-lsb-core wget software-properties-common gnupg
elif command_exists yum; then # For older RHEL/CentOS
    yum install -y redhat-lsb-core wget software-properties-common gnupg
elif command_exists pacman; then # For Arch-like systems
    pacman -Sy --noconfirm lsb-release wget gnupg
    # Arch doesn't use software-properties-common, pacman handles repos directly
else
    echo "Error: Cannot find a package manager to install essential dependencies. Please install them manually."
    exit 1
fi
echo "Essential dependencies installed/checked."


# 2. Download and run LLVM official installation script
echo "Downloading LLVM official installation script from ${LLVM_SH_URL}..."
wget -q -O "${LLVM_SH_SCRIPT}" "${LLVM_SH_URL}"
chmod +x "${LLVM_SH_SCRIPT}"

echo "Running LLVM installation script for version ${LLVM_VERSION}..."
# The llvm.sh script adds the repo and installs the base clang package
bash "${LLVM_SH_SCRIPT}" "${LLVM_VERSION}"

# Clean up the downloaded script
rm "${LLVM_SH_SCRIPT}"

# 3. Explicitly install clang-format-${LLVM_VERSION}
echo "Installing clang-format-${LLVM_VERSION}..."
# Refresh package list again after llvm.sh might have added new repos or updated existing ones
apt update
# Debian package revisions differ by distribution; select the exact upstream
# release, then pin its matching runtime libraries as well as the executable.
PACKAGE_VERSION=$(apt-cache madison "clang-format-${LLVM_VERSION}" | awk -v expected="$CLANG_FORMAT_VERSION" '
    {
        upstream = $3
        sub(/^[0-9]+:/, "", upstream)
        split(upstream, parts, /[~+-]/)
        if (parts[1] == expected) { print $3; exit }
    }')
if [ -z "$PACKAGE_VERSION" ]; then
    echo "Error: clang-format ${CLANG_FORMAT_VERSION} is unavailable in the configured APT repositories."
    exit 1
fi
apt install -y --allow-downgrades \
    "clang-format-${LLVM_VERSION}=${PACKAGE_VERSION}" \
    "libclang-cpp${LLVM_VERSION}=${PACKAGE_VERSION}" \
    "libllvm${LLVM_VERSION}=${PACKAGE_VERSION}"

# 4. Set up update-alternatives for clang-format
echo "Setting up 'clang-format' command via update-alternatives..."
if command_exists update-alternatives; then
    update-alternatives --install /usr/bin/clang-format clang-format "/usr/bin/clang-format-${LLVM_VERSION}" 100
    update-alternatives --set clang-format "/usr/bin/clang-format-${LLVM_VERSION}"
else
    echo "Warning: update-alternatives command not found. You may need to manually link clang-format."
    echo "  Example: sudo ln -s /usr/bin/clang-format-${LLVM_VERSION} /usr/local/bin/clang-format"
fi

# Fail rather than silently accept a different patch release.
INSTALLED_VERSION=$(clang-format-${LLVM_VERSION} --version | sed -nE 's/.*version ([0-9]+\.[0-9]+\.[0-9]+).*/\1/p')
if [ "$INSTALLED_VERSION" != "$CLANG_FORMAT_VERSION" ]; then
    echo "Error: expected clang-format ${CLANG_FORMAT_VERSION}, found ${INSTALLED_VERSION}."
    exit 1
fi
echo "Verified clang-format ${INSTALLED_VERSION}."
echo "======================================================"
echo " LLVM and clang-format-${LLVM_VERSION} installation completed! "
echo " You can now use 'clang-format --version' or 'clang-format-${LLVM_VERSION} --version'."
echo "======================================================"
