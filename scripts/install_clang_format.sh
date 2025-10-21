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

# Exit immediately if a command exits with a non-zero status.
set -e

LLVM_SH_URL="https://apt.llvm.org/llvm.sh"
LLVM_SH_SCRIPT="llvm.sh"

# Parse LLVM version from command-line argument
if [ -n "$1" ]; then
    LLVM_VERSION=$1
else
    LLVM_VERSION=20
fi

echo "======================================================"
echo " Starting LLVM and clang-format-${LLVM_VERSION} installation "
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
./"${LLVM_SH_SCRIPT}" "${LLVM_VERSION}"

# Clean up the downloaded script
rm "${LLVM_SH_SCRIPT}"

# 3. Explicitly install clang-format-${LLVM_VERSION}
echo "Installing clang-format-${LLVM_VERSION}..."
# Refresh package list again after llvm.sh might have added new repos or updated existing ones
apt update
apt install -y "clang-format-${LLVM_VERSION}"

# 4. Set up update-alternatives for clang-format
echo "Setting up 'clang-format' command via update-alternatives..."
if command_exists update-alternatives; then
    # Check if clang-format-${LLVM_VERSION} is already registered
    if ! update-alternatives --query clang-format | grep -q "/usr/bin/clang-format-${LLVM_VERSION}"; then
        update-alternatives --install /usr/bin/clang-format clang-format "/usr/bin/clang-format-${LLVM_VERSION}" 100
        echo "clang-format-${LLVM_VERSION} registered as default 'clang-format'."
    else
        echo "clang-format-${LLVM_VERSION} is already registered."
    fi
    # Ensure our version is selected if multiple exist
    # This part is tricky for non-interactive. The previous `install` command usually sets it as default
    # if it's the first or highest priority. If not, a manual `--config` might be needed.
    # For now, we'll just inform the user if it's not the current default.
    if ! update-alternatives --query clang-format | grep -q "link currently points to /usr/bin/clang-format-${LLVM_VERSION}"; then
        echo "======================================================"
        echo "Note: If 'clang-format' does not point to version ${LLVM_VERSION}, you might need to manually select it via:"
        echo "   sudo update-alternatives --config clang-format"
        echo "======================================================"
    fi
else
    echo "Warning: update-alternatives command not found. You may need to manually link clang-format."
    echo "  Example: sudo ln -s /usr/bin/clang-format-${LLVM_VERSION} /usr/local/bin/clang-format"
fi

echo "======================================================"
echo " LLVM and clang-format-${LLVM_VERSION} installation completed! "
echo " You can now use 'clang-format --version' or 'clang-format-${LLVM_VERSION} --version'."
echo "======================================================"

# Final check for clang-format version
echo "Checking installed clang-format version:"
clang-format --version || true # Add || true to prevent script exit if clang-format is not in PATH or fails for some reason.
clang-format-${LLVM_VERSION} --version || true
