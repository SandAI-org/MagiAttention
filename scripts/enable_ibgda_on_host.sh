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

# ==========================================================
# Script Function: Enable NVIDIA Driver StreamMemOps and PeerMapping
# (IBGDA/GPU Direct Access Configuration)
# ==========================================================

echo "=========================================================="
echo "🚨 ATTENTION: HOST SYSTEM REQUIREMENT 🚨"
echo "This script modifies the Linux kernel module configuration for the NVIDIA driver."
echo "IT MUST BE RUN ON THE BARE-METAL HOST OPERATING SYSTEM, NOT inside a Docker or other containerized environment, as containers do not manage the host kernel."
echo "=========================================================="

# Check for root privileges
if [ "$EUID" -ne 0 ]; then
  echo "ERROR: Please run this script with sudo: sudo bash $0"
  exit 1
fi

CONFIG_FILE="/etc/modprobe.d/nvidia.conf"
CONFIG_LINE='options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"'

echo "=========================================================="
echo "STEP 1: Writing NVIDIA driver configuration to $CONFIG_FILE"
echo "=========================================================="

# Use tee to non-interactively write the configuration file
echo "$CONFIG_LINE" | tee "$CONFIG_FILE" > /dev/null

if [ $? -eq 0 ]; then
  echo "SUCCESS: Configuration file written successfully."
else
  echo "ERROR: Failed to write configuration file. Please check permissions."
  exit 1
fi

echo "=========================================================="
echo "STEP 2: Updating Initramfs (Attempting update-initramfs/dracut)"
echo "=========================================================="

# Logic to attempt update-initramfs and fallback to dracut (omitted for brevity, assume the logic from previous response is here)
# ... (The full update-initramfs/dracut logic is the same as the previous response) ...
update_initramfs_succeeded=false

if command -v update-initramfs &> /dev/null; then
  echo "Attempting to use 'update-initramfs -u'..."
  update-initramfs -u
  if [ $? -eq 0 ]; then
    echo "SUCCESS: update-initramfs completed successfully."
    update_initramfs_succeeded=true
  else
    echo "WARNING: update-initramfs failed. Falling back to 'dracut'."
  fi
else
  echo "INFO: 'update-initramfs' command not found. Falling back to 'dracut'."
fi

if [ "$update_initramfs_succeeded" = false ]; then
  if command -v dracut &> /dev/null; then
    echo "Attempting to use 'dracut -f'..."
    dracut -f
    if [ $? -eq 0 ]; then
      echo "SUCCESS: dracut completed successfully."
    else
      echo "ERROR: Initramfs update failed on both methods."
      exit 1
    fi
  else
    echo "ERROR: Neither 'update-initramfs' nor 'dracut' was found."
    exit 1
  fi
fi

echo "=========================================================="
echo "STEP 3: System Reboot and Verification Hint"
echo "=========================================================="

# HINT must be printed BEFORE the reboot command is issued
echo "The system must be rebooted for changes to take effect."
echo ""
echo "HINT: After the reboot, please run the following commands to verify the configuration:"
echo ""

echo "--- Verification Command 1: Check modprobe configuration ---"
echo "Command: sudo modprobe -c | grep NVreg"
echo "Expected output should include the line:"
echo "'$CONFIG_LINE'"
echo "------------------------------------------------------------"
echo ""

echo "--- Verification Command 2: Check active driver parameters ---"
echo "Command: cat /proc/driver/nvidia/params | grep -E 'EnableStreamMemOPs:|RegistryDwords:'"
echo "Expected output should include lines similar to:"
echo "EnableStreamMemOPs: 1"
echo "RegistryDwords: PeerMappingOverride=1;"
echo "------------------------------------------------------------"
echo ""

# The actual reboot prompt
read -r -p "Do you want to reboot now? (y/N): " response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  echo "Rebooting..."
  reboot
else
  echo "Skipping reboot. Please manually run 'sudo reboot' when convenient."
fi
