import subprocess
import sys
from setuptools import setup
from glob import glob


def install_wheel(wheel_path):
    """Install the provided wheel using pip."""
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', wheel_path, '--force-reinstall', '--no-deps'])


# Install all wheel files: magi_attention main wheel + sub-packages
# (flash_attn_cute, ffa_fa3, create_block_mask_cuda, magi_to_hstu_cuda, etc.)
wheel_files = sorted(glob('*.whl'))

if not wheel_files:
    raise FileNotFoundError("No wheel file found in package.")

# Installation order matters:
#   1. flash_attn_cute must be installed BEFORE ffa_fa3, because ffa_fa3
#      installs into flash_attn_cute.ffa_fa3 (a sub-package).
#   2. magi_attention must be installed last.
magi_wheels = [w for w in wheel_files if 'magi_attention' in w]
ffa_fa3_wheels = [w for w in wheel_files if 'ffa_fa3' in w and 'magi_attention' not in w]
other_wheels = [w for w in wheel_files if w not in magi_wheels and w not in ffa_fa3_wheels]

for whl in other_wheels + ffa_fa3_wheels + magi_wheels:
    print(f"Installing {whl}...")
    install_wheel(whl)

# TODO: 有什么好办法可以不装一个空的包吗？
setup(
    name='magi-attention-scm-installer',
    version='1.0.5',
    packages=[],
    install_requires=[],
    scripts=[],
    description="A fake package just using setup.py for installing wheels",
)