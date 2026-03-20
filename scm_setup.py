import subprocess
import sys
from setuptools import setup
from glob import glob


# 这里我们不做真正的安装逻辑，只是用setup.py作为调用wheel安装的一种方式
def install_wheel(wheel_path):
    """Install the provided wheel using pip."""
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', wheel_path, '--force-reinstall', '--no-deps'])


# 自动寻找当前目录下的wheel文件
wheel_files = glob('*.whl')

if wheel_files:
    install_wheel(wheel_files[0])
else:
    raise FileNotFoundError("No wheel file found in package.")

# TODO: 有什么好办法可以不装一个空的包吗？
setup(
    name='magi-attention-scm-installer',
    version='1.0.5',
    packages=[],
    install_requires=[],
    scripts=[],
    description="A fake package just using setup.py for installing wheels",
)