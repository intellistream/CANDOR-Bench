"""
Setup script for PyCANDYAlgo package - 仅用于安装预编译的 .so 文件
"""
from setuptools import setup
from setuptools.dist import Distribution
import os
import glob
import shutil
import sys

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(self):
        return True

# 查找所有编译好的 .so 文件
so_files = glob.glob('PyCANDYAlgo/PyCANDYAlgo*.so')

if not so_files:
    print("Error: No PyCANDYAlgo*.so file found. Please run ./build.sh first.", file=sys.stderr)
    sys.exit(1)

# 使用最简单的 setup，不定义任何 ext_modules
# 这样 setuptools 不会尝试编译任何东西
setup(
    name='PyCANDYAlgo',
    version='0.1.2',
    description='CANDY Algorithm implementations with Python bindings',
    author='IntelliStream',
    packages=['PyCANDYAlgo'],
    package_dir={'PyCANDYAlgo': 'PyCANDYAlgo'},
    package_data={'PyCANDYAlgo': ['*.so']},
    distclass=BinaryDistribution,
    zip_safe=False,
    python_requires='>=3.8',
)
