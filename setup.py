import os
import shutil
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import glob

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # Check if CMake is installed
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        # Set environment variables
        os.environ['CUDACXX'] = '/usr/local/cuda/bin/nvcc'
        if sys.platform == 'linux':
            os.environ['LD_LIBRARY_PATH'] = '/path/to/custom/libs:' + os.environ.get('LD_LIBRARY_PATH', '')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        try:
            torch_cmake_prefix = subprocess.check_output(
                [sys.executable, "-c", "import torch; print(torch.utils.cmake_prefix_path)"],
                text=True,
                encoding='utf-8'
            ).strip()
        except subprocess.CalledProcessError as e:
            print(f"Error getting torch cmake path: {e}")
            sys.exit(1)

        try:
            threads_output = subprocess.check_output(["nproc"], text=True, encoding='utf-8').strip()
            threads = threads_output if threads_output.isdigit() else "2" # <--- threads 在这里被定义和赋值
        except subprocess.CalledProcessError:
            threads = "2" # Fallback if nproc fails

        print(f"Using {threads} build threads.") 
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_PREFIX_PATH={torch_cmake_prefix}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
            f'-DVERSION_INFO={self.distribution.get_version()}',

            '-DENABLE_HDF5=OFF',
            '-DENABLE_PYBIND=ON',
            '-DCMAKE_INSTALL_PREFIX=/usr/local/lib',
            '-DENABLE_PAPI=OFF',
            '-DENABLE_SPTAG=ON',
            '-DENABLE_SPTAG=OFF',
            '-DENABLE_DiskANN=OFF',
            '-DPYBIND=ON',
        ]

        mkl_base_path = os.environ.get('MKLROOT', '/opt/conda')

        cmake_args.append(f'-DMKL_PATH={mkl_base_path}')
        cmake_args.append(f'-DMKL_INCLUDE_PATH={mkl_base_path}/include')

        print(f"DEBUG: Setting -DMKL_PATH={mkl_base_path}")
        print(f"DEBUG: Setting -DMKL_INCLUDE_PATH={mkl_base_path}/include")

        build_args = ['--config', cfg]
        build_args += ['--', '-j' + threads, 'VERBOSE=1']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            print(f"DEBUG: Created build directory: {self.build_temp}")


        print(f"DEBUG: CMake configure command: {['cmake', ext.sourcedir] + cmake_args}")
        subprocess.run(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, check=True)

        print(f"DEBUG: CMake build command: {['cmake', '--build', '.'] + build_args}")
        subprocess.run(['cmake', '--build', '.'] + build_args, cwd=self.build_temp, check=True)

        so_files = glob.glob(os.path.join(self.build_temp, '*.so'))
        print("Discovered .so files:")
        print(so_files)
        for file in so_files:
            target_path = os.path.join(extdir, os.path.basename(file))
            print(f"Copying {file} to {target_path}")
            shutil.copy(file, extdir)
setup(
    name='PyCANDYAlgo',
    version='0.1',
    author='Your Name',
    description='A simple python version of CANDY benchmark built with Pybind11 and CMake',
    long_description='',
    ext_modules=[CMakeExtension('.')],
    cmdclass={
        'build_ext': CMakeBuild,
    },
    zip_safe=False,
)