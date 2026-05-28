import os
import numpy as np
import pybind11
import tempfile
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = "0.1.0"


include_dirs = [
    pybind11.get_include(),
    np.get_include(),
]

bindings_dir = "python"
if bindings_dir in os.path.basename(os.getcwd()):
    source_files = ["./bindings.cpp"]
    include_dirs.extend(["../symqglib"])
else:
    source_files = ["./python/bindings.cpp"]
    include_dirs.extend(["./symqglib"])


libraries = []
extra_objects = []


ext_modules = [
    Extension(
        "symphonyqg",
        source_files,
        include_dirs=include_dirs,
        libraries=libraries,
        language="c++",
        extra_objects=extra_objects,
    ),
]


def has_flag(compiler, flagname):
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except:
            return False
    return True


def cpp_flag(compiler):
    if has_flag(compiler, "-std=c++20"):
        return "-std=c++20"
    elif has_flag(compiler, "-std=c++17"):
        return "-std=c++17"
    else:
        raise RuntimeError("Unsupported compiler -- at least C++17 support is needed!")


class BuildExt(build_ext):
    c_opts = {
        "unix": "-Ofast -Wall -lrt -march=native -fpic -fopenmp -ftree-vectorize -DEIGEN_DONT_PARALLELIZE".split()
    }

    link_opts = {
        "unix": "-fopenmp -pthread -Wall".split(),
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())  # type: ignore
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        else:
            raise RuntimeError("Unsupported platform!")

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get(ct, []))

        build_ext.build_extensions(self)


setup(
    name="symphonyqg",
    version=__version__,
    description="SymphonyQG",
    author="Yutong Gou",
    long_description="""SymphonyQG: towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search""",
    ext_modules=ext_modules,
    install_requires=["numpy==1.26.4", "pybind11>=2.13.1"],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
