from __future__ import annotations

import argparse
import os
import shutil
import site
import subprocess
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ALGORITHMS_IMPL = ROOT / "algorithms_impl"
GAMMAFRESH_DIR = ALGORITHMS_IMPL / "GammaFresh"


def info(message: str) -> None:
    print(f"-> {message}", flush=True)


def ok(message: str) -> None:
    print(f"OK {message}", flush=True)


def warn(message: str) -> None:
    print(f"WARN {message}", flush=True)


def run(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, check=check)


def prepend_env_path(env: dict[str, str], key: str, value: Path) -> None:
    if not value.exists():
        return
    current = env.get(key, "")
    parts = [part for part in current.split(os.pathsep) if part]
    value_str = str(value)
    if value_str not in parts:
        env[key] = os.pathsep.join([value_str, *parts])


def prepend_preload(env: dict[str, str], value: Path) -> None:
    if not value.exists():
        return
    current = env.get("LD_PRELOAD", "")
    parts = [part for part in current.split(os.pathsep) if part]
    value_str = str(value)
    if value_str not in parts:
        env["LD_PRELOAD"] = os.pathsep.join([value_str, *parts])


def native_env() -> dict[str, str]:
    env = os.environ.copy()
    prepend_env_path(env, "PATH", Path(sys.executable).parent)

    for mklroot in (Path("/opt/intel/oneapi/mkl/latest"), Path("/opt/intel/mkl")):
        if mklroot.exists():
            env["MKLROOT"] = str(mklroot)
            prepend_env_path(env, "LD_LIBRARY_PATH", mklroot / "lib" / "intel64")
            prepend_env_path(env, "LIBRARY_PATH", mklroot / "lib" / "intel64")
            prepend_env_path(env, "CPATH", mklroot / "include")
            prepend_env_path(env, "CMAKE_PREFIX_PATH", mklroot)
            break

    for compiler_lib in (Path("/opt/intel/oneapi/compiler/2025.3/lib"), Path("/opt/intel/oneapi/compiler/latest/lib")):
        if compiler_lib.exists():
            prepend_env_path(env, "LD_LIBRARY_PATH", compiler_lib)
            prepend_preload(env, compiler_lib / "libiomp5.so")
            break

    prepend_preload(env, Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6"))

    try:
        import torch

        prepend_env_path(env, "LD_LIBRARY_PATH", Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass

    return env


def check_python() -> None:
    version = sys.version_info
    info(f"Python: {sys.executable} ({version.major}.{version.minor}.{version.micro})")
    if (version.major, version.minor) != (3, 10):
        raise SystemExit(
            "This native benchmark stack is pinned to Python 3.10. "
            "Run `uv sync --python 3.10` or keep the repository `.python-version` file."
        )


def check_system_deps() -> None:
    missing = [tool for tool in ("cmake", "make", "git", "pkg-config") if shutil.which(tool) is None]
    if shutil.which("ninja") is None:
        warn("ninja not found; GammaFresh will use CMake's default generator")
    if missing:
        raise SystemExit(f"Missing system build tools: {', '.join(missing)}")
    ok("system build tools are available")


def init_submodules(env: dict[str, str]) -> None:
    if (ROOT / ".gitmodules").exists():
        run(["git", "submodule", "update", "--init", "--recursive"], env=env)
        ok("git submodules initialized")
    else:
        warn(".gitmodules not found; skipping submodule initialization")


def jobs_arg(value: int | None) -> str:
    if value:
        return str(value)
    count = os.cpu_count() or 4
    return str(max(1, min(count, 8)))


def extension_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def find_matching_so(root: Path, pattern: str) -> Path | None:
    suffix = extension_suffix()
    matches = sorted(root.glob(pattern))
    for path in matches:
        if path.name.endswith(suffix):
            return path
    return matches[0] if matches else None


def site_packages() -> Path:
    return Path(site.getsitepackages()[0])


def copy_to_site_packages(path: Path) -> None:
    target = site_packages() / path.name
    shutil.copy2(path, target)
    ok(f"installed {path.name} -> {target}")


def copy_to_project_root(path: Path) -> None:
    target = ROOT / path.name
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
        ok(f"installed {path.name} -> {target}")


def module_import_ok(module: str, env: dict[str, str]) -> bool:
    result = subprocess.run([sys.executable, "-c", f"import {module}"], env=env, text=True)
    return result.returncode == 0


def build_pycandy(env: dict[str, str], *, clean: bool, jobs: str) -> None:
    existing = find_matching_so(ALGORITHMS_IMPL, "PyCANDYAlgo*.so")
    if existing and not clean:
        ok(f"reusing {existing.relative_to(ROOT)}")
        copy_to_site_packages(existing)
        copy_to_project_root(existing)
        if module_import_ok("PyCANDYAlgo", env):
            return
        warn("existing PyCANDYAlgo does not import in this uv environment; rebuilding")
        clean = True
    else:
        if existing:
            ok(f"found {existing.relative_to(ROOT)}")

    cmd = ["bash", "build.sh", "--jobs", jobs]
    if clean:
        cmd.insert(2, "--clean")
    run(cmd, cwd=ALGORITHMS_IMPL, env=env)
    existing = find_matching_so(ALGORITHMS_IMPL, "PyCANDYAlgo*.so")
    if not existing:
        raise SystemExit("PyCANDYAlgo build completed but no matching .so was found")
    copy_to_site_packages(existing)
    copy_to_project_root(existing)


def build_gammafresh(env: dict[str, str], *, clean: bool, jobs: str, build_type: str) -> None:
    if not GAMMAFRESH_DIR.exists():
        warn(f"GammaFresh directory not found: {GAMMAFRESH_DIR}")
        return

    build_dir = GAMMAFRESH_DIR / "build"
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)

    existing = find_matching_so(build_dir / "src" / "bindings" / "python", "_candy*.so")
    if existing and not clean:
        ok(f"reusing {existing.relative_to(ROOT)}")
        return

    generator_args = ["-G", "Ninja"] if shutil.which("ninja") else []
    run(
        [
            "cmake",
            "-S",
            str(GAMMAFRESH_DIR),
            "-B",
            str(build_dir),
            *generator_args,
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        env=env,
    )
    run(["cmake", "--build", str(build_dir), "-j", jobs], env=env)
    existing = find_matching_so(build_dir / "src" / "bindings" / "python", "_candy*.so")
    if not existing:
        raise SystemExit("GammaFresh build completed but _candy*.so was not found")
    ok(f"built {existing.relative_to(ROOT)}")


def build_vsag(env: dict[str, str], *, clean: bool, jobs: str) -> None:
    vsag_dir = ALGORITHMS_IMPL / "vsag"
    if not vsag_dir.exists():
        warn(f"VSAG directory not found: {vsag_dir}")
        return

    build_dir = vsag_dir / "build-release"
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)

    if not (build_dir / "CMakeCache.txt").exists():
        args = [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_PYBINDS=ON",
            "-DENABLE_TESTS=OFF",
            "-DENABLE_EXAMPLES=OFF",
            "-DENABLE_TOOLS=OFF",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-B",
            str(build_dir),
            "-S",
            str(vsag_dir),
        ]
        if env.get("MKLROOT"):
            args.extend([f"-DMKLROOT={env['MKLROOT']}", f"-DCMAKE_PREFIX_PATH={env['MKLROOT']}"])
        run(args, env=env)

    run(["cmake", "--build", str(build_dir), "--parallel", jobs], env=env)
    so_file = find_matching_so(build_dir, "_pyvsag*.so")
    if not so_file:
        warn("_pyvsag*.so not found after VSAG build")
        return

    package_dir = vsag_dir / "python" / "pyvsag"
    package_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(so_file, package_dir / so_file.name)
    version_file = package_dir / "_version.py"
    if not version_file.exists():
        version_file.write_text('__version__ = "0.0.1+dev"\n__version_tuple__ = (0, 0, 1, "dev")\n')
    run(["uv", "pip", "install", "--python", sys.executable, "-e", str(vsag_dir / "python"), "--force-reinstall", "--no-build-isolation"], env=env)
    ok("installed pyvsag")


def pybind11_cmake_arg(env: dict[str, str]) -> str | None:
    code = "import pybind11; print(pybind11.get_cmake_dir())"
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True)
    if result.returncode != 0:
        warn("pybind11 cmake dir not available")
        return None
    return f"-Dpybind11_DIR={result.stdout.strip()}"


def patch_gti_n2_header(header: Path) -> None:
    if not header.exists():
        return
    text = header.read_text()
    include = '#include "spdlog/sinks/stdout_color_sinks.h"'
    if include in text:
        return
    text = text.replace('#include "spdlog/spdlog.h"', '#include "spdlog/spdlog.h"\n' + include)
    header.write_text(text)


def build_gti(env: dict[str, str], *, clean: bool, jobs: str, pybind_arg: str | None) -> None:
    gti_dir = ALGORITHMS_IMPL / "gti" / "GTI"
    if not gti_dir.exists():
        warn(f"GTI directory not found: {gti_dir}")
        return

    n2_dir = gti_dir / "extern_libraries" / "n2"
    if n2_dir.exists():
        patch_gti_n2_header(n2_dir / "include" / "n2" / "hnsw_build.h")
        run(["make", "clean"], cwd=n2_dir, env=env, check=False)
        n2_env = env.copy()
        n2_env["CXXFLAGS"] = "-D_GLIBCXX_USE_CXX11_ABI=0"
        run(["make", "shared_lib", f"-j{jobs}"], cwd=n2_dir, env=n2_env, check=False)

    build_dir = gti_dir / "build"
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)
    if clean and (gti_dir / "bin").exists():
        shutil.rmtree(gti_dir / "bin")
    build_dir.mkdir(parents=True, exist_ok=True)
    (gti_dir / "bin").mkdir(exist_ok=True)

    cmake_cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release", f"-DPYTHON_EXECUTABLE={sys.executable}"]
    if pybind_arg:
        cmake_cmd.append(pybind_arg)
    run(cmake_cmd, cwd=build_dir, env=env, check=False)
    run(["make", "gti_wrapper", f"-j{jobs}"], cwd=build_dir, env=env, check=False)
    so_file = find_matching_so(build_dir, "gti_wrapper*.so")
    if so_file:
        copy_to_site_packages(so_file)
    else:
        warn("gti_wrapper*.so not found")


def build_cmake_binding(
    name: str,
    directory: Path,
    pattern: str,
    env: dict[str, str],
    *,
    clean: bool,
    jobs: str,
    pybind_arg: str | None,
    extra_cmake_args: list[str] | None = None,
) -> None:
    if not directory.exists():
        warn(f"{name} directory not found: {directory}")
        return

    build_dir = directory / "build"
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release", f"-DPYTHON_EXECUTABLE={sys.executable}"]
    if extra_cmake_args:
        cmake_cmd.extend(extra_cmake_args)
    if pybind_arg:
        cmake_cmd.append(pybind_arg)
    run(cmake_cmd, cwd=build_dir, env=env)
    run(["make", f"-j{jobs}"], cwd=build_dir, env=env)

    so_file = find_matching_so(build_dir, pattern)
    if so_file:
        copy_to_site_packages(so_file)
    else:
        warn(f"{pattern} not found for {name}")


def build_diskann_tools(env: dict[str, str], *, jobs: str) -> None:
    diskann_root = ROOT / "DiskANN"
    if not diskann_root.exists():
        warn(f"DiskANN directory not found: {diskann_root}")
        return
    build_dir = diskann_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    if not (build_dir / "CMakeCache.txt").exists():
        run(["cmake", ".."], cwd=build_dir, env=env, check=False)
    run(["make", f"-j{jobs}"], cwd=build_dir, env=env, check=False)
    gt_bin = build_dir / "apps" / "utils" / "compute_groundtruth"
    if gt_bin.exists():
        ok(f"DiskANN compute_groundtruth available: {gt_bin}")
    else:
        warn(f"DiskANN compute_groundtruth not found: {gt_bin}")


def verify_import(module: str, *, required: bool, env: dict[str, str]) -> None:
    result = subprocess.run([sys.executable, "-c", f"import {module}; print('{module} OK')"], env=env, text=True)
    if result.returncode == 0:
        ok(f"{module} import")
    elif required:
        raise SystemExit(f"required module import failed: {module}")
    else:
        warn(f"optional module import failed: {module}")


def verify(env: dict[str, str]) -> None:
    verify_import("numpy", required=True, env=env)
    verify_import("PyCANDYAlgo", required=True, env=env)
    verify_import("bench.algorithms.gammafresh.gammafresh", required=True, env=env)
    verify_import("bench.algorithms.faiss_HNSW.faiss_HNSW", required=True, env=env)
    verify_import("bench.algorithms.faiss_IVFPQ.faiss_IVFPQ", required=True, env=env)
    for module in ("gti_wrapper", "ipdiskann", "plsh_python", "pyvsag"):
        verify_import(module, required=False, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy CANDOR-Bench native modules into the uv environment")
    parser.add_argument("--skip-system-deps", action="store_true", help="Skip system build tool checks")
    parser.add_argument("--skip-submodules", action="store_true", help="Skip git submodule initialization")
    parser.add_argument("--skip-build", action="store_true", help="Only verify the uv environment and imports")
    parser.add_argument("--skip-optional", action="store_true", help="Skip optional GTI/IP-DiskANN/PLSH/VSAG builds")
    parser.add_argument("--clean", action="store_true", help="Remove native build directories before rebuilding")
    parser.add_argument("--jobs", type=int, default=None, help="Parallel build jobs")
    parser.add_argument("--build-type", default="Release", choices=["Debug", "Release"], help="CMake build type for GammaFresh")
    args = parser.parse_args()

    check_python()
    if not args.skip_system_deps:
        check_system_deps()

    env = native_env()
    jobs = jobs_arg(args.jobs)
    info(f"site-packages: {site_packages()}")
    info(f"parallel jobs: {jobs}")

    if not args.skip_submodules:
        init_submodules(env)

    if not args.skip_build:
        build_gammafresh(env, clean=args.clean, jobs=jobs, build_type=args.build_type)
        build_pycandy(env, clean=args.clean, jobs=jobs)
        if not args.skip_optional:
            pybind_arg = pybind11_cmake_arg(env)
            build_vsag(env, clean=args.clean, jobs=jobs)
            build_gti(env, clean=args.clean, jobs=jobs, pybind_arg=pybind_arg)
            build_cmake_binding(
                "IP-DiskANN",
                ALGORITHMS_IMPL / "ipdiskann",
                "ipdiskann*.so",
                env,
                clean=args.clean,
                jobs=jobs,
                pybind_arg=pybind_arg,
                extra_cmake_args=["-DPYBIND=ON"],
            )
            build_cmake_binding(
                "PLSH",
                ALGORITHMS_IMPL / "plsh",
                "plsh_python*.so",
                env,
                clean=args.clean,
                jobs=jobs,
                pybind_arg=pybind_arg,
            )
            build_diskann_tools(env, jobs=jobs)

    verify(env)
    ok("uv deployment completed")


if __name__ == "__main__":
    main()
