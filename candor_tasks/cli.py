from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
GAMMAFRESH_DIR = ROOT / "algorithms_impl" / "GammaFresh"


def _run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _gammafresh_python() -> str:
    completed = subprocess.run(
        ["uv", "run", "--project", str(GAMMAFRESH_DIR), "python", "-c", "import sys; print(sys.executable)"],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _gammafresh_build(build_type: str = "Release") -> None:
    _run(["uv", "sync", "--project", str(GAMMAFRESH_DIR)])
    python_exe = _gammafresh_python()
    build_dir = GAMMAFRESH_DIR / "build"
    _run(
        [
            "cmake",
            "-S",
            str(GAMMAFRESH_DIR),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DPython_EXECUTABLE={python_exe}",
        ]
    )
    _run(["cmake", "--build", str(build_dir), "-j"])


def build_gammafresh() -> None:
    parser = argparse.ArgumentParser(description="Build GammaFresh bindings with uv-managed Python")
    parser.add_argument("--build-type", default="Release", choices=["Debug", "Release"])
    args = parser.parse_args()
    _gammafresh_build(args.build_type)


def run_tests() -> None:
    parser = argparse.ArgumentParser(description="Run root and GammaFresh tests")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    _run(["uv", "sync"])
    _run(["uv", "sync", "--project", str(GAMMAFRESH_DIR)])
    if not args.skip_build:
        _gammafresh_build("Release")

    _run(["python", "-m", "pytest"])
    _run(["uv", "run", "--project", str(GAMMAFRESH_DIR), "pytest", "scripts/python_tests"], cwd=GAMMAFRESH_DIR)
    _run(["ctest", "--test-dir", str(GAMMAFRESH_DIR / "build"), "--output-on-failure"])


if __name__ == "__main__":
    command = Path(sys.argv[0]).name
    if command == "candor-build-gammafresh":
        build_gammafresh()
    elif command == "candor-test":
        run_tests()
    else:
        raise SystemExit(f"Unknown entrypoint: {command}")
