from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GAMMAFRESH_PYTHON = ROOT / "GammaFresh" / "python"
GAMMAFRESH_BINDINGS = ROOT / "GammaFresh" / "build" / "src" / "bindings" / "python"

for path in (GAMMAFRESH_PYTHON, GAMMAFRESH_BINDINGS):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
