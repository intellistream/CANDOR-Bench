"""Shared bootstrap for the driver test scripts: puts the build output on
sys.path and imports the module. The single place that knows where the
built extension lives."""

import os
import sys


def import_native():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "src", "index", "concurrent", "build", "lib"))
    import concurrency_native

    return concurrency_native
