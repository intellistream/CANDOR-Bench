"""pytest configuration for SAGE-DB-Bench.

This configuration ensures third-party algorithm tests are skipped.
"""
import pytest
from pathlib import Path


def pytest_configure(config):
    """Register custom markers for test organization."""
    config.addinivalue_line(
        "markers", "third_party: tests from third-party algorithm libraries (skipped by default)"
    )
    config.addinivalue_line(
        "markers", "algorithm_impl: tests for algorithm implementations"
    )


def pytest_ignore_collect(collection_path, path, config):
    """Ignore third-party algorithm test files during collection.
    
    This runs before pytest tries to import the test files, preventing
    ImportError from missing dependencies like diskannpy, faiss.contrib, etc.
    """
    path_str = str(collection_path)

    # Ignore all test files under algorithms_impl/
    if "algorithms_impl" in path_str and collection_path.name.startswith("test_"):
        return True

    # Ignore specific algorithm directories
    ignore_patterns = [
        "/diskann-ms/",
        "/DiskANN/",
        "/faiss/",
        "/ipdiskann/",
        "/vsag/",
        "/SPTAG/",
        "/plsh/",
        "/gti/",
        "/candy/",
        "/puck/",
    ]

    for pattern in ignore_patterns:
        if pattern in path_str and collection_path.suffix == ".py":
            return True

    return False


def pytest_collection_modifyitems(config, items):
    """Mark any remaining third-party tests that slipped through.
    
    This is a backup in case pytest_ignore_collect doesn't catch everything.
    """
    skip_third_party = pytest.mark.skip(
        reason="Third-party algorithm test - requires external dependencies"
    )

    for item in items:
        item_path = str(item.fspath)

        # Skip all tests under algorithms_impl/
        if "algorithms_impl" in item_path:
            item.add_marker(skip_third_party)
            continue

        # Also skip if path contains specific algorithm names
        third_party_patterns = [
            "/DiskANN/",
            "/diskann-ms/",
            "/faiss/",
            "/ipdiskann/",
            "/vsag/",
            "/SPTAG/",
            "/plsh/",
            "/gti/",
        ]

        for pattern in third_party_patterns:
            if pattern in item_path:
                item.add_marker(skip_third_party)
                break
