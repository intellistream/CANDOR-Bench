"""pytest configuration for SAGE-DB-Bench.

This configuration ensures third-party algorithm tests are skipped.
"""
import pytest


def pytest_configure(config):
    """Register custom markers for test organization."""
    config.addinivalue_line(
        "markers", "third_party: tests from third-party algorithm libraries (skipped by default)"
    )
    config.addinivalue_line(
        "markers", "algorithm_impl: tests for algorithm implementations"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip third-party algorithm tests.
    
    Tests in algorithms_impl/ are third-party library tests that may have
    external dependencies (diskannpy, faiss.contrib, etc.) not available
    in standard SAGE installations.
    """
    skip_third_party = pytest.mark.skip(
        reason="Third-party algorithm test - requires external dependencies"
    )
    
    for item in items:
        item_path = str(item.fspath)
        
        # Skip all tests under algorithms_impl/
        if "algorithms_impl" in item_path:
            item.add_marker(skip_third_party)
        
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
