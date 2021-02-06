"""Configuration for pytest."""
import pytest


_DEFAULT_SKIPSLOW = True

def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption(
        "--skip-slow",
        action="store_true",
        help="skips slow tests",
        default=_DEFAULT_SKIPSLOW,
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,  # Only used for cli override
        help="run slow tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow") and config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
