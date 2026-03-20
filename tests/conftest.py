"""
Pytest fixtures for VecGrid test suite.
"""

import pytest
from vecgrid import InProcessTransport


@pytest.fixture(autouse=True)
def reset_transport():
    """Reset InProcessTransport registry before and after every test
    to prevent cross-test contamination from stale nodes."""
    InProcessTransport.reset()
    yield
    InProcessTransport.reset()


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory path as a string (compatible with
    existing tests that expect str, not pathlib.Path)."""
    return str(tmp_path)
