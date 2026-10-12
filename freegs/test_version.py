"""
Test that the package imports when its distribution metadata is missing,
as for a copy on PYTHONPATH without an installed dist-info
"""

import importlib
from importlib import metadata

import freegs


def test_version_without_metadata(monkeypatch):
    def missing_metadata(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "metadata", missing_metadata)
    try:
        reloaded = importlib.reload(freegs)
        assert reloaded.__version__ == "unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(freegs)
