from __future__ import annotations

import io

import xbatcher


def test_show_versions() -> None:
    """
    Test xbatcher.show_versions()

    Based on https://github.com/pydata/xarray/blob/main/xarray/tests/test_print_versions.py
    """
    f = io.StringIO()
    xbatcher.show_versions(file=f)
    assert 'xbatcher information' in f.getvalue()
