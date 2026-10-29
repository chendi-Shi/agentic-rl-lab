"""Test fixtures.

pytest's built-in ``tmp_path`` creates its directory in a way that leaves a
DACL granting nothing to the current user on this machine, so every test that
uses it fails with PermissionError. The replacement below only uses
``Path.mkdir``, which inherits the workspace ACLs normally.

Kept deliberately tiny: tests should not depend on pytest internals.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

_WORK_ROOT = Path(".pytest_work")


@pytest.fixture
def tmp_path() -> Path:
    case = _WORK_ROOT / f"case-{uuid.uuid4().hex[:12]}"
    case.mkdir(parents=True, exist_ok=True)
    try:
        yield case
    finally:
        shutil.rmtree(case, ignore_errors=True)
