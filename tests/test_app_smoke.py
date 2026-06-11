"""AppTest render smoke — confirms the app renders on Python 3.14 with no crashes.

Must run AFTER data/synthetic/generate_fleet.py (CI does this first).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_app_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).parent.parent / "demo" / "app.py"), default_timeout=90)
    at.run()
    assert not at.exception, f"App raised: {at.exception}"
