# Copyright 2026 The RPent Authors.
"""Test configuration for the signed vLLM integration."""

from __future__ import annotations

import sys
from pathlib import Path


CURRENT_RPENT_ROOT = Path(__file__).resolve().parents[6] / "third_party" / "frameworks" / "rpent"
if str(CURRENT_RPENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_RPENT_ROOT))
