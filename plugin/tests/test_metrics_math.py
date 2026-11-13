# SPDX-License-Identifier: Apache-2.0
"""CPU-only examples for the small hand-typing exercise."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1] / "gdn_recoverssm_ascend" / "metrics_math.py"
)
SPEC = importlib.util.spec_from_file_location("gdn_metrics_math_exercise", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "actual,captured,want",
    [(0, 0, 0.0), (100, 108, 8 / 108), (108, 108, 0.0), (0, 5, 1.0)],
)
def test_padding_fraction(actual, captured, want):
    assert MODULE.padding_fraction(actual, captured) == pytest.approx(want)


@pytest.mark.parametrize("actual,captured", [(-1, 0), (5, 4), (0, -1)])
def test_padding_fraction_rejects_invalid_counts(actual, captured):
    with pytest.raises(ValueError):
        MODULE.padding_fraction(actual, captured)
