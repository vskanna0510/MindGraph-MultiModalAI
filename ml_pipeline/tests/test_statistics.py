"""Tests for statistical analysis utilities."""

import pytest

from ml_pipeline.evaluation.statistics import confidence_interval, descriptive_stats, paired_t_test


def test_descriptive_stats() -> None:
    stats = descriptive_stats([1.0, 2.0, 3.0, 4.0])
    assert stats.mean == 2.5
    assert stats.median == 2.5
    assert stats.n == 4


def test_confidence_interval() -> None:
    low, high = confidence_interval([1.0, 2.0, 3.0, 4.0])
    assert low < 2.5 < high


def test_paired_t_test() -> None:
    before = [0.8, 0.82, 0.79, 0.81]
    after = [0.85, 0.86, 0.84, 0.87]
    result = paired_t_test(before, after)
    assert result.test == "paired_t_test"
    assert isinstance(result.p_value, float)
