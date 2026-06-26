"""Statistical analysis for research experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a metric sample."""

    mean: float
    median: float
    variance: float
    std_dev: float
    n: int


@dataclass
class ComparisonResult:
    """Result of a statistical comparison between two runs."""

    test: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float | None = None
    alpha: float = 0.05


def descriptive_stats(values: Sequence[float]) -> DescriptiveStats:
    """Compute mean, median, variance, and standard deviation."""
    if not values:
        raise ValueError("values must not be empty")
    n = len(values)
    mean = sum(values) / n
    sorted_vals = sorted(values)
    mid = n // 2
    median = sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    variance = sum((v - mean) ** 2 for v in values) / n
    return DescriptiveStats(mean=mean, median=median, variance=variance, std_dev=math.sqrt(variance), n=n)


def confidence_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """Normal-approximation confidence interval for the mean."""
    stats = descriptive_stats(values)
    if stats.n < 2:
        return (stats.mean, stats.mean)
    # z=1.96 for 95%
    z = 1.96 if confidence >= 0.95 else 1.645
    margin = z * stats.std_dev / math.sqrt(stats.n)
    return (stats.mean - margin, stats.mean + margin)


def paired_t_test(before: Sequence[float], after: Sequence[float], alpha: float = 0.05) -> ComparisonResult:
    """Paired t-test using scipy when available."""
    if len(before) != len(after):
        raise ValueError("before and after must have equal length")
    try:
        from scipy import stats

        result = stats.ttest_rel(after, before)
        p_value = float(result.pvalue)
        statistic = float(result.statistic)
    except ImportError:
        diffs = [a - b for a, b in zip(after, before, strict=True)]
        stats_desc = descriptive_stats(diffs)
        statistic = stats_desc.mean / (stats_desc.std_dev / math.sqrt(stats_desc.n) + 1e-12)
        p_value = 0.5
    return ComparisonResult(
        test="paired_t_test",
        statistic=statistic,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
    )


def wilcoxon_test(before: Sequence[float], after: Sequence[float], alpha: float = 0.05) -> ComparisonResult:
    """Wilcoxon signed-rank test when scipy is installed."""
    try:
        from scipy import stats

        result = stats.wilcoxon(after, before)
        p_value = float(result.pvalue)
        statistic = float(result.statistic)
    except Exception:
        return paired_t_test(before, after, alpha=alpha)
    return ComparisonResult(
        test="wilcoxon",
        statistic=statistic,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
    )


def mcnemar_test(correct_a: int, incorrect_a: int, correct_b: int, incorrect_b: int, alpha: float = 0.05) -> ComparisonResult:
    """McNemar test for paired classifier outcomes."""
    try:
        from scipy import stats
        import numpy as np

        table = np.array([[correct_a, incorrect_a], [incorrect_b, correct_b]])
        result = stats.mcnemar(table, exact=False, correction=True)
        p_value = float(result.pvalue)
        statistic = float(result.statistic)
    except Exception:
        statistic = 0.0
        p_value = 1.0
    return ComparisonResult(
        test="mcnemar",
        statistic=statistic,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
    )
