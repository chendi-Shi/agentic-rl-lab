"""Stability and difficulty diagnostics for GRPO groups."""

from __future__ import annotations

from arlab.diagnostics.stats import (
    CollapseReport,
    DifficultyReport,
    GroupStats,
    classify_difficulty,
    detect_advantage_collapse,
    group_stats,
    length_percentiles,
)

__all__ = [
    "CollapseReport",
    "DifficultyReport",
    "GroupStats",
    "classify_difficulty",
    "detect_advantage_collapse",
    "group_stats",
    "length_percentiles",
]
