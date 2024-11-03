"""pl-matching package.

Functionality for matching timeseries data.
"""

from __future__ import annotations

from .matcher import Matcher, MatchingMethod, TimesArray, Timeseries

__all__: list[str] = ["Matcher", "Timeseries", "MatchingMethod", "TimesArray"]
