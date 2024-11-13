"""pl-matching package.

Functionality for matching timeseries data.
"""

from __future__ import annotations

from .matcher import MatchedGroup, MatchedIndividual, MatchingMethod

__all__: list[str] = ["MatchedIndividual", "MatchedGroup", "MatchingMethod"]
