"""pl-matching package.

Functionality for matching timeseries data.
"""

from __future__ import annotations

from .matching_method import MatchingMethod
from .numpy_timeseries import NumpyTimeseries
from .sampled_data import SampledData, SampledDataGroups

__all__: list[str] = [
    "SampledData",
    "SampledDataGroups",
    "MatchingMethod",
    "NumpyTimeseries",
]
