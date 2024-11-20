"""pl-matching package.

Functionality for matching timeseries data.
"""

from __future__ import annotations

from .numpy_timeseries import NumpyTimeseries
from .sample import MatchingMethod, sample
from .sampled_data import SampledData, SampledDataGroups

__all__: list[str] = [
    "sample",
    "SampledData",
    "SampledDataGroups",
    "MatchingMethod",
    "NumpyTimeseries",
]
