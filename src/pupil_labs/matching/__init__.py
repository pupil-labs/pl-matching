"""pl-matching package.

Functionality for matching timeseries data.
"""

from __future__ import annotations

from .matcher import Matcher, Sensor

__all__: list[str] = ["Matcher", "Sensor"]
