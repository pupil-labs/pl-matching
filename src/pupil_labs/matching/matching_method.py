from enum import Enum


class MatchingMethod(Enum):
    NEAREST = "nearest"
    BEFORE = "before"
    AFTER = "after"
    INTERPOLATE = "interpolate"
