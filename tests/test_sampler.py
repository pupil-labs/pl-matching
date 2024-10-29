import numpy as np
import pytest

from pupil_labs.matching.matcher import Matcher, MatchingMethod
from pupil_labs.neon_recording.sensors.numpy_timeseries import NumpyTimeseries


@pytest.mark.parametrize(
    "target_ts, sensor1, sensor2",
    [
        (np.arange(100), np.arange(100), np.arange(100)),
        (np.arange(0, 1000, 10), np.arange(1000), np.arange(2000)),
        (np.arange(0, 1000, 10), np.arange(0, 1000, 2), np.arange(0, 2000, 5)),
    ],
)
def test_nearest_basic(target_ts, sensor1, sensor2):
    sensor1 = NumpyTimeseries(sensor1)
    sensor2 = NumpyTimeseries(sensor2)

    matcher = Matcher(target_ts, [sensor1, sensor2], MatchingMethod.NEAREST)

    for target, (val1, val2) in zip(target_ts, matcher):
        assert val1 == val2 == target


def test_nearest_out_of_range():
    target_ts = np.arange(1000)
    sensor1 = NumpyTimeseries(np.arange(-2000, -1000))
    sensor2 = NumpyTimeseries(np.arange(2000, 3000))

    matcher = Matcher(target_ts, [sensor1, sensor2], MatchingMethod.NEAREST)

    for val1, val2 in matcher:
        assert val1 == -1001
        assert val2 == 2000


def test_nearest_minimizes_distance():
    ts1 = np.random.random_integers(0, 1000, 50)
    ts1.sort()
    sensor1 = NumpyTimeseries(ts1)

    ts2 = np.random.random_integers(0, 1000, 50)
    ts2.sort()
    sensor2 = NumpyTimeseries(ts2)

    matcher = Matcher(ts1, [sensor1, sensor2], MatchingMethod.NEAREST)

    for a, b in matcher:
        min_delta = np.min(np.abs(ts2 - a))
        assert np.abs(a - b) == min_delta
