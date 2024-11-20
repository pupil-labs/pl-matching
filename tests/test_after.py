import numpy as np
import pytest

from pupil_labs.matching import MatchingMethod, NumpyTimeseries, sample


@pytest.mark.parametrize(
    "target_ts, sensor_ts, sensor_data",
    [
        (np.arange(0, 1000, 10), np.arange(0, 1000, 10), np.arange(0, 1000, 10)),
        (np.arange(0, 1000, 10), np.arange(0, 1000, 10) + 1, np.arange(0, 1000, 10)),
        (np.arange(0, 1000, 10), np.arange(0, 1010, 10) - 1, np.arange(-10, 1000, 10)),
        (np.arange(0, 1000, 10), np.arange(0, 1000), np.arange(0, 1000)),
        (np.arange(0, 1000, 10), np.arange(0, 2000), np.arange(0, 2000)),
        (np.arange(0, 1000, 10), np.arange(0, 1000, 2), np.arange(0, 1000, 2)),
        (np.arange(0, 1000, 10), np.arange(0, 2000, 5), np.arange(0, 2000, 5)),
    ],
)
def test_basic(target_ts, sensor_ts, sensor_data):
    target_ts = np.arange(0, 1000, 10)
    sensor = NumpyTimeseries(sensor_ts, sensor_data)

    matched_data = sample(target_ts, sensor, method=MatchingMethod.AFTER)
    assert len(matched_data) == len(target_ts)
    for target, val in zip(target_ts, matched_data):
        assert val == target

    assert matched_data[0] == 0
    assert matched_data[1] == 10
    assert matched_data[21] == 210
    assert matched_data[-1] == 990

    assert list(matched_data[0:1]) == [0]
    assert list(matched_data[1:3]) == [10, 20]
    assert list(matched_data[10:15]) == [100, 110, 120, 130, 140]
    assert list(matched_data[-2:]) == [980, 990]


def test_out_of_range():
    target_ts = np.arange(1000)

    sensor = NumpyTimeseries(np.arange(-2000, -1000))
    matched_data = sample(target_ts, sensor, method=MatchingMethod.AFTER)
    for val in matched_data:
        assert val is None

    sensor = NumpyTimeseries(np.arange(2000, 3000))
    matched_data = sample(target_ts, sensor, method=MatchingMethod.AFTER)
    for val in matched_data:
        assert val == 2000


def test_minimizes_distance():
    target_ts = np.random.random_integers(0, 1000, 50)
    target_ts.sort()

    sensor_ts = np.random.random_integers(0, 1000, 50)
    sensor_ts.sort()
    sensor = NumpyTimeseries(sensor_ts)

    matched_data = sample(target_ts, sensor, method=MatchingMethod.AFTER)
    for t, v in zip(target_ts, matched_data):
        min_delta = sensor_ts - t
        min_delta = min_delta[min_delta >= 0].min()
        assert v - t == min_delta
