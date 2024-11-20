import numpy as np
import pytest

from pupil_labs.matching import MatchingMethod, NumpyTimeseries, sample


@pytest.mark.parametrize(
    "target_ts, sensor_ts, sensor_data",
    [
        (np.arange(0, 1000, 10), np.arange(0, 1000, 10), np.arange(0, 1000, 10)),
        (np.arange(0, 1000, 10), np.arange(0, 1000, 10) + 1, np.arange(0, 1000, 10)),
        (np.arange(0, 1000, 10), np.arange(0, 1000, 10) - 1, np.arange(0, 1000, 10)),
        (np.arange(0, 1000, 10), np.arange(0, 1000), np.arange(0, 1000)),
        (np.arange(0, 1000, 10), np.arange(0, 2000), np.arange(0, 2000)),
        (np.arange(0, 1000, 10), np.arange(0, 1000, 2), np.arange(0, 1000, 2)),
        (np.arange(0, 1000, 10), np.arange(0, 2000, 5), np.arange(0, 2000, 5)),
    ],
)
def test_nearest_basic(target_ts, sensor_ts, sensor_data):
    target_ts = np.arange(0, 1000, 10)
    sensor = NumpyTimeseries(sensor_ts, sensor_data)

    matched_datas = [
        sample(target_ts, sensor),
        sample(target_ts, sensor, method=MatchingMethod.NEAREST),
    ]
    for matched_data in matched_datas:
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


@pytest.mark.parametrize(
    "target_ts, sensor1, sensor2",
    [
        (np.arange(100), np.arange(100), np.arange(100)),
        (np.arange(0, 1000, 10), np.arange(1000), np.arange(2000)),
        (np.arange(0, 1000, 10), np.arange(0, 1000, 2), np.arange(0, 2000, 5)),
    ],
)
def test_slicing(target_ts, sensor1, sensor2):
    sensor1 = NumpyTimeseries(sensor1)
    sensor2 = NumpyTimeseries(sensor2)

    matched_data1 = sample(target_ts, sensor1)
    matched_data2 = sample(target_ts, sensor2)

    n = 10
    matched_data1 = matched_data1[n : n + 10]
    matched_data2 = matched_data2[n : n + 10]
    target_ts = target_ts[n : n + 10]

    for target, val1, val2 in zip(target_ts, matched_data1, matched_data2):
        assert val1 == val2 == target


def test_nearest_out_of_range():
    target_ts = np.arange(1000)
    sensor1 = NumpyTimeseries(np.arange(-2000, -1000))
    sensor2 = NumpyTimeseries(np.arange(2000, 3000))

    matched_data1 = sample(target_ts, sensor1)
    matched_data2 = sample(target_ts, sensor2)

    for val1, val2 in zip(matched_data1, matched_data2):
        assert val1 == -1001
        assert val2 == 2000


def test_nearest_minimizes_distance():
    ts1 = np.random.random_integers(0, 1000, 50)
    ts1.sort()
    sensor1 = NumpyTimeseries(ts1)

    ts2 = np.random.random_integers(0, 1000, 50)
    ts2.sort()
    sensor2 = NumpyTimeseries(ts2)

    matched_data1 = sample(ts1, sensor1)
    matched_data2 = sample(ts1, sensor2)

    for a, b in zip(matched_data1, matched_data2):
        min_delta = np.min(np.abs(ts2 - a))
        assert np.abs(a - b) == min_delta
