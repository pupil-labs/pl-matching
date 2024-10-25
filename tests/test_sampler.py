from typing import Iterator, overload

import numpy as np
import numpy.typing as npt
import pytest

from pupil_labs.matching.matcher import Matcher, MatchingMethod, Sensor
from pupil_labs.video.array_like import ArrayLike


class ArraySensor(Sensor):
    def __init__(self, timestamps: npt.NDArray[np.int32]):
        self.timestamps = timestamps

    @overload
    def __getitem__(self, key: int, /) -> int: ...
    @overload
    def __getitem__(self, key: slice, /) -> ArrayLike[int]: ...
    def __getitem__(self, key: int | slice, /) -> int | ArrayLike[int]:
        return self.timestamps[key]

    def __len__(self) -> int:
        return len(self.timestamps)

    def __iter__(self) -> Iterator[int]:
        for i in range(len(self)):
            yield self[i]


@pytest.mark.parametrize(
    "target_ts, sensor1, sensor2",
    [
        (np.arange(100), np.arange(100), np.arange(100)),
        (np.arange(0, 1000, 10), np.arange(1000), np.arange(2000)),
        (np.arange(0, 1000, 10), np.arange(0, 1000, 2), np.arange(0, 2000, 5)),
    ],
)
def test_matcher_nearest_basic(target_ts, sensor1, sensor2):
    sensor1 = ArraySensor(sensor1)
    sensor2 = ArraySensor(sensor2)

    matcher = Matcher(target_ts, [sensor1, sensor2], MatchingMethod.NEAREST)

    for target, (val1, val2) in zip(target_ts, matcher):
        assert val1 == val2 == target


def test_matcher_nearest_out_of_range():
    target_ts = np.arange(1000)
    sensor1 = ArraySensor(np.arange(-2000, -1000))
    sensor2 = ArraySensor(np.arange(2000, 3000))

    matcher = Matcher(target_ts, [sensor1, sensor2], MatchingMethod.NEAREST)

    for val1, val2 in matcher:
        assert val1 == -1001
        assert val2 == 2000
