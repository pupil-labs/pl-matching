from enum import Enum
from typing import Iterator, Sequence, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pupil_labs.video.array_like import ArrayLike

TimesArray = npt.NDArray[np.int32]


class MatchingMethod(Enum):
    NEAREST = "nearest"
    BEFORE = "before"
    AFTER = "after"
    INTERPOLATE = "interpolate"


class Sensor(ArrayLike):
    timestamps: TimesArray


class Sampler(ArrayLike[Sequence]):
    def __init__(
        self, target_ts: TimesArray, sensors: Sequence[Sensor], method: MatchingMethod
    ) -> None:
        if len(sensors) == 0:
            raise ValueError("At least one sensor is required")

        self.target_ts = target_ts
        self.sensors = sensors
        self.method = method

        target_ts_df = pd.DataFrame(target_ts, columns=["ts"])
        target_ts_df.index.name = "target"
        target_ts_df.reset_index(inplace=True)

        dfs = []
        for sensor in sensors:
            df = pd.DataFrame(sensor.timestamps, columns=["ts"])
            df.index.name = "sensor"
            df.reset_index(inplace=True)
            dfs.append(df)

        self.matching_dfs = []
        if method == MatchingMethod.NEAREST:
            for df in dfs:
                matched = pd.merge_asof(target_ts_df, df, on="ts", direction="nearest")
                self.matching_dfs.append(matched)

    def __len__(self) -> int:
        return len(self.target_ts)

    @overload
    def __getitem__(self, key: int) -> Sequence: ...
    @overload
    def __getitem__(self, key: slice) -> ArrayLike[Sequence]: ...
    def __getitem__(self, key: int | slice) -> Sequence | ArrayLike[Sequence]:
        if isinstance(key, int):
            result = []
            for sensor, matched in zip(self.sensors, self.matching_dfs):
                target_idx = matched.loc[key, "sensor"].item()
                val = sensor[target_idx]
                result.append(val)
            return result
        else:
            raise NotImplementedError

    def __iter__(self) -> Iterator[Sequence]:
        for i in range(len(self)):
            yield self[i]
