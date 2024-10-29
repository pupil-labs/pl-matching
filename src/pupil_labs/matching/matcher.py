from enum import Enum
from typing import Iterator, Optional, Sequence, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pupil_labs.video.array_like import ArrayLike

TimesArray = npt.NDArray[np.number]


class MatchingMethod(Enum):
    NEAREST = "nearest"
    BEFORE = "before"
    AFTER = "after"
    INTERPOLATE = "interpolate"


class Timeseries(ArrayLike):
    timestamps: TimesArray


class Matcher(ArrayLike[Sequence]):
    def __init__(
        self,
        target_ts: TimesArray,
        timeseries: Timeseries | Sequence[Timeseries],
        method: MatchingMethod,
        tolerance: Optional[float] = None,
        include_ts: bool = False,
    ) -> None:
        if isinstance(timeseries, Timeseries):
            timeseries = [timeseries]

        if len(timeseries) == 0:
            raise ValueError("At least one timeseries is required")

        self.target_ts = target_ts
        self.timeseries = timeseries
        self.method = method
        self.tolerance = tolerance
        self.include_ts = include_ts

        target_ts_df = pd.DataFrame(target_ts, columns=["ts"])
        target_ts_df.index.name = "target"
        target_ts_df.reset_index(inplace=True)

        dfs = []
        for ts in timeseries:
            df = pd.DataFrame(ts.timestamps, columns=["ts"])
            df.index.name = "data"
            df.reset_index(inplace=True)
            dfs.append(df)

        self.matching_dfs = []
        if method == MatchingMethod.NEAREST:
            for df in dfs:
                matched = pd.merge_asof(
                    target_ts_df, df, on="ts", direction="nearest", tolerance=tolerance
                )
                self.matching_dfs.append(matched)
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.target_ts)

    @overload
    def __getitem__(self, key: int) -> Sequence: ...
    @overload
    def __getitem__(self, key: slice) -> ArrayLike[Sequence]: ...
    def __getitem__(self, key: int | slice) -> Sequence | ArrayLike[Sequence]:
        if isinstance(key, int):
            result = []
            for ts, matched in zip(self.timeseries, self.matching_dfs):
                target_idx = int(matched.loc[key, "data"])
                val = ts[target_idx]
                if self.include_ts:
                    val = (ts.timestamps[target_idx], val)
                result.append(val)
            return result
        else:
            raise NotImplementedError

    def __iter__(self) -> Iterator[Sequence]:
        for i in range(len(self)):
            yield self[i]
