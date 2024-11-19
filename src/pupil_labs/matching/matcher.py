from enum import Enum
from typing import (
    Callable,
    Generic,
    Iterator,
    Optional,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd

from pupil_labs.video.array_like import ArrayLike


class MatchingMethod(Enum):
    NEAREST = "nearest"
    BEFORE = "before"
    AFTER = "after"
    INTERPOLATE = "interpolate"


T = TypeVar("T")


class MatchedIndividual(Generic[T], ArrayLike[T]):
    def __init__(
        self,
        target_ts: ArrayLike[int] | ArrayLike[float],
        timeseries: ArrayLike[T],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[float] = None,
        get_timeseries_ts: Callable[
            [ArrayLike[T]], ArrayLike[int]
        ] = lambda timeseries: timeseries.timestamps,
    ) -> None:
        self.target_ts = target_ts
        self.timeseries = timeseries
        self.method = method
        self.tolerance = tolerance

        target_df = pd.DataFrame(target_ts, columns=["target_ts"])
        target_df.index.name = "target"
        target_df.reset_index(inplace=True)

        ts = get_timeseries_ts(timeseries)
        data_df = pd.DataFrame(ts, columns=["data_ts"])
        data_df.index.name = "data"
        data_df.reset_index(inplace=True)

        if method == MatchingMethod.NEAREST:
            self.matching_df = pd.merge_asof(
                target_df,
                data_df,
                left_on="target_ts",
                right_on="data_ts",
                direction="nearest",
                tolerance=tolerance,
            )
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.target_ts)

    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> ArrayLike[T]: ...
    def __getitem__(self, key: int | slice) -> T | ArrayLike[T]:
        if isinstance(key, int):
            if self.method == MatchingMethod.NEAREST:
                data_index = self.matching_df.loc[key, "data"]
                if np.isnan(data_index):
                    return None

                data_index = int(data_index)
            else:
                raise NotImplementedError

            return self.timeseries[data_index]

        else:
            raise NotImplementedError

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]


class MatchedGroup(Generic[T], ArrayLike[T]):
    def __init__(
        self,
        target_ts: ArrayLike[int] | ArrayLike[float],
        timeseries: ArrayLike[T],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[float] = None,
        get_timeseries_ts: Callable[
            [ArrayLike[T]], ArrayLike[int] | ArrayLike[float]
        ] = lambda timeseries: timeseries.timestamps,
    ) -> None:
        self.target_ts = target_ts
        self.timeseries = timeseries
        self.method = method
        self.tolerance = tolerance

        target_df = pd.DataFrame(target_ts, columns=["target_ts"])
        target_df.index.name = "target"
        target_df.reset_index(inplace=True)

        ts = get_timeseries_ts(timeseries)
        data_df = pd.DataFrame(ts, columns=["data_ts"])
        data_df.index.name = "data"
        data_df.reset_index(inplace=True)

        if method == MatchingMethod.NEAREST:
            self.matching_df = (
                pd.merge_asof(
                    data_df,
                    target_df,
                    left_on="data_ts",
                    right_on="target_ts",
                    direction="nearest",
                    tolerance=tolerance,
                )
                .set_index("target_ts")
                .dropna()
            )
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.target_ts)

    def __getitem__(self, key: int | slice) -> ArrayLike[T]:
        if isinstance(key, int):
            if self.method == MatchingMethod.NEAREST:
                target_ts = self.target_ts[key]
                try:
                    data_index = self.matching_df.loc[target_ts, "data"]
                    if isinstance(data_index, pd.Series):
                        data_index = data_index.values
                    else:
                        data_index = np.array([data_index])
                    data_index = data_index.astype(np.int64)
                except KeyError:
                    return None
            else:
                raise NotImplementedError

            return self.timeseries[data_index]

        else:
            raise NotImplementedError

    def __iter__(self) -> Iterator[ArrayLike[T]]:
        for i in range(len(self)):
            yield self[i]
