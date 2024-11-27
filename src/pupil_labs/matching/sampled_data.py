from typing import Callable, Generic, Iterator, Literal, Optional, TypeVar, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pupil_labs.matching.matching_method import MatchingMethod
from pupil_labs.video.array_like import ArrayLike

T = TypeVar("T", covariant=True)


# TODO Make this a full NeonTimeseries
class SampledData(Generic[T], ArrayLike[Optional[T]]):
    def __init__(
        self,
        target_ts: ArrayLike[int],
        timeseries: ArrayLike[T],
        matching_df: pd.DataFrame,
    ) -> None:
        self._target_ts = np.array(target_ts, dtype=np.int64)
        self.timeseries = timeseries
        self.matching_df = matching_df

    @staticmethod
    def sample(
        target_ts: ArrayLike[int],
        timeseries: ArrayLike[T],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[int] = None,
        get_timeseries_ts: Callable[
            [ArrayLike[T]], ArrayLike[int]
        ] = lambda timeseries: timeseries.abs_timestamp,  # type: ignore
    ) -> "SampledData[T]":
        target_ts = np.array(target_ts)
        target_df = pd.DataFrame(target_ts, columns=["target_ts"])
        target_df.index.name = "target"
        target_df.reset_index(inplace=True)

        ts = get_timeseries_ts(timeseries)
        ts = np.array(ts)
        data_df = pd.DataFrame(ts, columns=["data_ts"])
        data_df.index.name = "data"
        data_df.reset_index(inplace=True)

        direction: Literal["nearest", "backward", "forward"]
        if method == MatchingMethod.NEAREST:
            direction = "nearest"
        elif method == MatchingMethod.BEFORE:
            direction = "backward"
        elif method == MatchingMethod.AFTER:
            direction = "forward"
        else:
            raise ValueError(f"Invalid method: {method}")

        matching_df = pd.merge_asof(
            target_df,
            data_df,
            left_on="target_ts",
            right_on="data_ts",
            direction=direction,
            tolerance=tolerance,
        )
        return SampledData(target_ts, timeseries, matching_df)

    @property
    def abs_timestamp(self) -> npt.NDArray[np.int64]:
        return self._target_ts

    def __len__(self) -> int:
        return len(self._target_ts)

    @overload
    def __getitem__(self, key: int) -> Optional[T]: ...
    @overload
    def __getitem__(self, key: slice) -> "SampledData[T]": ...
    def __getitem__(self, key: int | slice) -> "Optional[T] | SampledData[T]":
        if isinstance(key, int):
            data_index = self.matching_df.iloc[key]["data"]
            if np.isnan(data_index):
                return None

            data_index = int(data_index)
            result: Optional[T] = self.timeseries[data_index]
            return result

        else:
            return SampledData(
                target_ts=self._target_ts[key],
                timeseries=self.timeseries,
                matching_df=self.matching_df.iloc[key],
            )

    def __iter__(self) -> Iterator[Optional[T]]:
        for i in range(len(self)):
            yield self[i]


# TODO: make this class an ArrayLike as well
class SampledDataGroups(Generic[T]):
    def __init__(
        self,
        target_ts: ArrayLike[int],
        timeseries: ArrayLike[T],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[int] = None,
        get_timeseries_ts: Callable[
            [ArrayLike[T]], ArrayLike[int]
        ] = lambda timeseries: timeseries.abs_timestamp,  # type: ignore
    ) -> None:
        self._target_ts = np.array(target_ts, dtype=np.int64)
        self.timeseries = timeseries
        self.method = method
        self.tolerance = tolerance

        target_ts = np.array(target_ts)
        target_df = pd.DataFrame(target_ts, columns=["target_ts"])
        target_df.index.name = "target"
        target_df.reset_index(inplace=True)

        ts = get_timeseries_ts(timeseries)
        ts = np.array(ts)
        data_df = pd.DataFrame(ts, columns=["data_ts"])
        data_df.index.name = "data"
        data_df.reset_index(inplace=True)

        # TODO: Support other methods
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

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        return self._target_ts

    def __len__(self) -> int:
        return len(self._target_ts)

    def __getitem__(self, key: int | slice) -> ArrayLike[T]:
        if isinstance(key, int):
            target_ts = self._target_ts[key]
            try:
                data_selection = self.matching_df.loc[target_ts, "data"]
                if isinstance(data_selection, pd.Series):
                    data_index = data_selection.values
                else:
                    data_index = np.array([data_selection])

            except KeyError:
                return []
            # TODO: check if timeseries is an array and use array indexing instead if so
            result = [self.timeseries[int(i)] for i in data_index.astype(int)]
            return result

        else:
            raise NotImplementedError

    def __iter__(self) -> Iterator[ArrayLike[T]]:
        for i in range(len(self)):
            yield self[i]
