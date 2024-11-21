from typing import (
    Generic,
    Iterator,
    Optional,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

from pupil_labs.matching.matching_method import MatchingMethod
from pupil_labs.video.array_like import ArrayLike

T = TypeVar("T")


class SampledDataBase(Generic[T]):
    def __init__(
        self,
        target_ts: ArrayLike[int],
        timeseries: ArrayLike[T],
        method: MatchingMethod,
        tolerance: Optional[int],
        matching_df: pd.DataFrame,
    ) -> None:
        self._target_ts = np.array(target_ts, dtype=np.int64)
        self.timeseries = timeseries
        self.method = method
        self.tolerance = tolerance
        self.matching_df = matching_df

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        return self._target_ts

    def __len__(self) -> int:
        return len(self._target_ts)


class SampledData(Generic[T], SampledDataBase, ArrayLike[Optional[T]]):
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
                method=self.method,
                tolerance=self.tolerance,
                matching_df=self.matching_df.iloc[key],
            )

    def __iter__(self) -> Iterator[Optional[T]]:
        for i in range(len(self)):
            yield self[i]


# TODO: make this class an ArrayLike as well
class SampledDataGroups(SampledDataBase[T]):
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
