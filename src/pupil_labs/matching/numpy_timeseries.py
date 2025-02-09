from typing import Iterator, Optional, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pupil_labs.matching.matching_method import MatchingMethod
from pupil_labs.matching.sampled_data import SampledData
from pupil_labs.video.array_like import ArrayLike


class NumpyTimeseries:
    def __init__(
        self,
        abs_timestamp: npt.NDArray[np.int64],
        data: Optional[npt.NDArray] = None,
    ):
        self.abs_timestamp = abs_timestamp
        if data is None:
            self.data = abs_timestamp
        else:
            assert len(data) == len(abs_timestamp)
            self.data = data

    @overload
    def __getitem__(self, key: int, /) -> int: ...
    @overload
    def __getitem__(self, key: slice, /) -> ArrayLike[int]: ...
    def __getitem__(self, key: int | slice, /) -> int | ArrayLike[int]:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.abs_timestamp)

    def __iter__(self) -> Iterator[int]:
        for i in range(len(self)):
            yield self[i]

    def sample(
        self,
        target_ts: npt.NDArray[np.float64],
        method: MatchingMethod = MatchingMethod.NEAREST,
    ) -> ArrayLike:
        return SampledData.sample(target_ts, self, method)

    @staticmethod
    def from_dataframe(df: pd.DataFrame, ts_column: str) -> "NumpyTimeseries":
        time: npt.NDArray[np.int64] = df[ts_column].values.astype(np.int64)
        data = df.drop(columns=[ts_column]).values
        return NumpyTimeseries(time, data)
