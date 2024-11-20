from typing import Callable, Literal, Optional, TypeVar, Union, overload

import numpy as np
import pandas as pd

from pupil_labs.matching.matching_method import MatchingMethod
from pupil_labs.matching.sampled_data import SampledData, SampledDataGroups
from pupil_labs.video.array_like import ArrayLike

T = TypeVar("T")


# Typing of boolean keyword arguments with a default value is complicated if they
# influence the return type. The below was created based on:
# https://github.com/python/mypy/issues/8634
@overload
def sample(
    target_ts: ArrayLike[int],
    timeseries: ArrayLike[T],
    method: MatchingMethod = MatchingMethod.NEAREST,
    tolerance: Optional[int] = None,
    get_timeseries_ts: Callable[
        [ArrayLike[T]], ArrayLike[int]
    ] = lambda timeseries: timeseries.timestamps,  # type: ignore
    return_groups: Literal[False] = ...,
) -> SampledData[T]: ...
@overload
def sample(
    target_ts: ArrayLike[int],
    timeseries: ArrayLike[T],
    method: MatchingMethod = MatchingMethod.NEAREST,
    tolerance: Optional[int] = None,
    get_timeseries_ts: Callable[
        [ArrayLike[T]], ArrayLike[int]
    ] = lambda timeseries: timeseries.timestamps,  # type: ignore
    *,
    return_groups: Literal[True],
) -> SampledDataGroups[T]: ...
@overload
def sample(
    target_ts: ArrayLike[int],
    timeseries: ArrayLike[T],
    method: MatchingMethod = MatchingMethod.NEAREST,
    tolerance: Optional[int] = None,
    get_timeseries_ts: Callable[
        [ArrayLike[T]], ArrayLike[int]
    ] = lambda timeseries: timeseries.timestamps,  # type: ignore
    *,
    return_groups: Literal[True],
) -> SampledDataGroups[T]: ...
def sample(
    target_ts: ArrayLike[int],
    timeseries: ArrayLike[T],
    method: MatchingMethod = MatchingMethod.NEAREST,
    tolerance: Optional[int] = None,
    get_timeseries_ts: Callable[
        [ArrayLike[T]], ArrayLike[int]
    ] = lambda timeseries: timeseries.timestamps,  # type: ignore
    return_groups: bool = False,
) -> Union[SampledData[T] | SampledDataGroups[T]]:
    if return_groups:
        target_ts = np.array(target_ts)
        target_df = pd.DataFrame(target_ts, columns=["target_ts"])
        target_df.index.name = "target"
        target_df.reset_index(inplace=True)

        ts = get_timeseries_ts(timeseries)
        ts = np.array(ts)
        data_df = pd.DataFrame(ts, columns=["data_ts"])
        data_df.index.name = "data"
        data_df.reset_index(inplace=True)

        if method == MatchingMethod.NEAREST:
            matching_df = (
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

        return SampledDataGroups(
            target_ts,
            timeseries,
            method,
            tolerance,
            matching_df,
        )
    else:
        target_ts = np.array(target_ts)
        target_df = pd.DataFrame(target_ts, columns=["target_ts"])
        target_df.index.name = "target"
        target_df.reset_index(inplace=True)

        ts = get_timeseries_ts(timeseries)
        ts = np.array(ts)
        data_df = pd.DataFrame(ts, columns=["data_ts"])
        data_df.index.name = "data"
        data_df.reset_index(inplace=True)
        if method == MatchingMethod.INTERPOLATE:
            raise NotImplementedError
        else:
            direction_map = {
                MatchingMethod.NEAREST: "nearest",
                MatchingMethod.BEFORE: "backward",
                MatchingMethod.AFTER: "forward",
            }
            matching_df = pd.merge_asof(
                target_df,
                data_df,
                left_on="target_ts",
                right_on="data_ts",
                direction=direction_map[method],
                tolerance=tolerance,
            )

        return SampledData(
            target_ts,
            timeseries,
            method,
            tolerance,
            matching_df,
        )
