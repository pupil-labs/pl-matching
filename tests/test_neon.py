from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pupil_labs.neon_recording as nr
from pupil_labs.matching import MatchingMethod, NumpyTimeseries, SampledData


@pytest.fixture
def rec():
    rec_dir = Path("tests/data/demo_recording")
    return nr.load(rec_dir)


@pytest.fixture
def csv_export_path():
    rec_dir = Path("tests/data/demo_csv")
    return rec_dir


@pytest.mark.parametrize(
    "sensor_name",
    ["gaze", "eye_state", "events"],
)
def test_scalar_match_itself(rec: nr.NeonRecording, sensor_name: str):
    sensor = getattr(rec, sensor_name)
    target_ts = sensor.abs_timestamp
    matcher = SampledData.sample(target_ts, sensor, MatchingMethod.NEAREST)

    for i in range(len(matcher)):
        datum = matcher[i]
        assert datum.abs_ts == target_ts[i]


@pytest.mark.parametrize(
    "sensor_name",
    [
        "eye",
        "scene",
    ],
)
def test_video_match_itself(rec: nr.NeonRecording, sensor_name: str):
    sensor = getattr(rec, sensor_name)
    target_ts = sensor.abs_timestamp
    matcher = SampledData.sample(target_ts, sensor, MatchingMethod.NEAREST)

    for i in range(len(matcher)):
        assert matcher[i].index == i


def test_rt_gaze_to_csv_gaze(
    rec: nr.NeonRecording,
    csv_export_path: Path,
):
    csv_data = pd.read_csv(csv_export_path / "gaze.csv")
    csv_data = NumpyTimeseries(
        csv_data["timestamp [ns]"].values,
        csv_data[["gaze x [px]", "gaze y [px]"]].values,
    )
    target_ts = rec.gaze.abs_timestamp
    gaze_rt_data = SampledData.sample(
        target_ts,
        rec.gaze,
        MatchingMethod.NEAREST,
    )
    gaze_csv_data = SampledData.sample(
        target_ts,
        csv_data,
        MatchingMethod.NEAREST,
    )

    matched_data = zip(gaze_rt_data, gaze_csv_data)

    for a, b in matched_data:
        # Real-time and post hoc data is not identical, but should be close
        assert np.allclose((a.x, a.y), b, rtol=5e-2)


def test_sampling_sampled_data(rec: nr.NeonRecording):
    target_ts = rec.gaze.abs_timestamp
    gaze_data = SampledData.sample(
        target_ts,
        rec.gaze,
    )

    gaze_data_sampled = SampledData.sample(
        target_ts,
        gaze_data,
    )

    for a, b in zip(gaze_data, gaze_data_sampled):
        assert a == b
