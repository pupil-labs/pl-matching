from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

import pupil_labs.neon_recording as nr
from pupil_labs.matching.matcher import Matcher, MatchingMethod
from pupil_labs.neon_recording.numpy_timeseries import NumpyTimeseries


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
    target_ts = sensor.timestamps
    matcher = Matcher(
        target_ts, [sensor], MatchingMethod.NEAREST, include_timeseries_ts=True
    )

    for i in range(len(matcher)):
        t, _ = matcher[i]
        assert t == target_ts[i]


@pytest.mark.parametrize(
    "sensor_name",
    [
        "eye",
        "scene",
    ],
)
def test_video_match_itself(rec: nr.NeonRecording, sensor_name: str):
    sensor = getattr(rec, sensor_name)
    target_ts = sensor.timestamps
    matcher = Matcher(target_ts, [sensor], MatchingMethod.NEAREST)

    for i in range(len(matcher)):
        assert matcher[i][0].index == i


def test_iteration(rec: nr.NeonRecording):
    target_ts = np.arange(0, 10, 0.1) + rec.start_ts
    matcher = Matcher(
        target_ts,
        [rec.gaze, rec.eye_state, rec.eye, rec.scene, rec.events],
        MatchingMethod.NEAREST,
    )

    for _ in matcher:
        pass


@pytest.mark.parametrize(
    "sensor_name, csv_name, data_columns, tolerance",
    [
        ("gaze", "gaze.csv", ["gaze x [px]", "gaze y [px]"], 5e-2),
        # (
        #     "eye_state",
        #     "3d_eye_states.csv",
        #     [
        #         "pupil diameter left [mm]",
        #         "eyeball center left x [mm]",
        #         "eyeball center left y [mm]",
        #         "eyeball center left z [mm]",
        #         "optical axis left x",
        #         "optical axis left y",
        #         "optical axis left z",
        #         "pupil diameter right [mm]",
        #         "eyeball center right x [mm]",
        #         "eyeball center right y [mm]",
        #         "eyeball center right z [mm]",
        #         "optical axis right x",
        #         "optical axis right y",
        #         "optical axis right z",
        #     ],
        #     5e-2,
        # ),
    ],
)
def test_compare_to_csv(
    rec: nr.NeonRecording,
    csv_export_path: Path,
    sensor_name: str,
    csv_name: str,
    data_columns: List[str],
    tolerance: float,
):
    csv_data = pd.read_csv(csv_export_path / csv_name)
    csv_data["timestamp [s]"] = csv_data["timestamp [ns]"] / 1e9
    csv_data = NumpyTimeseries(
        csv_data["timestamp [s]"].values,
        csv_data[data_columns].values,
    )
    sensor = getattr(rec, sensor_name)
    target_ts = sensor.timestamps
    matcher = Matcher(
        target_ts,
        [sensor, csv_data],
        MatchingMethod.NEAREST,
    )

    for a, b in matcher:
        # Real-time and post hoc data is not identical, but should be close
        assert np.allclose(a, b, rtol=tolerance)
