from pathlib import Path

import numpy as np
import pytest

import pupil_labs.neon_recording as nr
from pupil_labs.matching.matcher import Matcher, MatchingMethod


# create fixture for neon recording
@pytest.fixture
def rec():
    rec_dir = Path("tests/data/demo_recording")
    return nr.load(rec_dir)


@pytest.mark.parametrize(
    "sensor_name",
    ["gaze", "eye_state", "events"],
)
def test_scalar_data(rec: nr.NeonRecording, sensor_name: str):
    sensor = getattr(rec, sensor_name)
    target_ts = sensor.timestamps
    matcher = Matcher(target_ts, [sensor], MatchingMethod.NEAREST)

    for i in range(len(matcher)):
        assert matcher[i][0].ts == target_ts[i]


@pytest.mark.parametrize(
    "sensor_name",
    [
        "eye",
        "scene",
    ],
)
def test_video_data(rec: nr.NeonRecording, sensor_name: str):
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
