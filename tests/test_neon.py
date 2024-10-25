from pathlib import Path

import pytest

import pupil_labs.neon_recording as nr
from pupil_labs.matching.matcher import Matcher, MatchingMethod


@pytest.mark.parametrize(
    "sensor_name",
    ["gaze", "eye_state", "events"],
)
def test_scalar_data(sensor_name: str):
    rec_dir = Path("tests/data/demo_recording")
    rec = nr.load(rec_dir)

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
def test_video_data(sensor_name: str):
    rec_dir = Path("tests/data/demo_recording")
    rec = nr.load(rec_dir)

    sensor = getattr(rec, sensor_name)
    target_ts = sensor.timestamps
    matcher = Matcher(target_ts, [sensor], MatchingMethod.NEAREST)

    for i in range(len(matcher)):
        assert matcher[i][0].index == i


def test_multi_rec():
    rec_dir = Path("tests/data/demo_recording")
    rec1 = nr.load(rec_dir)
    rec1 = nr.load(rec_dir)

    target_ts = rec1.gaze.timestamps
    matcher = Matcher(target_ts, [rec1.gaze], MatchingMethod.NEAREST)

    for i in range(len(matcher)):
        assert matcher[i][0].ts == target_ts[i]
