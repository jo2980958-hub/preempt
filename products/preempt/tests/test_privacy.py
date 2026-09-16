"""The privacy guarantee, asserted rather than asserted-in-marketing.

The claim is: in strict mode, no bytes derived from camera pixels are written to
disk. These tests try to break that four different ways, on the real pipeline
running the real models over a real video file.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from preempt import PrivacyGuard, PrivacyViolation, Provenance
from preempt.pipeline import Pipeline
from preempt.render import encode_png, pose_card
from preempt.synth import default_room, make

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def noisy_video(tmp_path):
    """A video whose every frame is unique high-entropy content.

    Deliberately not a picture of a person: what is being tested is whether any
    of these bytes reach the disk, and content that is impossible to reproduce by
    accident makes the correlation test meaningful.
    """
    path = tmp_path / "source.mp4"
    rng = np.random.default_rng(4242)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*"mp4v"), 15.0, (320, 240)
    )
    frames = []
    for i in range(30):
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20 + i, 20), (120 + i, 200), (0, 0, 255), -1)
        frames.append(frame)
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0
    return path, frames


def test_strict_mode_persists_no_camera_bytes(tmp_path, noisy_video):
    path, _ = noisy_video
    room, _ = default_room()
    sink = tmp_path / "evidence"
    guard = PrivacyGuard("strict", sink=sink)
    pipeline = Pipeline(room, guard=guard)
    pipeline.run_video(path)

    ledger = guard.ledger
    assert ledger.frames_examined > 0, "the test video did not decode"
    assert ledger.camera_bytes_read > 0
    assert ledger.camera_bytes_persisted == 0
    assert ledger.frames_retained == 0
    assert ledger.clean is True


def test_written_artefacts_do_not_resemble_any_source_frame(tmp_path, noisy_video):
    """Not just 'we did not mean to write pixels': the files are checked."""
    path, frames = noisy_video
    room, _ = default_room()
    sink = tmp_path / "evidence"
    guard = PrivacyGuard("strict", sink=sink)
    pipeline = Pipeline(room, guard=guard)
    analysis = pipeline.run_video(path)

    card = pose_card(None, "Settled", ["a test card"], footer="test")
    guard.emit("card.png", encode_png(card), Provenance.SYNTHETIC)

    written = sorted(sink.glob("*.png")) if sink.is_dir() else []
    assert written, "the run wrote no evidence at all, so the test proves nothing"

    sources = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64) for f in frames]
    for artefact in written:
        image = cv2.imread(str(artefact), cv2.IMREAD_GRAYSCALE)
        assert image is not None
        for source in sources:
            resized = cv2.resize(source, (image.shape[1], image.shape[0]))
            a = image.astype(np.float64) - image.mean()
            b = resized - resized.mean()
            denominator = float(np.sqrt((a**2).sum() * (b**2).sum())) or 1.0
            correlation = abs(float((a * b).sum() / denominator))
            assert correlation < 0.35, (
                f"{artefact.name} correlates {correlation:.2f} with a source frame"
            )
    assert analysis.privacy["camera_bytes_persisted"] == 0


def test_emitting_a_camera_frame_raises_in_strict_mode(tmp_path):
    guard = PrivacyGuard("strict", sink=tmp_path)
    frame = np.full((32, 32, 3), 200, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", frame)
    assert ok
    with pytest.raises(PrivacyViolation):
        guard.emit("leak.png", buffer.tobytes(), Provenance.CAMERA)
    assert not list(tmp_path.glob("leak.png"))
    assert guard.ledger.refusals


def test_release_destroys_the_frame_in_place():
    guard = PrivacyGuard("strict")
    frame = np.full((16, 16, 3), 173, dtype=np.uint8)
    guard.release(frame)
    assert int(frame.max()) == 0, "the frame buffer still holds camera content"


def test_diagnostic_mode_is_refused_without_an_explicit_environment_opt_in(monkeypatch):
    monkeypatch.delenv("PREEMPT_ALLOW_DIAGNOSTIC", raising=False)
    with pytest.raises(PrivacyViolation):
        PrivacyGuard("diagnostic")
    monkeypatch.setenv("PREEMPT_ALLOW_DIAGNOSTIC", "1")
    guard = PrivacyGuard("diagnostic")
    assert guard.mode == "diagnostic"


def test_the_renderer_cannot_be_handed_a_frame():
    """The mechanical half of the argument: the signature will not take one."""
    import inspect

    from preempt import render

    for name in ("pose_card", "plan_card", "draw_figure", "room_outline"):
        signature = inspect.signature(getattr(render, name))
        for parameter in signature.parameters.values():
            assert "image" not in parameter.name, (
                f"render.{name} takes {parameter.name!r}; the renderer must never "
                "accept a camera frame"
            )


def test_a_pose_track_run_reads_no_image_bytes_at_all():
    sequence = make("bed-exit-steady")
    guard = PrivacyGuard("strict")
    pipeline = Pipeline(sequence.room, guard=guard)
    pipeline.run_track(sequence.frames, sequence.fps)
    assert guard.ledger.camera_bytes_read == 0
    assert guard.ledger.keypoint_bytes_kept > 0
    assert guard.ledger.clean
