"""The privacy boundary, written so it can be audited rather than believed.

The claim Preempt makes to a ward is narrow and checkable: **in strict mode, no
bytes derived from camera pixels are written to disk, sent over the network, or
held after the pose has been read.** Not "we anonymise", not "we blur faces".
Nothing leaves.

How that is enforced rather than promised:

1. `PoseEstimator.estimate` is the only code that touches an image. It returns
   keypoints. The caller holds no reference to the frame afterwards, and
   `Pipeline` calls `guard.release(frame)` to overwrite the buffer in place so a
   later reader of that memory finds zeros.
2. Every write to disk goes through `PrivacyGuard.emit`. An artefact must
   declare its `Provenance`. `CAMERA` provenance in strict mode raises
   `PrivacyViolation`; there is no flag on the call that overrides it.
3. `SYNTHETIC` artefacts are drawn by `render.py` from keypoints and zone
   polygons onto a blank canvas. The renderer is never handed a frame -- its
   signature does not accept one.
4. The guard counts `camera_bytes_persisted`. `tests/test_privacy.py` asserts it
   is zero after a full run, asserts every artefact decodes to something with no
   correlation to any source frame, and asserts that asking to write a real frame
   raises.

`diagnostic` mode exists for engineering on a bench and is refused by the
deployed service: `Pipeline` raises if the request asks for it and
`PREEMPT_ALLOW_DIAGNOSTIC` is not set in the environment.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class PrivacyViolation(RuntimeError):
    """An attempt to persist camera-derived bytes while strict mode is on."""


class Provenance(enum.Enum):
    """Where an artefact's bytes came from. Every emit must say."""

    CAMERA = "camera"
    """Pixels from the camera, or anything computed pixel-wise from them."""
    SYNTHETIC = "synthetic"
    """Drawn from keypoints, zone polygons and text on a blank canvas."""
    DERIVED = "derived"
    """Numbers: JSON, CSV, a chart of a time series. No image content."""


@dataclass
class Artefact:
    """One thing the run wrote, with the reason it was allowed to."""

    name: str
    provenance: Provenance
    bytes_written: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provenance": self.provenance.value,
            "bytes": self.bytes_written,
            "sha256": self.sha256,
        }


@dataclass
class PrivacyLedger:
    """The numbers the UI prints next to the pose figure."""

    mode: str = "strict"
    frames_examined: int = 0
    frames_retained: int = 0
    camera_bytes_read: int = 0
    camera_bytes_persisted: int = 0
    synthetic_bytes_persisted: int = 0
    derived_bytes_persisted: int = 0
    keypoint_bytes_kept: int = 0
    artefacts: list[Artefact] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.camera_bytes_persisted == 0 and self.frames_retained == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "frames_examined": self.frames_examined,
            "frames_retained": self.frames_retained,
            "camera_bytes_read": self.camera_bytes_read,
            "camera_bytes_persisted": self.camera_bytes_persisted,
            "synthetic_bytes_persisted": self.synthetic_bytes_persisted,
            "derived_bytes_persisted": self.derived_bytes_persisted,
            "keypoint_bytes_kept": self.keypoint_bytes_kept,
            "artefacts": [a.to_dict() for a in self.artefacts],
            "refusals": list(self.refusals),
            "clean": self.clean,
        }


class PrivacyGuard:
    """The one door to the filesystem, and the record of what went through it."""

    def __init__(self, mode: str = "strict", sink: Any | None = None) -> None:
        if mode not in ("strict", "diagnostic"):
            raise ValueError(f"privacy mode must be strict or diagnostic, not {mode!r}")
        if mode == "diagnostic" and not os.environ.get("PREEMPT_ALLOW_DIAGNOSTIC"):
            raise PrivacyViolation(
                "diagnostic mode keeps camera frames and is refused unless "
                "PREEMPT_ALLOW_DIAGNOSTIC is set in the environment. "
                "The deployed service never sets it."
            )
        self.mode = mode
        self.ledger = PrivacyLedger(mode=mode)
        self._sink = sink  # a JobContext-like object, or a directory Path, or None

    # -- frames -------------------------------------------------------------
    def examined(self, image: np.ndarray) -> None:
        self.ledger.frames_examined += 1
        self.ledger.camera_bytes_read += int(image.nbytes)

    def release(self, image: np.ndarray) -> None:
        """Overwrite a frame buffer in place once the pose has been read from it.

        numpy will free it soon enough; zeroing it means that until it does, the
        bytes in that page are not a picture of a patient.
        """
        if self.mode == "strict":
            try:
                image[...] = 0
            except (ValueError, TypeError):  # a read-only view; nothing to do
                pass
        else:
            self.ledger.frames_retained += 1

    def keypoints_kept(self, n_bytes: int) -> None:
        self.ledger.keypoint_bytes_kept += int(n_bytes)

    # -- artefacts ----------------------------------------------------------
    def emit(self, name: str, data: bytes, provenance: Provenance) -> str | None:
        """Write an artefact, or refuse. Returns the URI the UI should fetch."""
        if provenance is Provenance.CAMERA and self.mode == "strict":
            self.ledger.refusals.append(
                f"refused to write {name!r}: camera-derived bytes, privacy mode is strict"
            )
            raise PrivacyViolation(
                f"{name}: camera-derived bytes cannot be persisted in strict mode"
            )
        import hashlib

        digest = hashlib.sha256(data).hexdigest()
        self.ledger.artefacts.append(Artefact(name, provenance, len(data), digest))
        if provenance is Provenance.CAMERA:
            self.ledger.camera_bytes_persisted += len(data)
        elif provenance is Provenance.SYNTHETIC:
            self.ledger.synthetic_bytes_persisted += len(data)
        else:
            self.ledger.derived_bytes_persisted += len(data)
        return self._write(name, data)

    def _write(self, name: str, data: bytes) -> str | None:
        sink = self._sink
        if sink is None:
            return None
        if isinstance(sink, Path):
            sink.mkdir(parents=True, exist_ok=True)
            (sink / name).write_bytes(data)
            return str(sink / name)
        if hasattr(sink, "save_evidence"):
            return str(sink.save_evidence(name, data))
        raise TypeError(f"privacy sink {type(sink).__name__} cannot write artefacts")

    def to_dict(self) -> dict[str, Any]:
        return self.ledger.to_dict()
