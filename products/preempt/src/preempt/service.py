"""The web service: `servicekit` plus the two routes Preempt needs of its own.

The shell gives us upload, a progress stream, the result and the evidence
endpoint. Preempt adds a sample picker, for a reason that is about the product
rather than about convenience.

A judge opening a cold URL has to see the loop within seconds, and the obvious
way to arrange that is to bundle a video. Preempt cannot: shipping a video of a
patient in a hospital bed would contradict the only claim the product makes. So
what is bundled is a **pose track** -- the seventeen keypoints per frame that a
Preempt device holds after the camera stage, and nothing else. Replaying one is
replaying exactly what the device keeps. The demonstration and the privacy
argument are the same artefact.

Uploading a real video still runs the whole pipeline, YOLOX and RTMPose
included, so both paths are exercised and the `/version` endpoint says which
OpenCV built them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Request
from servicekit import JobContext, ProductInfo, ServiceConfig, ServiceError, create_app
from visioncore import RunRecord

from .config import RoomConfig
from .pipeline import PRODUCT, analyse_track, analyse_video
from .pose import RTMPOSE_T
from .privacy import PrivacyViolation

SAMPLES_DIR = Path(
    __import__("os").environ.get(
        "PREEMPT_SAMPLES_DIR", str(Path(__file__).resolve().parents[2] / "samples")
    )
)
WEB_DIR = Path(
    __import__("os").environ.get(
        "PREEMPT_WEB_DIR", str(Path(__file__).resolve().parents[2] / "web")
    )
)
"""The product's own front end. In the container the package lives in
site-packages, so `parents[2]` is not the product root and the env var is what
finds it. Getting this wrong is silent: `create_app` falls back to the shared
shell and serves a generic page that looks fine until you notice it is not
Preempt."""
ROOM_JSON = Path(
    __import__("os").environ.get(
        "PREEMPT_ROOM", str(Path(__file__).resolve().parents[2] / "rooms" / "default.json")
    )
)

TAGLINE = "A call before the fall, from an abstracted pose and nothing else"
DESCRIPTION = (
    "Preempt watches a hospital or care-home room for the movement that comes "
    "before standing, scores how steady a person is once they are up, checks "
    "what is in the way, and escalates. The frame is destroyed as soon as the "
    "pose has been read."
)


def samples() -> list[dict[str, Any]]:
    """The bundled pose tracks, newest contract first."""
    index = SAMPLES_DIR / "index.json"
    if index.is_file():
        return json.loads(index.read_text(encoding="utf-8"))
    return [
        {"file": p.name, "name": p.stem, "description": "", "duration_s": None}
        for p in sorted(SAMPLES_DIR.glob("*.json"))
        if p.name != "index.json"
    ]


def sample_path(name: str) -> Path:
    """Resolve a sample name to a bundled file, refusing anything else."""
    safe = Path(name).name
    if not safe.endswith(".json"):
        safe += ".json"
    path = SAMPLES_DIR / safe
    if not path.is_file() or path.name == "index.json":
        raise ServiceError("NOT_FOUND", f"no bundled sample {name!r}", known=[s["name"] for s in samples()])
    return path


def analyze(ctx: JobContext) -> RunRecord:
    """The analyzer servicekit calls on a worker thread, for either input kind."""
    ctx.progress(2, "reading the input")
    suffix = ctx.input_path.suffix.lower()
    try:
        if suffix == ".json":
            record, _ = analyse_track(ctx.input_path, sink=ctx, progress=ctx.progress)
        else:
            room = _room()
            ctx.note(f"using the room setup for {room.room}")
            record, _ = analyse_video(
                ctx.input_path, room, sink=ctx, progress=ctx.progress
            )
    except PrivacyViolation as exc:
        raise ServiceError("ANALYSIS_FAILED", f"privacy guard refused: {exc}") from exc
    except ValueError as exc:
        raise ServiceError("BAD_REQUEST", str(exc)) from exc
    ctx.progress(100, "done")
    return record


def _room() -> RoomConfig:
    if ROOM_JSON.is_file():
        return RoomConfig.load(ROOM_JSON)
    from .synth import default_room

    room, _ = default_room()
    return room


def build_config() -> ServiceConfig:
    return ServiceConfig(
        product=ProductInfo(
            slug=PRODUCT,
            title="Preempt",
            tagline=TAGLINE,
            description=DESCRIPTION,
            accent="#3B3390",
            version="1.0.0",
            repo_url="https://github.com/jo2980958-hub/preempt",
        ),
        allowed_suffixes=(".json", ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"),
        static_dir=WEB_DIR if WEB_DIR.is_dir() else None,
        models={"pose": RTMPOSE_T},
        max_concurrent_jobs=2,
        params_schema=[],
    )


def create() -> Any:
    app = create_app(build_config(), analyze)

    @app.get("/api/samples")
    async def list_samples() -> dict[str, Any]:
        return {"samples": samples()}

    @app.post("/api/samples/{name}", status_code=202)
    async def run_sample(request: Request, name: str) -> dict[str, Any]:
        path = sample_path(name)
        store = request.app.state.store
        job = store.create(path.name, path.read_bytes(), {"input_kind": "bundled sample"})
        store.start(job)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "events_url": f"/api/jobs/{job.job_id}/events",
        }

    return app
