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

from fastapi import File, Form, Request, UploadFile
from fastapi.routing import APIRoute
from servicekit import JobContext, ProductInfo, ServiceConfig, ServiceError, create_app
from visioncore import RunRecord

from .config import RoomConfig, RoomError, parse_room
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
        raise ServiceError(
            "NOT_FOUND", f"no bundled sample {name!r}", known=[s["name"] for s in samples()]
        )
    return path


def analyze(ctx: JobContext) -> RunRecord:
    """The analyzer servicekit calls on a worker thread, for either input kind."""
    ctx.progress(2, "reading the input")
    suffix = ctx.input_path.suffix.lower()
    try:
        if suffix == ".json":
            record, _ = analyse_track(ctx.input_path, sink=ctx, progress=ctx.progress)
        else:
            room, source = job_room(ctx.params)
            ctx.note(f"measuring against the room setup {room.room!r} ({source})")
            record, _ = analyse_video(
                ctx.input_path, room, sink=ctx, progress=ctx.progress, room_source=source
            )
    except PrivacyViolation as exc:
        raise ServiceError("ANALYSIS_FAILED", f"privacy guard refused: {exc}") from exc
    except ValueError as exc:
        raise ServiceError("BAD_REQUEST", str(exc)) from exc
    ctx.progress(100, "done")
    return record


def job_room(params: dict[str, Any]) -> tuple[RoomConfig, str]:
    """The room a job brought with it, or the image's default, and which it was."""
    if params.get("room") is not None:
        return parse_room(params["room"]), "uploaded"
    return _room(), "default"


def _room() -> RoomConfig:
    if ROOM_JSON.is_file():
        return RoomConfig.load(ROOM_JSON)
    from .synth import default_room

    room, _ = default_room()
    return room


ROOM_MAX_BYTES = 256 * 1024


def checked_room(data: Any) -> RoomConfig:
    """Validate an uploaded room with the CLI's own parser, as a clean 400."""
    if isinstance(data, (bytes, str)):
        if len(data) > ROOM_MAX_BYTES:
            raise ServiceError("BAD_REQUEST", "the room setup is larger than 256 KB", field="room")
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ServiceError(
                "BAD_REQUEST", f"the room setup is not valid JSON: {exc}", field="room"
            ) from exc
    try:
        room = parse_room(data)
    except RoomError as exc:
        raise ServiceError(
            "BAD_REQUEST", f"invalid room setup, {exc.field}: {exc.problem}", field=exc.field
        ) from exc
    if room.privacy_mode != "strict":
        raise ServiceError(
            "BAD_REQUEST",
            "invalid room setup, privacy_mode: this service only runs strict rooms, "
            "which destroy every frame after the pose is read",
            field="privacy_mode",
        )
    return room


def room_summary(room: RoomConfig, source: str) -> dict[str, Any]:
    """What the UI and the result say about the room a run was measured against."""
    return {
        "name": room.room,
        "source": source,
        "calibration": room.calibration.to_dict(),
        "zones": len(room.zones),
        "notes": room.notes,
    }


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
    config = build_config()
    app = create_app(config, analyze)
    _replace_job_route(app, config)

    @app.get("/api/rooms/default")
    async def default_room_setup() -> dict[str, Any]:
        room = _room()
        return {"room": room.to_dict(), "summary": room_summary(room, "default")}

    @app.post("/api/rooms/check")
    async def check_room(request: Request) -> dict[str, Any]:
        room = checked_room(await request.body())
        return {"room": room.to_dict(), "summary": room_summary(room, "uploaded")}

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


def _replace_job_route(app: Any, config: ServiceConfig) -> None:
    """Swap servicekit's upload route for one that also takes a room setup.

    servicekit is shared by five products and is not ours to change, so the
    shared `POST /api/jobs` is removed from this app's router and an equivalent
    one registered in its place. It accepts everything the shared route did, plus
    a room: either `room` inside the `params` JSON, or a second file part named
    `room`. The room is validated here, before a job exists, so a bad one is a
    400 naming the field rather than a job that fails a minute later.
    """
    app.router.routes[:] = [
        route
        for route in app.router.routes
        if not (
            isinstance(route, APIRoute) and route.path == "/api/jobs" and "POST" in route.methods
        )
    ]

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        request: Request,
        file: UploadFile = File(...),
        params: str = Form("{}"),
        room: UploadFile | None = File(None),
    ) -> dict[str, Any]:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in config.allowed_suffixes:
            raise ServiceError(
                "UNSUPPORTED_MEDIA",
                f"{suffix or 'that file type'} is not accepted",
                accepted=list(config.allowed_suffixes),
            )
        try:
            parsed = json.loads(params) if params else {}
        except json.JSONDecodeError as exc:
            raise ServiceError("BAD_REQUEST", f"params is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ServiceError("BAD_REQUEST", "params must be a JSON object")

        room_data: Any = parsed.pop("room", None)
        if room is not None:
            if room_data is not None:
                raise ServiceError(
                    "BAD_REQUEST",
                    "send the room setup once: as params.room or as the room file, not both",
                    field="room",
                )
            room_data = await room.read(ROOM_MAX_BYTES + 1)
        if room_data is not None:
            if suffix == ".json":
                raise ServiceError(
                    "BAD_REQUEST",
                    "a pose track carries the room it was recorded in; a room setup "
                    "can only be sent with a video",
                    field="room",
                )
            checked = checked_room(room_data)
            parsed["room"] = checked.to_dict()

        data = await _read_upload(file, config)
        store = request.app.state.store
        job = store.create(file.filename or "upload", data, parsed)
        store.start(job)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "events_url": f"/api/jobs/{job.job_id}/events",
        }


async def _read_upload(file: UploadFile, config: ServiceConfig) -> bytes:
    """Read an upload under the size limit. Mirrors servicekit's own reader."""
    chunks, total = [], 0
    while chunk := await file.read(1 << 20):
        total += len(chunk)
        if total > config.max_upload_bytes:
            raise ServiceError(
                "TOO_LARGE",
                f"upload exceeds {config.max_upload_bytes // (1024 * 1024)} MB",
                max_bytes=config.max_upload_bytes,
            )
        chunks.append(chunk)
    if total == 0:
        raise ServiceError("BAD_REQUEST", "the uploaded file is empty")
    return b"".join(chunks)
