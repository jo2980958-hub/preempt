"""Command line: run a sample, analyse a video, evaluate, benchmark, make samples.

    python -m preempt.cli sample bed-exit-steady
    python -m preempt.cli video ward.mp4 --room rooms/default.json
    python -m preempt.cli evaluate --seeds 5 --out eval/synthetic.json
    python -m preempt.cli urfall eval/data --room eval/urfall-room.json
    python -m preempt.cli bench
    python -m preempt.cli samples --out samples
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import RoomConfig


def _print_record(record, analysis) -> None:
    metrics = record.metrics
    print(f"  source            {record.input.get('source')}")
    print(f"  peak state        {metrics['peak_state']}  ({metrics['peak_headline']})")
    print(f"  lead time         {metrics['lead_time_s']} s")
    print(f"  calls             {[c['rung'] for c in analysis.calls]}")
    print(f"  view usable       {metrics['view']['usable_fraction'] * 100:.0f}% of frames")
    privacy = metrics["privacy"]
    print(
        f"  privacy           {privacy['camera_bytes_persisted']} camera bytes written, "
        f"{privacy['frames_retained']} frames retained, clean={privacy['clean']}"
    )
    peak = analysis.peak
    if peak:
        for reason in peak.risk.reasons:
            print(f"    - {reason}")
    for refusal in record.refusals:
        print(f"  refusal           {refusal.code}: {refusal.message}")


def cmd_sample(args: argparse.Namespace) -> int:
    from .pipeline import analyse_track
    from .service import sample_path

    path = Path(args.name) if Path(args.name).is_file() else sample_path(args.name)
    sink = Path(args.evidence) if args.evidence else None
    record, analysis = analyse_track(path, sink=sink)
    print(f"preempt: {path.name}")
    _print_record(record, analysis)
    if args.json:
        Path(args.json).write_text(record.to_json(), encoding="utf-8")
        print(f"  wrote             {args.json}")
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    from .pipeline import analyse_video

    room = RoomConfig.load(args.room) if args.room else _default_room()
    sink = Path(args.evidence) if args.evidence else None
    started = time.perf_counter()
    record, analysis = analyse_video(args.path, room, sink=sink)
    print(f"preempt: {args.path}")
    _print_record(record, analysis)
    print(f"  wall clock        {time.perf_counter() - started:.1f} s")
    for stage in record.stages:
        print(f"    {stage.name:24s} {stage.ms_per_call:7.2f} ms x {stage.calls}")
    if args.json:
        Path(args.json).write_text(record.to_json(), encoding="utf-8")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import run_synthetic

    report = run_synthetic(seeds=args.seeds, quiet_loops=args.quiet_loops)
    print(json.dumps(report.summary(), indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report.to_json(), encoding="utf-8")
        print(f"wrote {args.out}")
    failures = report.summary()["failures"]
    return 1 if args.strict and failures else 0


def cmd_urfall(args: argparse.Namespace) -> int:
    from .evaluate import run_urfall

    report = run_urfall(Path(args.root), room_json=Path(args.room), limit=args.limit)
    print(json.dumps(report.summary(), indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report.to_json(), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Recover a room setup from frames of somebody walking about in it."""
    import cv2

    from .calibrate import calibrate_from_people
    from .config import RoomConfig
    from .pose import PoseEstimator

    paths = sorted(Path(args.frames).glob("*.png")) + sorted(Path(args.frames).glob("*.jpg"))
    if not paths:
        print(f"no frames under {args.frames}")
        return 1
    estimator = PoseEstimator()
    poses = []
    size = None
    for index, path in enumerate(paths[:: args.stride]):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        size = (image.shape[1], image.shape[0])
        pose = estimator.estimate(image, index, index / 30.0)
        if pose is not None:
            poses.append(pose)
    if size is None:
        print("no frames decoded")
        return 1
    result = calibrate_from_people(
        poses, size, stature_m=args.stature, focal_px=args.focal
    )
    print(json.dumps(result.to_dict(), indent=2))
    if not result.ok:
        return 1
    room = RoomConfig(
        room=args.room_name,
        floor=result.plane,
        notes=(
            "Calibrated from people walking in the room rather than from measured "
            "floor points. " + "; ".join(result.assumptions)
        ),
    )
    room.save(args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Time the two models under each OpenCV 5 DNN engine, on this machine."""
    import cv2
    import numpy as np

    from .pose import RTMPOSE_T, PoseEstimator

    print(f"OpenCV {cv2.__version__}, {cv2.getNumThreads()} threads")
    image = np.random.default_rng(7).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    for engine in ("classic", "new"):
        estimator = PoseEstimator(engine=engine)
        box = (400.0, 120.0, 760.0, 690.0)
        for _ in range(3):
            estimator.keypoints(image, box)
        started = time.perf_counter()
        for _ in range(args.runs):
            estimator.keypoints(image, box)
        pose_ms = (time.perf_counter() - started) / args.runs * 1000.0
        for _ in range(2):
            estimator.largest_person(image)
        started = time.perf_counter()
        for _ in range(max(3, args.runs // 4)):
            estimator.largest_person(image)
        detect_ms = (time.perf_counter() - started) / max(3, args.runs // 4) * 1000.0
        print(
            f"  ENGINE_{engine.upper():8s} "
            f"{RTMPOSE_T.name} {pose_ms:6.2f} ms   yolox-tiny {detect_ms:6.2f} ms"
        )
    return 0


def cmd_samples(args: argparse.Namespace) -> int:
    from .synth import SCENARIOS, make

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for name in args.names or sorted(SCENARIOS):
        sequence = make(name)
        sequence.save(out / f"{name}.json")
        index.append(
            {
                "file": f"{name}.json",
                "name": name,
                "description": sequence.description,
                "duration_s": round(sequence.duration_s, 1),
            }
        )
        print(f"  {name:24s} {sequence.duration_s:5.1f} s")
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return 0


def _default_room() -> RoomConfig:
    from .synth import default_room

    room, _ = default_room()
    return room


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="preempt", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample", help="analyse a bundled pose track")
    p.add_argument("name")
    p.add_argument("--evidence", help="directory to write the evidence cards into")
    p.add_argument("--json", help="write the run record here")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("video", help="analyse a video through YOLOX and RTMPose")
    p.add_argument("path")
    p.add_argument("--room", help="room setup JSON")
    p.add_argument("--evidence")
    p.add_argument("--json")
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("evaluate", help="the synthetic evaluation")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--quiet-loops", type=int, default=12)
    p.add_argument("--out")
    p.add_argument("--strict", action="store_true", help="exit non-zero on any failure")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("urfall", help="evaluate on the UR Fall Detection Dataset")
    p.add_argument("root")
    p.add_argument("--room", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--out")
    p.set_defaults(func=cmd_urfall)

    p = sub.add_parser("calibrate", help="recover a room setup from a walking person")
    p.add_argument("frames", help="a directory of frames")
    p.add_argument("--out", required=True)
    p.add_argument("--room-name", default="room")
    p.add_argument("--stature", type=float, default=1.75)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--focal", type=float, help="known focal length in pixels")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("bench", help="time the models under each OpenCV 5 DNN engine")
    p.add_argument("--runs", type=int, default=40)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("samples", help="regenerate the bundled pose tracks")
    p.add_argument("--out", default="samples")
    p.add_argument("--names", nargs="*")
    p.set_defaults(func=cmd_samples)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
