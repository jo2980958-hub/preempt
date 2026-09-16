# Preempt

**A call before the fall, from an abstracted pose and nothing else.**

Preempt watches a hospital or care-home room and raises a call *before* someone
falls, not after. It recognises the movement that precedes standing up from a bed
or a chair and calls while the person is still supported. It scores how steady
they are once they are on their feet. It checks what is in the way. And it says
plainly when it cannot see.

It is privacy-first by construction: the camera frame is destroyed as soon as the
pose has been read, so the evidence a nurse sees is a stick figure on a room plan
and never a photograph. That is a testable claim, not a slogan, and
`tests/test_privacy.py` tests it.

**Live: <https://2uhvgzwrwc.us-east-2.awsapprunner.com>**

| | |
|---|---|
| Median lead time before the person is upright, 10 bed exits | **4.60 s** (4.07 to 5.60) |
| Median lead time, all 15 bed and chair exits | **4.47 s** (0.73 to 5.60) |
| Detection rate, synthetic decision layer | **100 %** (25 of 25) |
| False alarms in 1.23 hours of observed quiet | **0** |
| Real CDC chair-stand footage, hand-set room | **3 of 3** stands called, 0 false calls (was 3) |
| Fall sequences reaching "on the floor", UR Fall | **10 of 12**, median 0.50 s behind ground truth |
| Camera bytes written to disk | **0** |
| Tests | **124**, green |

Numbers, method and the failures in [docs/evaluation.md](docs/evaluation.md),
including the two flaws that real footage found and what fixing them changed.

---

## What it does

1. **Bed and chair exit.** Standing up is four phases and the first two happen
   while the person is still supported. A pressure pad under a mattress cannot
   know anything until the weight has gone. A camera watching the upper body
   arrive over the feet can see it several seconds earlier, and that gap is the
   product.
2. **Unsteady gait.** Sway about the fitted walking line and a hand going out to a
   wall, scored over four seconds of walking rather than read off one frame.
3. **Hazards on the route.** A walking frame out of reach, clutter between the bed
   and the door, a wet-floor sign on the route. All classical OpenCV, all in
   metres on the floor plane.
4. **Escalation.** A quiet prompt in the room, then a call to the station, then an
   urgent call when somebody is on the floor. A prompt nobody answers becomes a
   station call by itself.
5. **Honesty rails.** Too dark, lens blocked, camera knocked, nobody in view. Each
   is named, each carries the remedy, and none of them produce a clinical call.

---

## Running it

```bash
# from the repository root
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python \
  -e packages/visioncore -e packages/servicekit -e products/preempt

products/preempt/models/fetch.sh        # 33 MB of Apache-2.0 ONNX, not committed

# a bundled pose track, which is all a Preempt device ever keeps
.venv/bin/python -m preempt.cli sample bed-exit-steady --evidence /tmp/preempt

# a video, through YOLOX and RTMPose
.venv/bin/python -m preempt.cli video ward.mp4 --room products/preempt/rooms/default.json

# the web service
.venv/bin/python -m uvicorn preempt.main:app --port 8000
```

A video is only as good as the room it is measured against. Upload one with its
room setup, either as a second file or inside the job params:

```bash
curl -F file=@ward.mp4 -F room=@my-room.json https://2uhvgzwrwc.us-east-2.awsapprunner.com/api/jobs
curl -F file=@ward.mp4 -F 'params={"room": {...}}' .../api/jobs
```

The room is checked by the same parser as `--room`, and a bad one is a 400 that
names the field. Without one the synthetic ward's room is used, and the result
says so. `POST /api/rooms/check` validates a room on its own. In the browser, a
video stops at a preview that draws the room over its first frame before
anything is sent.

Other commands: `evaluate`, `urfall`, `calibrate`, `bench`, `samples`.
`python -m preempt.cli --help`.

## Testing

```bash
.venv/bin/python -m pytest products/preempt/tests -q
```

124 tests. The privacy tests run the whole pipeline with the real models over a
real video file and assert that nothing derived from those pixels was written.
The pose tests are skipped, loudly, if `models/fetch.sh` has not been run.
`tests/test_real_footage.py` replays keypoints read from a public-domain CDC
chair-stand clip (no pixels are in the repository) and fails on the engine
before the sit-down fix.

## Deploying

```bash
infra/ecr.sh preempt --context . --dockerfile products/preempt/Dockerfile
AWS_REGION=us-east-2 infra/apprunner.sh preempt --cpu 2 --memory 4
```

Resources, rates and the two decisions worth recording in
[docs/costs.md](docs/costs.md).

---

## Pinned dependencies

`opencv-python-headless==5.0.0.93`, `numpy==2.5.3`, `fastapi==0.141.1`,
`uvicorn==0.53.0`, `pydantic==2.13.5`, `python-multipart==0.0.32`. The full
transitive set is in `constraints.txt` at the repository root.

**The OpenCV pin is load-bearing.** `pip install opencv-python` unpinned resolves
to 4.14.x, which shipped after 5.0.0. `import visioncore` raises on a 4.x wheel,
the Docker build asserts the major version, and `/version` prints what is running.

### Models

| Model | Licence | Source |
|---|---|---|
| RTMPose-t (body7, 256x192) | Apache-2.0 | OpenMMLab MMPose |
| YOLOX-tiny (416x416) | Apache-2.0 | Megvii |

**No AGPL.** Ultralytics' pose models are AGPL-3.0, whose section 13 makes a
hosted demo a source-disclosure event. Nothing here imports `ultralytics` and
nothing may.

**No face recognition, ever, in this product.** There is no face model, no
embedding and no identity of any kind.

### Evaluation data

The UR Fall Detection Dataset is **CC BY-NC-SA 4.0, non-commercial academic use**.
It is fetched at evaluation time by `eval/fetch_urfall.sh` and nothing derived
from it is committed here or appears in any document, deck or video.

---

## Documentation

- [docs/report.md](docs/report.md) — the technical report: problem, users,
  architecture, the OpenCV 5 implementation, AWS, evaluation, limitations, and the
  consent and dignity questions a ward would ask
- [docs/architecture.md](docs/architecture.md) — pipeline, geometry and AWS diagrams
- [docs/evaluation.md](docs/evaluation.md) — the numbers and the failures
- [docs/costs.md](docs/costs.md) — what was created on AWS and what it costs
- [docs/devpost.md](docs/devpost.md) — submission text
- [docs/narration.md](docs/narration.md) — the video script
