# Preempt

**A call before the fall, from an abstracted pose and nothing else.**
An entry in the OpenCV AI Competition 2026.

Preempt watches a hospital or care-home room and raises a call *before* someone
falls. It recognises the movement that precedes standing up from a bed or chair
and calls while the person is still supported; it scores how steady they are once
they are on their feet; it checks what is in the way; and it says plainly when it
cannot see.

The camera frame is destroyed as soon as the pose has been read. The evidence a
nurse sees is a stick figure on a room plan, never a photograph. That is a tested
claim, not a slogan.

**Live: <https://2uhvgzwrwc.us-east-2.awsapprunner.com>**

| | |
|---|---|
| Median lead time before the person is upright, 10 bed exits | **4.60 s** (4.07 to 5.60) |
| Median lead time, all 15 bed and chair exits | **4.47 s** (0.73 to 5.60) |
| Exits called, synthetic decision layer | **25 of 25** |
| False alarms in 1.23 hours of observed quiet | **0** |
| Real CDC chair-stand footage, hand-set room | **3 of 3** stands called, 0 false calls |
| Real fall sequences reaching "on the floor" | **10 of 12**, median 0.50 s behind ground truth |
| Camera bytes written to disk | **0** |
| Tests | **124**, green |

Lead time is measured on synthetic sequences because no real dataset labels the
frame at which a person began to stand. On the real chair-stand footage, which
asks for the fastest stands a person can do, the three calls came 0.08, 0.42 and
0.41 s before upright.

## Start here

- **[products/preempt/README.md](products/preempt/README.md)** — what it is and how to run it
- [products/preempt/docs/report.md](products/preempt/docs/report.md) — the technical report
- [products/preempt/docs/architecture.md](products/preempt/docs/architecture.md) — the diagrams
- [products/preempt/docs/evaluation.md](products/preempt/docs/evaluation.md) — the numbers and the failures

## Layout

```
packages/visioncore     shared OpenCV 5 primitives: IO, calibration, timing, run records
packages/servicekit     shared FastAPI service shell and UI shell
products/preempt        this product: engine, service, tests, evaluation, docs
infra/                  ECR and App Runner deployment scripts
constraints.txt         the pinned transitive dependency set
```

The two `packages/` directories are a shared foundation built alongside four
sibling entries. They are included here so this repository builds on its own.

## Quick start

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python \
  -e packages/visioncore -e packages/servicekit -e products/preempt
products/preempt/models/fetch.sh
.venv/bin/python -m pytest products/preempt/tests -q
.venv/bin/python -m preempt.cli sample bed-exit-steady
.venv/bin/python -m uvicorn preempt.main:app --port 8000
```

## Licences

Code in this repository is released under the MIT licence (see `LICENSE`). The two models it runs are **Apache-2.0**
(RTMPose-t from OpenMMLab, YOLOX-tiny from Megvii) and are fetched rather than
committed. Nothing here imports `ultralytics`, whose AGPL-3.0 section 13 would
make a hosted demo a source-disclosure event.

The UR Fall Detection Dataset used for evaluation is **CC BY-NC-SA 4.0,
non-commercial academic use**. It is fetched at evaluation time and nothing
derived from it is committed here or appears in any document or video.

There is no face recognition in this product, and there must not be.
