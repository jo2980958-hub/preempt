# Evaluation

Four numbers matter for a product that claims to call before a fall: how many
seconds of warning it gives, how often it notices, how often it calls when
nothing is happening, and what it does when it is wrong. All four are below, with
the failures named.

One sentence before any of them, because it is the honest frame for everything
that follows: **no deployed-system outcome trial exists for this class of
product.** We searched for one across camera-based safety products in several
domains and could not find one. If a judge asks whether this has been shown to
save lives on a real ward, the answer is that nobody has published that evidence,
for this product or for any comparable one.

Reproduce everything here with:

```bash
python -m preempt.cli evaluate --seeds 5 --quiet-loops 16 --out eval/synthetic.json
eval/fetch_urfall.sh 12
python -m preempt.cli urfall eval/data --room eval/urfall-room.json --out eval/urfall.json
python -m preempt.cli bench --runs 40
```

---

## 1. Two tracks, and what each one can prove

**The synthetic track** measures the decision layer. Nine scenarios are generated
by projecting a seventeen-joint body model through a pinhole camera whose floor
homography, vertical vanishing point and reference height the engine is then
given, so the geometry is exact and any error is the engine's. Keypoints get
Gaussian noise at roughly the level a real pose estimator produces, joints drop
out at random, and scores vary, because a synthetic sequence with clean keypoints
would flatter every threshold in the product.

It can prove: given pose, does the state machine call at the right moment, how
many seconds ahead, and how often does it call when it should not. The decisive
property is that **the answer is written into the sequence**: the frame at which
the person began to stand is known, so lead time is measurable. No real dataset
carries that label.

It cannot prove anything about pose estimation on a real patient in a real bed
under a real blanket.

**The UR Fall track** measures the whole pipeline on real footage: real people,
real rooms, real falls, with per-frame ground truth of whether the person is on
the floor. It is the honest counterweight, and it is where the product's real
limitation shows up.

### The dataset, and its licence

UR Fall Detection Dataset, Kwolek and Kepski, University of Rzeszów.
<http://fenix.ur.edu.pl/~mkepski/ds/uf.html>. Quoted from the dataset page:

> "This work is licensed under a Creative Commons
> Attribution-NonCommercial-ShareAlike 4.0 International License and is intended
> for non-commercial academic use."

Non-commercial, and ShareAlike. So: it is fetched at evaluation time by
`eval/fetch_urfall.sh`, **no frame and no image derived from it is committed to
this repository**, and none appears in the report, the deck or the video. That is
a deliberate constraint on what we may publish, not an oversight.

Twenty-one sequences were used: twelve labelled falls and nine activity
sequences, cam0 RGB at 640x480.

---

## 2. Lead time, which is the whole product

Measured from the first call going out to the person being upright on their feet
and staying upright for half a second. Bed and chair exits only, because those are
the sequences in which being upright is a thing that happens.

| | seconds |
|---|---|
| median | **4.60** |
| mean | 4.69 |
| range | 4.07 to 5.60 |
| sequences | 10 |

For comparison, the quantity a fall-detection system reports is negative by
definition: it fires after the event.

The lead comes from where the call is raised. Standing up from a seat is four
phases, and the first two happen while the person is still supported: flexion
momentum, momentum transfer, extension, stabilisation. A pressure pad under a
mattress cannot know anything until the weight has gone, which is the end of
phase two. A camera watching the upper body arrive over the feet can see phase
one. That gap is the product, and 4.6 seconds is how wide it measured.

## 3. Detection rate and false alarms, synthetic track

Nine scenarios at five noise seeds each, with the quiet scenarios looped sixteen
times so that the false-alarm denominator is worth dividing by.

| | |
|---|---|
| sequences | 45 |
| should have called | 20 |
| called | 20 (**100 per cent**) |
| should not have called | 25 |
| called anyway | **0** |
| quiet time observed | 1.18 hours |
| false alarms per bed-night (12 h) | **0.0** |
| failures | none |

**Read the false-alarm figure with its denominator.** 1.18 hours of quiet
extrapolated to a twelve-hour night is an assumption, not a measurement. One
stray call in that window would have scaled to about ten a night. The number is
zero because nothing fired, not because the observation was long.

The quiet scenarios are the ones a ward would worry about: a patient asleep who
turns over twice, a patient who sits up in bed to drink and lies back down, a
visitor shifting in the chair, a member of staff walking across the room.

## 4. The whole pipeline on real footage

Twenty-one UR Fall sequences, pose estimation included, against the dataset's own
per-frame label for "lying on the ground".

| | |
|---|---|
| fall sequences reaching "on the floor" | **10 of 12** |
| activity sequences that do contain lying frames, detected | 2 of 2 |
| activity sequences with no lying frames that said "on the floor" | **2 of 10** |
| median delay behind the ground-truth frame | **0.50 s** |
| delay range | 0.13 to 0.97 s |

Half a second behind the labelled frame is the `floor_hold_s` setting doing its
job: the state has to hold before an urgent call goes out, because a person who
sat down heavily for a third of a second is not a person on the floor.

### The three failures, named

**fall-01 and fall-03 were missed.** The person is detected, but pose collapses
as they land and the state never holds long enough.

**adl-01 and adl-02 falsely said "on the floor".** Both are sequences where the
subject bends down or crouches. Geometrically that is close to lying: the head
travels forward and down, and back-projects onto the floor plane about where a
lying person's would. The bound that separates them is narrow and it is stated
explicitly in `Thresholds.floor_body_max_m`.

## 5. The binding limitation, measured

The decision layer is not the weak part. Pose estimation on a person who is
already on the floor is. Over 561 ground-truth lying frames across the twelve
fall sequences:

| | frames | share |
|---|---|---|
| a person was detected at all | 519 | 93 % |
| six or more usable joints came back | 383 | **68 %** |
| the body back-projected onto the floor plane consistently | 260 | 46 % |
| the head landed a body length from the feet | 336 | 60 % |

A third of the frames that matter most produce too little pose to reason about.
The cause is that a detector trained mostly on upright people sees a person lying
down as a wide, short, unfamiliar blob, and a top-down pose estimator given a bad
box returns noise.

Two things were done about it and both are in the code. The estimator **coasts**:
when the detector loses the person it re-runs pose on the last box widened by a
fifth, for up to six frames, and drops the result if the keypoint scores come back
near zero. And the floor state has a **grace period**, so a hold that lapses for
under 0.4 seconds does not restart, because with a 68 per cent per-frame yield a
strict hold never completes.

Neither fixes it. What would fix it is a detector or a pose model trained with
supine and prone people in it, and that is the first thing we would change with
more time.

## 6. Gait, and a measure that was removed

Gait is scored over a four-second window from two things: lateral sway about the
fitted walking line, and a hand held out to a wall.

Sway separates cleanly on the synthetic walks: 2.3 to 3.8 cm RMS for a steady
walk against 10.8 to 12.7 cm for an unsteady one.

**Step-time variability was in the score and was taken out.** The coefficient of
variation of step intervals is the standard clinical gait-variability statistic
and it was the obvious thing to include. At the fifteen hertz this pipeline
samples at, a 1.9 Hz cadence gives about eight samples per step and the peak of
the smoothed foot-separation signal lands a sample either side at random. Measured
on synthetic walks that are steady by construction, the CV came out between 0.26
and 0.52, which is the same magnitude as the clinical effect it was meant to
detect, and it produced two false alarms in twenty quiet sequences on its own.

It is still computed and still shown to a clinician. It contributes nothing to
the score, and `Thresholds.step_time_cv_warn` says so where somebody tuning the
product will read it.

## 7. Speed, and the OpenCV 5 engine

`python -m preempt.cli bench --runs 40`, x86, 22 threads, `opencv-python==5.0.0.93`,
on a machine that was running four other builds at the time, so treat the absolute
numbers as an upper bound and the ratios as the result.

| stage | ENGINE_CLASSIC | ENGINE_NEW | speedup |
|---|---|---|---|
| RTMPose-t 256x192, crop and normalise included | 9.30 ms | 7.83 ms | 1.19x |
| YOLOX-tiny 416x416, letterbox included | 44.01 ms | 27.46 ms | 1.60x |

On an unloaded machine the same script gave 6.46 / 5.37 ms and 30.01 / 18.10 ms,
and the bare `net.forward()` for RTMPose was 5.3 ms classic against 3.1 ms new,
a 1.7x ratio. OpenCV's new graph engine is the default and it is faster on the
same weights at the same call site; the two engines agree on the keypoints to
within a pixel, which `tests/test_pose.py` asserts, so the speed is free.

At the deployed sample rate of 15 Hz the pipeline runs comfortably faster than
real time on two vCPU.

## 8. Geometry, checked against closed-form truth

`tests/test_geometry.py` checks the floor frame against a camera whose answers are
known exactly.

- The four setup points round-trip to under 0.01 px.
- A floor point back-projects to within 0.1 mm of where it came from.
- Heights above the floor are recovered to **under 1 cm** across the room, at
  heights from 0.15 m to 1.7 m.
- The floor shadow of a point at a known height is exact to 1 mm.
- The projective vertical matches the true image direction of "up" to 0.06
  degrees, and differs from the naive image y axis by up to **31.7 degrees**
  across this room. Taking the image vertical for "up" is fine near the middle of
  the frame and puts thirty degrees of error into a trunk angle at the corner.

The auto-calibration in `preempt.calibrate` recovers, from a person walking about
a synthetic room, a focal length of 752.8 px against a true 760 and a camera
height of 2.52 m against a true 2.55, and heights across the room to within 1 cm.

## 9. Privacy, asserted rather than claimed

`tests/test_privacy.py` runs the whole pipeline with the real models over a real
video file whose every frame is unique high-entropy content, and then asserts:

- `camera_bytes_persisted == 0` and `frames_retained == 0`;
- every file the run wrote, decoded and correlated against every source frame,
  correlates below 0.35;
- asking the guard to write a camera frame raises `PrivacyViolation` and no file
  appears;
- `release()` leaves the frame buffer zeroed;
- diagnostic mode raises unless an environment variable is set that the deployed
  service never sets;
- and, mechanically, no function in `render.py` accepts a parameter named `image`,
  checked by reflection, so there is no code path from a camera frame to a saved
  file.

## 10. What these numbers are not

They are not a clinical result. They are not evidence that fewer people fall.
Nearly half of inpatient falls are a knee buckling, and 40 per cent of the falls
in one rehabilitation study happened during gait training, which is while a
therapist was already present and holding on. A camera does not prevent a knee
giving way, and this product does not claim to.

What it claims is narrower and is what was measured: that the movement which
precedes standing is visible several seconds before the weight leaves the bed,
that it can be told apart from turning over and from lying back down, and that
someone on the floor can be recognised geometrically rather than guessed.
