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

# the real-footage regression: CDC chair-stand keypoints, no pixels
python -m pytest tests/test_real_footage.py
# the clips themselves (public domain, not in the repository; see section 5)
python -m preempt.cli video cdc-chair-stand-oblique.mp4 --room cdc-oblique-hand-room-chairzone.json
```

---

## 1. Three tracks, and what each one can prove

**The synthetic track** measures the decision layer. Twelve scenarios are generated
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

**The CDC chair-stand clips** are the third track, added after the first two were
published. They are real footage of the movement Preempt is built to call:
an older woman standing up from a chair and sitting back down, filmed by the US
Centers for Disease Control for its STEADI falls-prevention training, public
domain. They have no per-frame labels, so they cannot give a lead-time
distribution, but they found two flaws that neither of the other tracks could.
Section 5 has them.

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

| | all exits | bed exits | quick chair stands |
|---|---|---|---|
| sequences | 15 | 10 | 5 |
| median | **4.47 s** | **4.60 s** | 1.00 s |
| mean | 3.44 s | 4.69 s | 0.93 s |
| range | 0.73 to 5.60 s | 4.07 to 5.60 s | 0.73 to 1.00 s |

The first published version had only the ten bed exits, and its headline was
4.60 s. The five chair stands came in with the sit-down fix in section 5. They
are a patient standing straight up out of a chair with a 1.2 second lean, and
they give about a second of warning. A fast stand leaves little time, and the
median over all exits is lower because of that. No lead time moved with the fix:
all fifteen are the same to the hundredth of a second before and after it.

For comparison, the quantity a fall-detection system reports is negative by
definition: it fires after the event.

The lead comes from where the call is raised. Standing up from a seat is four
phases, and the first two happen while the person is still supported: flexion
momentum, momentum transfer, extension, stabilisation. A pressure pad under a
mattress cannot know anything until the weight has gone, which is the end of
phase two. A camera watching the upper body arrive over the feet can see phase
one. That gap is the product. For a slow bed exit it measured 4.6 seconds wide,
and for a quick stand out of a chair about one second.

## 3. Detection rate and false alarms, synthetic track

Twelve scenarios at five noise seeds each, with the quiet scenarios looped sixteen
times so that the false-alarm denominator is worth dividing by.

| | |
|---|---|
| sequences | 60 |
| should have called | 25 |
| called | 25 (**100 per cent**) |
| should not have called | 35 |
| called anyway | **0** |
| calls raised as someone sat down | **0** |
| quiet time observed | 1.23 hours |
| false alarms per bed-night (12 h) | **0.0** |
| failures | none |

**Read the false-alarm figure with its denominator.** 1.23 hours of quiet
extrapolated to a twelve-hour night is an assumption, not a measurement. One
stray call in that window would have scaled to about ten a night, and before the
sit-down fix one did (section 5). The number is zero because nothing fired, not
because the observation was long.

The quiet scenarios are the ones a ward would worry about: a patient asleep who
turns over twice, a patient who sits up in bed to drink and lies back down, a
visitor shifting in the chair, somebody walking to the chair and sitting down in
it, and a member of staff walking across the room.

A sequence where the person stands up and then sits straight back down holds one
event to call and one not to call. Graded per sequence, a call on the sit-down
would hide behind the correct call on the rise. So the harness counts calls
raised after `sit_down_starts` separately, and any such call fails the sequence.
That is how a false call on every sit-down went unnoticed until real footage
showed it.

### The cost of privacy, measured rather than assumed

One of the ten scenarios exists only to make a cost visible. A ward will pause the
camera for washing and dressing, and it should: those are the activities where
camera acceptance is lowest. They are also, from section 1, among the activities
where a large share of falls happen.

So `personal-care-pause` runs a sequence in which staff pause the camera for nine
seconds. The product does not show a calm green light through it. It reports

```
"blind_by_choice_s": 9.0,  "faulty_s": 0.0
```

and refuses with `BLIND_BY_CHOICE`, whose message says that the window is
**unknowable, not absent**. No call is raised, not even a maintenance notice,
because staff pressed the button and do not need telling.

Two tests assert it, and the number belongs in a ward report next to the call
count: a system that was blind for three hours of a shift has a detection rate
that means something different from one that watched all night. Calls missed in
those periods cannot be counted, which means no evaluation of this product,
including this one, can put a number on them.

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

These were re-run after the fixes in section 5. All 21 outcomes, delays and
call counts are identical to the engine before them.

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

## 5. Real footage of standing up: the CDC chair-stand clips

Three single-shot cuts from two CDC STEADI training videos on Wikimedia Commons,
both public domain (`{{PD-USGov-HHS-CDC}}`): 12.4 s and 12.8 s of the *30-Second
Chair Stand Test* from the front and from an oblique angle, and 17.2 s of the
*Timed Up and Go Test*. None is in this repository. Only the keypoints Preempt
read from the oblique cut are committed, in `tests/data/`, and that file contains
no pixels. The people in the clips are demonstrators in a public training video,
not patients.

Nobody measured these rooms. Preempt needs four floor points and a vertical
reference, and auto-calibration refused both fixed shots, correctly, because
nobody walks toward or away from the camera in them. So the rooms were set up by
hand: a level camera, a focal length of 1035 px from a partial calibration of
the front shot, and a camera height of 0.89 m (front) or 0.81 m (oblique)
found by assuming the standing clinician is 1.65 m tall. **These are
assumptions.** Every room file says so in its `calibration` block, and the
interface and evidence cards now print it (flaw 1 below).

Stands were counted from the video: three in each chair-stand cut and one in the
TUG cut. On the oblique cut the moment each stand became upright was read from
the keypoints (knee angle first past 150 degrees: 1.08, 5.17 and 9.33 s), not
from Preempt's own states.

### Before and after the two fixes

Local CLI, the same rooms as the first runs, OpenCV 5.0.0. A hit is a clinical
call (nudge, station or urgent) raised during a stand, before the person is
upright. A false call is one raised anywhere else.

| Clip | Room | Stands | Calls | Hits | Misses | False calls | Lead times |
|---|---|---|---|---|---|---|---|
| front | default (synthetic ward) | 3 | 0 → 0 | 0 → 0 | 3 → 3 | 0 → 0 | — |
| front | hand setup, no zones | 3 | 0 → 0 | 0 → 0 | 3 → 3 | 0 → 0 | — |
| front | hand setup, chair zone | 3 | 0 → 0 | 0 → 0 | 3 → 3 | 0 → 0 | — |
| oblique | default (synthetic ward) | 3 | 0 → 0 | 0 → 0 | 3 → 3 | 0 → 0 | — |
| oblique | hand setup, no zones | 3 | 1 → 1 | 1 → 1 | 2 → 2 | 0 → 0 | 0.08 s → 0.08 s |
| oblique | hand setup, chair zone | 3 | **6 → 3** | 3 → 3 | 0 → 0 | **3 → 0** | 0.08, 0.42, 0.41 s, unchanged |
| TUG | default (synthetic ward) | 1 | 0 → 0 | 0 → 0 | 1 → 1 | 0 → 0 | — |

The TUG cut produced two `maintenance` notices and a `VIEW_UNUSABLE` refusal
before and after, which is right: the camera pans to follow her, and 22 per cent
of frames were marked "camera moved".

### Flaw 1: the service could not take anyone else's room

The live service had one room built in, the synthetic ward's, and an upload could
not bring its own. Every real clip was measured against the wrong floor. On the
oblique cut the woman's feet fell inside the default room's *bed* zone, so every
stand read as "supported" and all three were missed. The CLI already took
`--room`, and the first runs above used it.

**What changed.** An upload can now carry its room, as `room` inside the job
params or as a second file. It is checked by the same parser the CLI uses, and a
bad room is a 400 that names the field (`floor.image_points: needs exactly 4
points, got 3`). The result records the room it used, whether that room came
with the upload or is the default, and whether its camera height and focal
length were measured, assumed or synthetic. Both evidence cards print the same
line. In the interface a video no longer runs when you pick it. It stops at a
preview that draws the room's floor points and zones over the first frame. The
browser decodes that frame from the local file, and nothing is sent until you
press Watch. With the default room on the oblique clip, the preview shows the
bed zone lying across the chair and her feet, and warns that 5 of 24 setup points
fall outside the frame, before any analysis runs.

The oblique hand-setup room, uploaded to the redeployed service with this clip,
gives the same three calls at the same times as the CLI. The next subsection
has the live result.

### Flaw 2: sitting down raised a call to get up

With the chair-zone room, all three stands were called. Three more `nudge` calls
fired at 2.83, 7.00 and 11.00 s, each one as she sat back down. The rule behind
them is "the shoulders have arrived over the feet": to lower yourself onto a
seat you lean forward over your feet with your knees bent, and from one camera
that looks the same as the lean before standing up. A threshold cannot separate
them, because the two postures really are alike.

**What changed.** Direction of travel. A sit-down is the hips coming *down* from
standing: hip height falling at least as fast as a rise goes up (0.25 m/s,
the existing `rise_hip_speed_mps`, fitted over half a second), or the knees
closing at 90 degrees a second or more when there is no metric height, while
the person was upright within the detector's two-second history. While that
holds, the lean it produces is thrown away, and evidence for a rise has to be
gathered again from after the person has sat. A rise starts with the hips seated
and still, or going up, so its evidence is unaffected. The reading says
"sitting down: the hips came down from standing at 0.57 m/s", so the timeline
shows why nothing was called.

**The test that guards it** replays the oblique keypoints with the hand room. It
failed on the old code with exactly `[2.833, 7.0, 11.0]` and passes on the new
code, where all three stands are still called at their old times.

**Overfitting check.** The fix was measured on the full synthetic evaluation
before and after, and two synthetic chair scenarios were added. One is a walk to
the chair and a sit-down. The other is a stand out of the chair and straight back
down. Their body model puts the feet under the knees and the shoulders over the
feet on the way down, as in the footage.

| synthetic evaluation | before | after |
|---|---|---|
| published set, 50 sequences: detection | 100 % (20 of 20) | 100 % (20 of 20) |
| published set: false alarms, 1.21 h quiet | 0 | 0 |
| published set: median lead, 10 bed exits | 4.60 s (4.07 to 5.60) | 4.60 s (4.07 to 5.60) |
| with the chair scenarios, 60 sequences: detection | 100 % (25 of 25) | 100 % (25 of 25) |
| with the chair scenarios: false alarms, 1.23 h quiet | **1** (9.8 per bed-night) | **0** |
| with the chair scenarios: calls raised on a sit-down | **3** in 10 sequences | **0** |
| with the chair scenarios: median lead, 15 exits | 4.47 s (0.73 to 5.60) | 4.47 s (0.73 to 5.60) |

The fix cost nothing measurable on either track. No detection was lost and no
lead time changed. The old engine also failed on 3 of the 10 new synthetic
sit-down sequences, so the flaw was not something only this one clip could
trigger.

### A third finding: standing still reported as walking

For most of each stand on the oblique cut, the woman stood still and Preempt said
`walking` (36 of 40 upright readings). **The cause was geometry, not movement.**
The torso's floor position is the hips' shadow: slide down the vertical from the
hips by their height. With the camera at about hip height (0.81 m, hips at
0.86 m), the ray to the hips is nearly level and that construction is singular.
One pixel of keypoint noise moved the torso tens of metres. It was placed 25 m
away and back between frames, and the Kalman filter read that as 16 m/s. The
synthetic ward camera, 2.55 m up in a corner, never exposed this. There one pixel
moves the shadow 5 to 14 mm. On the CDC cameras it moves it 0.05 to 0.1 m with the
person seated, and from 0.2 m to over a kilometre with them standing.

**What changed.** The conditioning is now measured on every frame. If one pixel
of hip error would move the torso more than 3 cm (about 12 cm at RTMPose's
roughly 4 px of noise), the feet are used instead, because they are on the floor
and always well placed. A short run of bad frames inside good ones (under a
second) is not swapped to the feet. The filter predicts through it instead. The
first version did swap, and on UR Fall `adl-12` the jump from hips to feet read
as sway: its unsteady-gait station calls went from 1 to 4. With the gap handled,
all 21 UR Fall outcomes match the old engine exactly, call for call. On the
oblique cut, walking readings during still stands fell from 36 of 40 to
**0 of 40**. The synthetic evaluation did not change, as the table above shows.

**What it did not fix, and why.** On the front cut, 17 of 40 upright readings
still say `walking`, down from 30. The cause is the stated limitation that
Preempt tracks one person, the one with the largest box. While the woman sits,
the standing clinician has the bigger box and is the one tracked. As the woman
rises, tracking jumps from the clinician to her, 0.68 m across the floor in
one frame, and the Kalman velocity takes most of a second to decay. It is the same
flip that means no stand on the front cut is ever seen from start to finish,
which is why all three are missed. This is real-footage evidence for the
limitation, and it is not fixed here.

### What these clips do and do not show

- Lead times here are 0.1 to 0.4 s. A 30-second chair-stand test is the worst
  case: it asks for fast repeated stands with no slow preparation. The slow bed
  exit that the 4.6 s figure describes is not in this footage.
- The three hits needed a room drawn by hand, with a camera height estimated from
  a person assumed to be 1.65 m tall, after trying two zone layouts. Read them as
  "the pipeline runs end to end on real pixels and responds to real stands", not
  as accuracy.
- Two people in the frame defeats the tracker, as the front cut shows.

## 6. The binding limitation, measured

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

## 7. Gait, and a measure that was removed

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

## 8. Speed, and the OpenCV 5 engine

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

## 9. Geometry, checked against closed-form truth

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

## 10. Privacy, asserted rather than claimed

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

## 11. What these numbers are not

They are not a clinical result. They are not evidence that fewer people fall.
Nearly half of inpatient falls are a knee buckling, and 40 per cent of the falls
in one rehabilitation study happened during gait training, which is while a
therapist was already present and holding on. A camera does not prevent a knee
giving way, and this product does not claim to.

What it claims is narrower and is what was measured: that the movement which
precedes standing is visible several seconds before the weight leaves the bed,
that it can be told apart from turning over and from lying back down, and that
someone on the floor can be recognised geometrically rather than guessed.
