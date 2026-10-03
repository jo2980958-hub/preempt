# Preempt evaluation

Four quantities decide this product: lead time, detection rate, false-alarm rate,
and what it does when it is wrong.

| | |
|---|---|
| Median lead before upright, 10 bed exits | **4.60 s** (4.07 to 5.60) |
| Median lead, all 15 bed and chair exits | **4.47 s** (0.73 to 5.60) |
| Exits called, synthetic decision layer | **25 of 25** |
| False alarms in 1.23 h of observed quiet | **0** |
| Real falls reaching "on the floor", UR Fall | **10 of 12**, median 0.50 s behind ground truth |
| CDC chair-stand, oblique cut, hand-set room | **3 of 3** stands called, 0 false calls |
| Lying frames giving six or more usable joints | **68 %** (383 of 561) |
| Camera bytes written to disk | **0** |

**No deployed-system outcome trial exists for this class of product.** We
searched across camera-based safety products in several domains and found none.
Nothing below is evidence that fewer people fall, for this product or any
comparable one.

```bash
python -m preempt.cli evaluate --seeds 5 --quiet-loops 16 --out eval/synthetic.json
eval/fetch_urfall.sh 12 && python -m preempt.cli urfall eval/data --room eval/urfall-room.json
python -m preempt.cli bench --runs 40
python -m pytest tests/test_real_footage.py
python -m preempt.cli video cdc-chair-stand-oblique.mp4 --room cdc-oblique-hand-room-chairzone.json
```

## 1. Three tracks, and what each one can prove

**The synthetic track** measures the decision layer. Twelve scenarios project a
seventeen-joint body model through a pinhole camera whose floor homography,
vertical vanishing point and reference height the engine is then given, so the
geometry is exact and any error is the engine's. Keypoints get noise at the level
a real pose estimator produces, with dropouts and varying scores. The answer is
written into the sequence — the frame at which the person began to stand is
known, so lead time is measurable, and no real dataset carries that label. It
proves nothing about pose on a real patient under a real blanket.

**The UR Fall track** measures the whole pipeline on real footage, against
per-frame ground truth for lying on the floor: twenty-one sequences, twelve
labelled falls and nine activity sequences, cam0 RGB at 640x480. It is where the
binding limitation shows up.

**The CDC chair-stand clips** are real footage of the movement Preempt is built
to call: an older woman standing up from a chair and sitting back down, filmed by
the US Centers for Disease Control for its STEADI falls-prevention training,
public domain. No per-frame labels, so no lead-time distribution, but they found
three flaws neither other track could.

**The UR Fall licence.** Kwolek and Kepski, University of Rzeszów,
<http://fenix.ur.edu.pl/~mkepski/ds/uf.html>, from the dataset page:

> "This work is licensed under a Creative Commons
> Attribution-NonCommercial-ShareAlike 4.0 International License and is intended
> for non-commercial academic use."

Non-commercial and ShareAlike, so it is fetched at evaluation time by
`eval/fetch_urfall.sh`; nothing derived from it is committed here or appears in
the report, the deck or the video.

## 2. Lead time, which is the whole product

From the first call going out to the person being upright and staying upright for
half a second. Bed and chair exits only, because those are the sequences in which
being upright happens.

| | all exits | bed exits | quick chair stands |
|---|---|---|---|
| sequences | 15 | 10 | 5 |
| median | **4.47 s** | **4.60 s** | 1.00 s |
| mean | 3.44 s | 4.69 s | 0.93 s |
| range | 0.73 to 5.60 s | 4.07 to 5.60 s | 0.73 to 1.00 s |

The lead comes from calling in phase one of a four-phase sit-to-stand, while the
person is still supported; a pressure pad cannot know anything until the weight
has gone at the end of phase two. A fall detector's lead is negative by
definition, because it fires after the event.

The five chair stands came in with the sit-down fix in section 5 — a patient
standing straight up with a 1.2 second lean — and they lower the median over all
exits. No lead time moved with the fix: all fifteen are the same to the hundredth
of a second before and after.

## 3. Detection rate and false alarms, synthetic track

Twelve scenarios at five noise seeds each, quiet scenarios looped sixteen times
so the false-alarm denominator is worth dividing by.

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
extrapolated to a twelve-hour night is an assumption. One stray call in that
window would have scaled to about ten a night, and before the sit-down fix one
did. The number is zero because nothing fired, not because the observation was
long.

The quiet scenarios are the ones a ward would worry about: a patient asleep who
turns over twice, one who sits up to drink and lies back down, a visitor shifting
in the chair, somebody walking to the chair and sitting in it, a member of staff
crossing the room. Calls raised after `sit_down_starts` are counted separately
and any such call fails the sequence; graded per sequence instead, a false call
on a sit-down hides behind the correct call on the rise, which is how one on
every sit-down went unnoticed until real footage showed it.

**The cost of a paused camera.** A ward will pause the camera for washing and
dressing, and it should: Maidhof et al. (*Front Public Health* 2023,
PMID 37469701) found camera acceptance lowest for intimate activities such as
changing clothes and showering. `personal-care-pause` makes that cost visible.
Staff pause for nine seconds and the product does not show a calm green light
through it: it reports `"blind_by_choice_s": 9.0, "faulty_s": 0.0` and refuses
with `BLIND_BY_CHOICE`, whose message says the window is unknowable, not absent.
No call is raised, not even a maintenance notice, because staff pressed the
button. Two tests assert it, and calls missed in those windows cannot be counted
by any evaluation, including this one.

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

Re-run after the fixes in section 5: all 21 outcomes, delays and call counts are
identical to the engine before them. The half-second delay is `floor_hold_s`,
which requires the state to hold before an urgent call goes out.

- **fall-01 and fall-03 were missed.** The person is detected, but pose collapses
  as they land and the state never holds long enough.
- **adl-01 and adl-02 falsely said "on the floor".** The subject bends down or
  crouches, which is geometrically close to lying: the head travels forward and
  down and back-projects onto the floor plane about where a lying person's would.
  The bound that separates them is narrow and is stated explicitly in
  `Thresholds.floor_body_max_m`.

## 5. Real footage of standing up: the CDC chair-stand clips

Three single-shot cuts from two CDC STEADI training videos on Wikimedia Commons,
both public domain (`{{PD-USGov-HHS-CDC}}`): 12.4 s and 12.8 s of the *30-Second
Chair Stand Test* from the front and from an oblique angle, and 17.2 s of the
*Timed Up and Go Test*. None is in this repository; only the keypoints read from
the oblique cut are committed, in `tests/data/`, and that file has no pixels.

Nobody measured these rooms, and auto-calibration refused both fixed shots,
correctly, because nobody walks toward or away from the camera in them. The rooms
were set up by hand: a level camera, a focal length of 1035 px from a partial
calibration of the front shot, and a camera height of 0.89 m (front) or 0.81 m
(oblique) from assuming the standing clinician is 1.65 m tall. These are
assumptions; every room file says so in its `calibration` block, as do the
interface and both evidence cards. On the oblique cut the moment each stand
became upright was read from the keypoints, knee angle first past 150 degrees, at
1.08, 5.17 and 9.33 s.

Local CLI, OpenCV 5.0.0, before and after the two fixes below. A hit is a
clinical call raised during a stand before the person is upright; a false call is
one raised anywhere else.

| Clip | Room | Stands | Calls | Hits | Misses | False calls | Lead times |
|---|---|---|---|---|---|---|---|
| front | default | 3 | 0 → 0 | 0 → 0 | 3 → 3 | 0 → 0 | — |
| front | hand setup, no zones | 3 | 0 → 0 | 0 → 0 | 3 → 3 | 0 → 0 | — |
| front | hand setup, chair zone | 3 | 0 → 0 | 0 → 0 | 3 → 3 | 0 → 0 | — |
| oblique | default | 3 | 0 → 0 | 0 → 0 | 3 → 3 | 0 → 0 | — |
| oblique | hand setup, no zones | 3 | 1 → 1 | 1 → 1 | 2 → 2 | 0 → 0 | 0.08 s → 0.08 s |
| oblique | hand setup, chair zone | 3 | **6 → 3** | 3 → 3 | 0 → 0 | **3 → 0** | 0.08, 0.42, 0.41 s, unchanged |
| TUG | default | 1 | 0 → 0 | 0 → 0 | 1 → 1 | 0 → 0 | — |

The TUG cut produced two `maintenance` notices and a `VIEW_UNUSABLE` refusal
before and after, which is right: the camera pans to follow her and 22 per cent
of frames were marked "camera moved".

### Flaw 1: the service could not take anyone else's room

The live service had one room built in, the synthetic ward's, so every real clip
was measured against the wrong floor. On the oblique cut the woman's feet fell
inside that room's *bed* zone, every stand read as "supported", and all three
were missed.

An upload now carries its room, checked by the same parser the CLI uses; a bad
room is a 400 that names the field (`floor.image_points: needs exactly 4 points,
got 3`). The result records which room it used and whether that room's camera
height and focal length were measured or assumed. A video stops at a preview that
draws the room over its first frame: with the default room on the oblique clip
the preview shows the bed zone lying across the chair and her feet, and warns
that 5 of 24 setup points fall outside the frame.

On the live service (git sha `9a1565a`, us-east-2) the oblique clip with its
hand-set chair-zone room gave three `nudge` calls at 1.00, 4.75 and 8.92 s, a
first lead of 0.08 s, 0 of 40 upright readings as walking and 0 camera bytes
written, with a per-frame state timeline identical to the local CLI's.

### Flaw 2: sitting down raised a call to get up

Three more `nudge` calls fired at 2.83, 7.00 and 11.00 s, each as she sat back
down. The rule behind them is "the shoulders have arrived over the feet", and to
lower yourself onto a seat you lean forward over your feet with your knees bent.
No threshold separates the two, because the postures are alike.

**What changed: direction of travel.** A sit-down is the hips coming down from
standing — hip height falling at least as fast as a rise goes up (0.25 m/s, the
existing `rise_hip_speed_mps`, fitted over half a second), or the knees closing
at 90 degrees a second or more where there is no metric height, while the person
was upright within the detector's two-second history. While that holds the lean
is thrown away; a rise starts with the hips seated and still, or going up, so its
evidence is unaffected. The timeline then reads "sitting down: the hips came down
from standing at 0.57 m/s". The guarding test replays the oblique keypoints with
the hand room: it failed on the old code with exactly `[2.833, 7.0, 11.0]` and
passes on the new one, where all three stands are still called at their old times.

As an overfitting check, two synthetic chair scenarios were added, which is ten
more sequences at five noise seeds: a walk to the chair and a sit-down, and a
stand out of the chair and straight back down.

| synthetic evaluation | before | after |
|---|---|---|
| published set, 50 sequences: detection | 100 % (20 of 20) | 100 % (20 of 20) |
| published set: false alarms, 1.21 h quiet | 0 | 0 |
| published set: median lead, 10 bed exits | 4.60 s (4.07 to 5.60) | 4.60 s (4.07 to 5.60) |
| with the chair scenarios, 60 sequences: detection | 100 % (25 of 25) | 100 % (25 of 25) |
| with the chair scenarios: false alarms, 1.23 h quiet | **1** (9.8 per bed-night) | **0** |
| with the chair scenarios: calls raised on a sit-down | **3** in 10 sequences | **0** |
| with the chair scenarios: median lead, 15 exits | 4.47 s (0.73 to 5.60) | 4.47 s (0.73 to 5.60) |

No detection was lost and no lead time changed. The old engine also failed 3 of
the 10 new synthetic sit-down sequences — two scenarios across five noise seeds —
so this was not something only one clip could trigger.

### Flaw 3: standing still reported as walking

On 36 of 40 upright readings on the oblique cut the woman was standing still and
Preempt said `walking`. **The cause was geometry, not movement.** The torso's
floor position is the hips' shadow, found by sliding down the vertical from the
hips by their height. With the camera at about hip height (0.81 m, hips at
0.86 m) the ray to the hips is nearly level and that construction is singular:
one pixel of keypoint noise moved the torso tens of metres, placing it 25 m away
and back between frames, which the Kalman filter read as 16 m/s. The synthetic
ward camera, 2.55 m up in a corner, moves the shadow 5 to 14 mm per pixel; the
CDC cameras move it 0.05 to 0.1 m with the person seated and from 0.2 m to over a
kilometre with them standing.

The conditioning is now measured every frame, and if one pixel of hip error would
move the torso more than 3 cm (about 12 cm at RTMPose's roughly 4 px of noise)
the feet are used instead. A short run of bad frames inside good ones (under a
second) is not swapped but predicted through, because the first version did swap
and on UR Fall `adl-12` the jump from hips to feet read as sway, taking its
unsteady-gait station calls from 1 to 4. With the gap handled, all 21 UR Fall
outcomes match the old engine call for call, and walking readings during still
stands on the oblique cut fell from 36 of 40 to **0 of 40**.

On the front cut, 17 of 40 upright readings still say `walking`, down from 30,
and that is not fixed. Preempt tracks the person with the largest box; while the
woman sits that is the standing clinician, and as she rises tracking jumps to
her, 0.68 m across the floor in one frame, with the Kalman velocity taking most
of a second to decay. The same flip means no stand on the front cut is seen from
start to finish, which is why all three are missed.

Lead times on these clips are 0.1 to 0.4 s: a 30-second chair-stand test asks for
fast repeated stands with no slow preparation, so the slow bed exit behind the
4.6 s figure is not in this footage.

## 6. The binding limitation, measured

The decision layer is not the weak part; pose estimation on a person already on
the floor is. Over 561 ground-truth lying frames across the twelve fall
sequences:

| | frames | share |
|---|---|---|
| a person was detected at all | 519 | 93 % |
| six or more usable joints came back | 383 | **68 %** |
| the body back-projected onto the floor plane consistently | 260 | 46 % |
| the head landed a body length from the feet | 336 | 60 % |

A third of the frames that matter most produce too little pose to reason about: a
detector trained mostly on upright people sees a person lying down as a wide,
short, unfamiliar blob, and a top-down pose estimator given a bad box returns
noise.

Detector coasting (up to six frames on the last box widened by a fifth) and a
0.4 second grace period on the floor state recover part of it — at a 68 per cent
per-frame yield a strict hold never completes. Neither fixes it. What would is a
detector or a pose model trained with supine and prone people in it.

## 7. Gait, and a measure that was removed

Gait is scored over a four-second window from lateral sway about the fitted
walking line and a hand held out to a wall. Sway separates cleanly on the
synthetic walks: 2.3 to 3.8 cm RMS steady, 10.8 to 12.7 cm unsteady.

**Step-time variability was in the score and was taken out.** The coefficient of
variation of step intervals is the standard clinical gait-variability statistic,
but at fifteen hertz a 1.9 Hz cadence gives about eight samples per step and the
peak of the smoothed foot-separation signal lands a sample either side at random.
On synthetic walks that are steady by construction the CV came out between 0.26
and 0.52 — the magnitude of the clinical effect it was meant to detect — and it
produced two false alarms in twenty quiet sequences on its own. It is still
computed and shown to a clinician, contributes nothing to the score, and
`Thresholds.step_time_cv_warn` says so.

## 8. Speed, and the OpenCV 5 engine

`python -m preempt.cli bench --runs 40`, x86, 22 threads,
`opencv-python==5.0.0.93`, on a machine running four other builds at the time, so
the ratios are the result and the absolute numbers an upper bound.

| stage | ENGINE_CLASSIC | ENGINE_NEW | speedup |
|---|---|---|---|
| RTMPose-t 256x192, crop and normalise included | 9.30 ms | 7.83 ms | 1.19x |
| YOLOX-tiny 416x416, letterbox included | 44.01 ms | 27.46 ms | 1.60x |

On an unloaded machine the same script gave 6.46 / 5.37 ms and 30.01 / 18.10 ms,
and the bare `net.forward()` for RTMPose 5.3 ms classic against 3.1 ms new, a
1.7x ratio. The new graph engine is the default and is faster on the same weights
at the same call site, and the two engines agree on the keypoints to within a
pixel (`tests/test_pose.py`). At the deployed 15 Hz sample rate the pipeline runs
faster than real time on two vCPU.

## 9. Geometry, checked against closed-form truth

`tests/test_geometry.py` checks the floor frame against a camera whose answers
are known exactly.

- The four setup points round-trip to under 0.01 px.
- A floor point back-projects to within 0.1 mm of where it came from.
- Heights above the floor come back to **under 1 cm** across the room, from
  0.15 m to 1.7 m.
- The floor shadow of a point at a known height is exact to 1 mm.
- The projective vertical matches the true image direction of "up" to 0.06
  degrees, and differs from the naive image y axis by up to **31.7 degrees**
  across this room. Taking the image vertical for "up" puts thirty degrees of
  error into a trunk angle at the corner of the frame.

From a person walking about a synthetic room, `preempt.calibrate` recovers a
focal length of 752.8 px against a true 760, a camera height of 2.52 m against a
true 2.55, and heights across the room to within 1 cm.

## 10. Privacy, asserted rather than claimed

`tests/test_privacy.py` runs the whole pipeline with the real models over a video
file whose every frame is unique high-entropy content, then asserts:

- `camera_bytes_persisted == 0` and `frames_retained == 0`;
- every file the run wrote, decoded and correlated against every source frame,
  correlates below 0.35;
- asking the guard to write a camera frame raises `PrivacyViolation`, and no file
  appears;
- `release()` leaves the frame buffer zeroed;
- diagnostic mode raises unless an environment variable the deployed service
  never sets is present;
- and, by reflection, that no function in `render.py` accepts a parameter named
  `image`, so there is no code path from a camera frame to a saved file.

## 11. What these numbers are not

Not a clinical result, and not evidence that fewer people fall. In one inpatient
rehabilitation study 47.5 per cent of falls were a knee buckling and 40 per cent
happened during gait training, while a therapist was already present and holding
on. A camera does not prevent a knee giving way.

What was measured is narrower: that the movement which precedes standing is
visible several seconds before the weight leaves the bed, that it can be told
apart from turning over and from lying back down, and that someone on the floor
can be recognised geometrically rather than guessed.
