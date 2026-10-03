# Preempt — technical report

**A call before the fall, from an abstracted pose and nothing else.**

Live endpoint: <https://2uhvgzwrwc.us-east-2.awsapprunner.com>
OpenCV 5.0.0.93, pinned. AWS App Runner, us-east-2. 124 tests.

---

## 0. Summary

Preempt raises a call *before* someone falls: it recognises the movement that precedes
standing up, calls while the person is still supported, scores how steady they are once
on their feet, and says plainly when it cannot see. The frame is destroyed as soon as
the pose has been read, so the evidence is a stick figure, never a photograph.

| | |
|---|---|
| Median lead before upright, 10 bed exits | **4.60 s** (4.07 to 5.60) |
| Median lead, all 15 bed and chair exits | **4.47 s** (0.73 to 5.60) |
| Detection rate, synthetic decision layer | **25 of 25** |
| False alarms in 1.23 hours of observed quiet | **0** |
| Real CDC chair-stand footage, room set by hand | **3 of 3** stands called, 0 false calls |
| Real falls reaching "on the floor", UR Fall | **10 of 12**, median 0.50 s behind ground truth |
| Camera bytes written to disk | **0** |

Lead time is synthetic because no real dataset labels the frame a stand begins. On the
real chair-stand clips, which ask for the fastest stands a patient can manage, the
calls came 0.08, 0.42 and 0.41 s before upright. Method: [evaluation.md](evaluation.md).

---

## 1. The problem, with the evidence

**The money is verified and large.** Florence, Bergen, Atherly and colleagues put the
medical costs attributable to fatal and nonfatal falls in the United States at
approximately **$50.0 billion** in 2015, of which Medicare paid approximately **$28.9
billion** for nonfatal falls, Medicaid $8.7 billion and other payers $12.0 billion
(*J Am Geriatr Soc*, April 2018, PMID 29512120).

**The regulator has already made a fall the provider's problem.** CMS lists "Falls and
Trauma" among the Hospital-Acquired Conditions, and in the Federal Register of 6 June
2011 extended that to Medicaid as "clinically applicable to all Medicaid populations".
A hospital is paid nothing extra for treating a fall with injury on its own ward, so
the buyer has the budget line already.

**The rate is a range, not a number**, roughly 0.2 to 5.4 per 1,000 patient-days
depending on acuity: a UK NHS dementia ward at **5.4 falls per 1,000 occupied bed
days** before a quality-improvement intervention took it to 1.4 (Sorlie et al., *BMJ
Open Quality* 2026, PMID 42225373), against a Chinese tertiary hospital at 0.22 to 0.29
across 325,377 admissions (Liao et al., *Scientific Reports* 2026, PMID 41826360).
Globally the WHO puts falls at an estimated 684,000 deaths a year and 37.3 million
needing medical attention.

**What a camera cannot do.** NYU Langone's inpatient rehabilitation unit, 40 falls
across 6,238 admissions, breaks them down by mechanism: **47.5 per cent were buckling**
and **40 per cent happened during gait training**, while a therapist was already in the
room with their hands on the patient (Camillieri et al., *Physical Therapy* 2025).
A camera does not prevent a knee giving way, and this product does not claim to.

**The contradiction at the centre of the domain.** Tham, Brady, Ziefle and Dinsmore,
reviewing older adults' acceptance of camera-based assisted-living technology, found
the "dominant barriers concerned the technology's privacy-invasive, obtrusive, and
stigmatizing qualities" (*Innovation in Aging* 2024, PMID 39968358). Maidhof, Offermann
and Ziefle measured it across twenty-five activities of daily living and found "the
strongest barrier perception... for intimate activities", privacy needs running "very
high (e.g., changing clothes, showering)" (*Frontiers in Public Health* 2023,
PMID 37469701). The objection is to being filmed, and it is strongest in the private
half of a patient's day. That is the design brief: process on the device, abstract to
pose, never store or transmit imagery, and be able to *show* it (section 5).

---

## 2. Users

**The person in the bed** gets a quiet spoken prompt in the room before anyone else is
told, because a person who is told someone is coming usually waits. **The night nurse
at the station** gets a call naming the room, the state and the reasons:
"bed 4, about to get up, the upper body is moving over the feet, the walking frame is
2.1 m away." Not a buzzer, not a score. And **the family** get the paragraph that
decides whether the camera stays in the room, which is the same paragraph the interface
prints.

---

## 3. Architecture

Full diagrams in [architecture.md](architecture.md). Everything after the frame is
destroyed works on seventeen (x, y, score) triples on a metric floor plane, and the
evidence is drawn from those keypoints onto a blank canvas.

```
frame -> view grade -> detect -> pose -> FRAME DESTROYED
      -> kinematics -> exit state / gait window / floor test
      -> risk state -> escalation -> evidence + RunRecord
```

`visioncore` carries the OpenCV 5 primitives and the run record, `servicekit` the
FastAPI shell and job queue.

---

## 4. The OpenCV 5 implementation

### 4.1 What OpenCV 5 changed, and what it forced

`readNetFromCaffe` and `readNetFromDarknet` are **removed** in 5.0, so everything is
ONNX and there is no second inference runtime in the image. `cv2.TrackerCSRT` and the
legacy tracking namespace are gone: there is one person, and continuity comes from a
Kalman filter on the floor position plus the detector-coasting in 4.6.
`VideoCapture.get()` returns **-1** for an unsupported property where 4.x returned 0,
so `visioncore.imageio` treats anything below zero as missing. And `cv2.FontFace` is
new and used: the evidence cards render through a real TrueType engine rather than
Hershey strokes.

**The version is pinned and asserted** — in three `pyproject.toml` files and in
`constraints.txt`, `import visioncore` raises on a 4.x wheel, the Docker build checks
the major version, `/version` prints what is running — because unpinned, `pip install
opencv-python` resolves to 4.14.x, which shipped *after* 5.0.0.

### 4.2 The DNN engine, measured

`readNetFromONNX(path, engine=cv2.dnn.ENGINE_NEW)` against `ENGINE_CLASSIC`:

| | classic | new | ratio |
|---|---|---|---|
| RTMPose-t, crop and normalise included | 9.30 ms | 7.83 ms | 1.19x |
| YOLOX-tiny, letterbox included | 44.01 ms | 27.46 ms | 1.60x |
| RTMPose-t bare `net.forward()` | 5.3 ms | 3.1 ms | 1.71x |

`tests/test_pose.py` asserts the two engines agree on the keypoints to within a pixel.
The wheel is built with `ONNX Runtime: NO`, so `ENGINE_ORT` is a constant with no
backend; asking for it raises.

### 4.3 Models, and the licence that was avoided

Person detection is **YOLOX-tiny** (Megvii, Apache-2.0); pose is **RTMPose-t body7**
(OpenMMLab MMPose, Apache-2.0). Ultralytics' YOLO pose models are **AGPL-3.0**, whose
section 13 extends copyleft to network use, making a hosted demo on those weights a
source-disclosure event: nothing here imports `ultralytics`, and the module docstring
says so. Both models are baked into the image at build time from their upstream URLs
with checksums.

### 4.4 The floor plane, which is where the product gets its metres

`findHomography` takes four floor points a nurse clicked and their measured spacing,
and gives two things exactly: back-projection from pixel to floor metres, and the
horizon, `l = H^-T (0,0,1)`. A point imaged past the horizon cannot be on the floor, so
`to_floor` returns NaN and the caller says so. `solvePnP` with `SOLVEPNP_IPPE_SQUARE`
was deliberately not used: it returns an ambiguous solution for a planar target, and was
measured reporting 2 degrees for a true 20 degree tilt.

One vertical reference of known height rescales the vertical vanishing point so one
unit of it is one metre, after which `p ~ H b~ + h vz_h` gives a height when the floor
position is known and a floor position when the height is known. Against a camera whose
answers are known in closed form, heights come back to **under 1 cm** across the room
from 0.15 m to 1.7 m. A single view cannot do both at once, so a point of unknown
height is assumed to sit above the person's foot contact, and the code records it.

**The construction is singular when the camera sits at hip height**, and real footage
found it. On a CDC clip filmed at 0.81 m, one pixel of keypoint noise moved a standing
woman's torso — the hips' shadow, slid down the vertical by their height — tens of
metres, and the Kalman filter reported 16 m/s for somebody standing still. The
conditioning is now measured every frame: if one pixel would move the shadow more than
3 cm, the feet are used instead. The synthetic ward camera, 2.55 m up, sits at 5 to
14 mm a pixel and never triggers it.

### 4.5 Setting a camera up by walking through the room

`preempt.calibrate` works from people rather than a tape measure: every upright stance
is a vertical segment in the image, all of them meet at the vertical vanishing point,
and any two at different depths put a point on the floor's horizon. With those and an
assumed stature a camera with square pixels and a central principal point is
determined, because `K^-1 vz` is parallel to `K^T l`. On a synthetic room it recovers a
focal length of **752.8 px against a true 760**, a camera height of **2.52 m against a
true 2.55**, and heights to within a centimetre.

The textbook two-point horizon construction, crossing head lines with foot lines, is
unusable on real footage: on the UR Fall camera it gave a horizon tilted 24 degrees and
no real solution for the focal length. The module searches for a level horizon instead,
and prefers a known focal length to either. It refuses outright when the footage cannot
support it, as it did on both fixed CDC chair-stand shots — an implausible camera height
on one, an implausible focal length on the other — because nobody in them walks toward
or away from the lens.

**Every room says how far its camera can be trusted, and travels with the job.** A
`calibration` block marks camera height and focal length as measured, assumed or exact,
and no block counts as unstated, which the interface treats as assumed. An upload
carries its own room, checked by the same parser as the command line, and a bad room is
a 400 naming the field. Before a video runs the interface draws that room over its
first frame, decoded in the browser, because the engine cannot tell a room belongs to a
different camera.

### 4.6 Where the seconds come from

Standing up from a seat is four phases — flexion momentum, momentum transfer,
extension, stabilisation — and a pressure pad knows nothing until the weight has gone,
the end of phase two. A camera watching the upper body arrive over the feet sees phase
one. The measured quantity is `lean_offset`: the shoulders' displacement from the foot
contact perpendicular to the projective vertical, over their distance along it.
Dimensionless, so it needs no scale and survives any camera position; signed, so
leaning back onto a pillow raises nothing. Strongly negative sitting on the edge of a
bed, it is driven toward zero by standing up, and that arrival *is* the
momentum-transfer phase.

A call is raised when three things hold together, because each alone has an innocent
explanation: the rate of that arrival is at or above 0.18 per second **and** the travel
across the window is real rather than a wobble; the shoulders have already come within
0.32 of being over the feet; and the feet are on the floor beside the furniture with
the person sitting a moment ago. The third removed the last synthetic false alarm,
because somebody crossing past the foot of the bed satisfies the edge and lean tests in
the same instant.

**None of the three could tell sitting down from getting up**, and real footage found
it: on a CDC chair-stand clip all three stands were called and three more calls fired
as the woman sat back down, because lowering yourself onto a seat is the same shape as
rising from one. A fourth condition uses direction of travel. If the hips are coming
down from standing — falling at least as fast as a rise goes up, 0.25 m/s over half a
second, or the knees closing at 90 degrees a second where there is no metric height —
and the person was upright in the last two seconds, the lean is thrown away; a rise
starts with the hips seated and still, or going up, so its evidence is untouched. The
three sit-down calls went, the three stands were still called at the same instants, and
nothing moved on the synthetic track. The old engine also failed 3 of the 10 new
synthetic sit-down sequences — two scenarios across five noise seeds — so this was not
a quirk of one clip. Evidence gets a 0.35 second grace before the clock restarts,
because one bad frame is keypoint noise.

### 4.7 Unsteady gait

Scored over a four-second window of walking, and only of walking: the window clears
the moment the person stops being on their feet, because letting a sit-to-stand into it
put a metre of apparent sway into the next four seconds. Sway is the RMS perpendicular
distance of the torso's floor track from a line fitted with `cv2.fitLine`, so walking
round a corner is not scored as sway, which a fixed-axis measure gets wrong. Measured:
2.3 to 3.8 cm for a steady walk against 10.8 to 12.7 cm for an unsteady one. A hand
held out to a wall for more than 0.6 seconds is the second measure.

Step-time variability, the standard clinical statistic, is computed, reported and
**deliberately not scored**: at fifteen hertz it came out between 0.26 and 0.52 on
walks steady by construction — the magnitude of the effect it was meant to detect — and
produced two false alarms in twenty quiet sequences on its own.
[evaluation.md](evaluation.md) has the numbers.

### 4.8 On the floor, geometrically

This test was wrong twice, silently. Asking whether the head was low **in metres**
needs a base point the head is above, and a lying person's head is a body length
*sideways* from their feet: the height came out between three and five metres and fired
zero times on twelve real falls. Measuring the body's **length flattened onto the
floor** saturates — standing and lying both gave about 2.2 m. What survives real footage
is where the head lands back-projected as if it were on the floor: 0.99 to 1.69 m from
the feet for someone on the floor, against 2.5 to 5.3 m for someone standing, or past
the horizon and therefore nowhere at all. Cleanly separated on UR Fall, with the
fraction of joints that back-project sensibly as a second condition.

### 4.9 Hazards

Three, all classical OpenCV with no model behind them. **A walking frame out of reach**
is a floor-plane distance in metres from the person to the aid's registered zone.
**Clutter on the route** is `absdiff` against the signed-off room, Otsu, `morphologyEx`
and `connectedComponentsWithStats`, keeping components whose lowest point back-projects
into the walking corridor above 400 square centimetres of floor. **A wet-floor sign**
is HSV `inRange` for saturated yellow, then `convexHull`, `approxPolyDP` and a solidity
and aspect test. A hazard never raises a state on its own: it raises the escalation one
rung when a state is already in play, and never into urgent.

---

## 5. Privacy by construction

The claim is narrow and checkable: **in strict mode, no bytes derived from camera
pixels are written to disk, sent over the network, or held after the pose has been
read.** Not anonymised, not blurred. Nothing leaves.

Four things enforce it. `PoseEstimator.estimate` is the only code that touches an
image, and it returns keypoints. Every write to disk goes through `PrivacyGuard.emit`,
which requires a `Provenance`, and `CAMERA` in strict mode raises with no override.
`SYNTHETIC` artefacts are drawn by `render.py` from keypoints onto a blank canvas, and
**the renderer's signatures do not accept an image** — a test asserts that by
reflection, so there is no code path from a camera frame to a saved file. `diagnostic`
mode raises unless an environment variable is set that the service never sets.

One place turns a frame into keypoints and one line stops it existing. From
`pipeline.Pipeline.run_video`:

```python
with stage("pose:total"):
    pose = self.estimator.estimate(image, frame.index, time_s)
view = self.view.grade(image, time_s, person_seen=pose is not None)
hazards = self.hazards.scan(image, time_s, ...)
# Everything the camera gave us has now been read. Destroy it.
self.guard.release(image)
```

and `PrivacyGuard.release`, where a strict-mode run overwrites the one buffer in place
rather than rebinding it, so one frame is in memory at a time and no reference survives
the loop:

```python
def release(self, image: np.ndarray) -> None:
    if self.mode == "strict":
        image[...] = 0
    else:
        self.ledger.frames_retained += 1
```

`tests/test_privacy.py` runs the whole pipeline with the real models over a video whose
every frame is unique high-entropy content, then asserts zero camera bytes persisted,
zero frames retained, correlation below 0.35 between every written file and every
source frame, and that asking to write a frame raises.

**A 1280x720 BGR frame is 2.76 MB. The seventeen keypoints kept from it are 408
bytes** — about 6,800 to 1, and what is discarded is all of the identity.

**The cost of privacy is published rather than hidden.** A ward will pause the camera
for washing, toileting and dressing, and there is a button for it, so a paused camera
is a first-class state rather than an absence: every run reports `blind_by_choice_s`
separately from `faulty_s`, and the record carries a `BLIND_BY_CHOICE` refusal saying
the window is unknowable, not absent. **Calls missed while the camera is paused cannot
be counted, by anyone, including us**, so this product publishes the denominator.

---

## 6. AWS deployment

Container to ECR, ECR to App Runner at 2 vCPU and 4 GB, always on, HTTPS with no load
balancer. Both ONNX models, the seven bundled pose tracks and the default room are
inside the image, so there is no network at runtime: the container answers `/healthz`
two seconds after start and a bundled sample returns in under a second. It runs in
**us-east-2** because the account allows two App Runner services per region and both
us-east-1 slots were held when this was deployed.

---

## 7. Limitations

**Pose on a person already on the floor is the binding constraint.** Over 561
ground-truth lying frames a person was detected in 93 per cent but only 68 per cent
gave six or more usable joints, because a detector trained mostly on upright people
sees a lying person as an unfamiliar blob. Coasting and a floor-state grace period
recover some of it; a detector trained with supine and prone people would fix it.

**One camera, one person.** A room with a visitor and a patient tracks the larger of
the two. In the front-facing CDC clip a clinician stands beside the patient throughout,
so tracking jumps to her only once she is nearly up, 0.68 m across the floor in one
frame: no stand is seen start to finish, all three are missed, and the jump reads as
walking for most of a second.

**The metres are only as good as the setup.** The walking-person calibration is only as
good as its assumed stature: assuming 1.75 m for a 1.60 m patient reads every height
about 9 per cent high. The real footage here used rooms set up by hand from an assumed
1.65 m clinician, so those rooms say `assumed` and so does every result from them.

**Fast stands leave little warning**: four to six seconds is a slow bed exit, a quick
stand out of a chair gives about one second on the synthetic track, and the CDC
chair-stand test gives 0.1 to 0.4 s. **Heights fail when the feet are not on the
floor** — on a bed the contact point is a mattress, so heights inflate, and the engine
detects this from the zones a nurse drew and falls back rather than reporting an
inflated number.

**Bathrooms are not covered,** because a camera does not belong in one: Preempt is a
bedroom-and-bay product.

**The false-alarm rate has a small denominator**: zero in 1.23 hours of observed quiet
is real, and it is 1.23 hours, not a bed-night. The synthetic quiet scenarios also
missed a real false call, since every sit-down raised one until real footage showed it.
**Crouching reads like lying** — two of ten activity sequences with no lying frames said
"on the floor" when the subject bent down, and the separating bound is narrow.

**No outcome evidence exists.** No deployed-system outcome trial could be found for any
camera safety product in this domain. Preempt has not been shown to reduce falls on a
real ward, and neither has anything comparable.

---

## 8. The questions a ward would ask

**"Can anyone watch the camera?"** No: no video output, no recording, no stream. The
frame is destroyed after the pose is read, in the same function call, and the only
images produced are drawn from keypoints.

**"Does it do face recognition?"** No, and it cannot: no face model, no embedding, no
identity of any kind. The nose, eyes and ears are kept as *positions* — enough to tell
where a head is, nowhere near enough to tell whose.

**"Who gets told, and in what order?"** The person in the room first, quietly, then
the station, and an urgent call only when someone is on the floor. A nudge nobody
answers becomes a station call after twenty seconds.

**"What if a patient says no?"** The camera comes out of that room; consent has to be
refusable without argument and revisited, because capacity changes. Where a patient
cannot consent it is a documented best-interests decision with the family, and what
this design offers is that there is nothing to show anyone.

**"Is it a medical device?"** No, a monitoring and alerting aid: no diagnosis, no
treatment advice, no clinical action. It tells a human being to walk down the corridor
and look.

---

## 9. Reproducing this

Install per the README, then:

```bash
python -m pytest products/preempt/tests -q                    # 124 tests
python -m preempt.cli evaluate --seeds 5 --quiet-loops 16
products/preempt/eval/fetch_urfall.sh 12
python -m preempt.cli urfall products/preempt/eval/data --room products/preempt/eval/urfall-room.json
```

Dependencies are pinned in `constraints.txt` and each `pyproject.toml`; the Docker
build asserts the OpenCV major version and `/version` prints what is running.
