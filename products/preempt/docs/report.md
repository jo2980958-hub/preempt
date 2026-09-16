# Preempt — technical report

**A call before the fall, from an abstracted pose and nothing else.**

Live endpoint: <https://2uhvgzwrwc.us-east-2.awsapprunner.com>
OpenCV 5.0.0.93, pinned. AWS App Runner, us-east-2. 102 tests.

---

## 1. The problem, with the evidence

Falls are the second leading cause of unintentional injury death worldwide. The
World Health Organization's falls fact sheet:

> "Each year an estimated 684 000 individuals die from falls globally of which
> over 80% are in low- and middle-income countries. Adults older than 60 years of
> age suffer the greatest number of fatal falls... Though not fatal, approximately
> 37.3 million falls severe enough to require medical attention occur each year."

and

> "falls are responsible for over 38 million DALYs... and result in more years
> lived with disability than transport injury, drowning, burns and poisoning
> combined."

The money is verified and large. Florence, Bergen, Atherly and colleagues,
*Journal of the American Geriatrics Society*, April 2018, PMID 29512120:

> "In 2015, the estimated medical costs attributable to fatal and nonfatal falls
> was approximately $50.0 billion. For nonfatal falls, Medicare paid approximately
> $28.9 billion, Medicaid $8.7 billion, and private and other payers $12.0
> billion."

### Why a hospital or a care home is the buyer

Because the regulator has already made a fall with injury the provider's problem.
CMS lists "Falls and Trauma", with its sub-categories of fractures, dislocations,
intracranial injuries, crushing injuries, burns and electric shock, among the
Hospital-Acquired Conditions, and in the Federal Register of 6 June 2011 extended
that treatment to Medicaid, writing that conditions including "Falls and
Trauma... are clinically applicable to all Medicaid populations."

A fall with injury on a hospital's watch is a Hospital-Acquired Condition, and the
hospital is paid nothing extra for treating what happened on its ward. The buyer
has the budget line already.

The enforcement pool is separately large. CMS's Care Compare "Penalties" dataset
(id `g6vv-u9sr`, release of 26 August 2026) contains 13,256 fine records totalling
**$456,752,787** over a trailing three-year window, plus 2,440 payment-denial
penalties. That file has no deficiency-type column, so the fall-specific share
cannot be isolated: it is the size of the enforcement pool that nursing homes
operate inside, and it must not be quoted as fines for falls.

### The rate, which is a range and not a number

Three primary studies, three very different wards:

- a UK NHS dementia ward at **5.4 falls per 1,000 occupied bed days**, reduced to
  1.4 by a quality-improvement intervention (Sorlie et al., *BMJ Open Quality*
  2026, PMID 42225373);
- a Chinese tertiary hospital across 325,377 admissions at **0.22 to 0.29 per
  1,000 patient-days** (Liao et al., *Scientific Reports* 2026, PMID 41826360);
- NYU Langone inpatient rehabilitation, **0.43 falls per 1,000 patient days**
  across 6,238 admissions (Camillieri et al., *Physical Therapy* 2025).

Cite the range, roughly 0.2 to 5.4 per 1,000 patient-days depending on unit
acuity. Rehabilitation and dementia wards run an order of magnitude above general
medical wards.

### What a camera cannot do, from the same source

The NYU study breaks the falls down by mechanism: **47.5 per cent were buckling**
and **40 per cent happened during gait training**, which is to say while a
therapist was already in the room with their hands on the patient. A camera does
not prevent a knee giving way, and it does not add a second pair of hands. Any
claim that a camera prevents those falls is false, and this product does not make
it.

### The honest problem with the whole idea

This is the contradiction at the centre of the domain and it belongs at the front
of the pitch rather than in a footnote.

Tham, Brady, Ziefle and Dinsmore, *Innovation in Aging* 2024, PMID 39968358, on
older adults' acceptance of camera-based assisted-living technology:

> "Dominant barriers concerned the technology's privacy-invasive, obtrusive, and
> stigmatizing qualities"

Maidhof, Offermann and Ziefle, *Frontiers in Public Health* 2023, PMID 37469701,
found distinct acceptance patterns across twenty-five activities of daily living,
with

> "very high (e.g., changing clothes, showering) and very low privacy needs (e.g.,
> gardening, eating, and drinking)... The strongest barrier perception was found
> for intimate activities and mainly regarded privacy concerns."

Bathing, toileting and dressing are where a large share of falls happen. They are
also exactly the activities where camera acceptance is lowest, by a wide margin.
**The coverage gap and the acceptance gap point at the same rooms.**

That is the design brief, not an objection to it. A product that answers it has to
process on the device and abstract to pose, never storing or transmitting imagery,
and has to be able to show that rather than assert it. Preempt is built that way
and section 5 is the proof.

---

## 2. Users

**The person in the bed.** They get a quiet spoken prompt in the room before
anyone else is told, because most of the time a person who is told someone is
coming will wait. Nobody sees a picture of them.

**The night nurse at the station.** They get a call that names the room, the
state and the reasons, in a sentence they can act on: "bed 4, about to get up, the
upper body is moving over the feet, the walking frame is 2.1 m away." Not a
buzzer. Not a score.

**The ward sister or clinical lead.** They tune the thresholds. Every one of them
is a physical quantity with a unit, on one dataclass, with the reasoning written
next to it, because a fused number out of a classifier cannot be argued with and
a hip height in metres can.

**The family.** They get the one paragraph that decides whether the camera stays
in the room, and it is the same paragraph the interface prints: the frame is
destroyed as soon as the pose has been read, the evidence is a stick figure on a
room plan, and the ledger on screen shows how many camera bytes were written,
which is zero.

---

## 3. Architecture

Full diagrams in [architecture.md](architecture.md). In one paragraph: a frame is
decoded, graded for usability, passed to a person detector and then a pose
estimator, and then destroyed. Everything after that works on seventeen (x, y,
score) triples projected onto a metric floor plane. A state machine decides what
is happening, an escalation ladder decides who is told, and a renderer draws the
evidence from keypoints onto a blank canvas.

```
frame -> view grade -> detect -> pose -> FRAME DESTROYED
      -> kinematics -> exit state / gait window / floor test
      -> risk state -> escalation -> evidence + RunRecord
```

---

## 4. The OpenCV 5 implementation

### 4.1 What OpenCV 5 changed, and what it forced

`cv2.dnn.readNetFromCaffe` and `readNetFromDarknet` are **removed** in OpenCV 5.
Everything is ONNX. That is not a restriction here, it is a simplification: both
models load through one call and there is no second inference runtime in the
image.

`cv2.TrackerCSRT` and the whole legacy tracking namespace are gone from the main
wheel. Preempt needs no object tracker: there is one person, and continuity comes
from a Kalman filter on the floor position plus the detector-coasting described in
4.6.

`VideoCapture.get()` returns **-1** for an unsupported property where 4.x returned
0. The shared `visioncore.imageio` treats anything below zero as missing, which is
why a container with no frame-rate metadata falls back to a configured hint rather
than dividing by zero.

`cv2.FontFace` is new, and it is used: the evidence cards render through a real
TrueType engine rather than the Hershey strokes, which is the difference between a
screenshot a ward will put on a poster and one that looks like a 1990s demo.

**The version is pinned and asserted.** `opencv-python-headless==5.0.0.93` in
three `pyproject.toml` files and in `constraints.txt`; `import visioncore` raises
on a 4.x wheel; the Docker build runs a version check as a build step; and
`/version` prints what is actually running. `pip install opencv-python` unpinned
resolves to 4.14.x, which shipped *after* 5.0.0, so an unpinned install would fail
the competition's core requirement silently.

### 4.2 The DNN engine, measured

`readNetFromONNX(path, engine=cv2.dnn.ENGINE_NEW)`. On the same weights at the
same call site, against `ENGINE_CLASSIC`:

| | classic | new | ratio |
|---|---|---|---|
| RTMPose-t, crop and normalise included | 9.30 ms | 7.83 ms | 1.19x |
| YOLOX-tiny, letterbox included | 44.01 ms | 27.46 ms | 1.60x |
| RTMPose-t bare `net.forward()` | 5.3 ms | 3.1 ms | 1.71x |

`tests/test_pose.py` asserts the two engines agree on the keypoints to within a
pixel, so the speed costs nothing.

The wheel is built with `ONNX Runtime: NO`, so `ENGINE_ORT` exists as a constant
with no backend behind it. Asking for it raises here rather than falling back
silently.

### 4.3 Models, and the licence that was avoided

| | | |
|---|---|---|
| Person detection | **YOLOX-tiny**, Megvii | Apache-2.0 |
| Pose | **RTMPose-t (body7)**, OpenMMLab MMPose | Apache-2.0 |

Ultralytics' YOLO pose models would have been easier and are **AGPL-3.0**, whose
section 13 extends copyleft to network use. A hosted demo API on those weights is
a source-disclosure event. Nothing in this repository imports `ultralytics`, and
both this report and the module docstring say so, so that a later contributor does
not add it by accident.

Both models are baked into the image at build time from their upstream URLs with
checksums, by `models/fetch.sh`. Neither binary is committed.

### 4.4 The floor plane, which is where the product gets its metres

A single homography between the image and the floor turns an image into a metric
frame, and almost everything interesting about a person about to fall is a
distance or a height in that frame.

`findHomography` from four floor points a nurse clicked and their measured
spacing. From it, exactly and with no further calibration:

- **back-projection**, pixel to floor metres, for any point genuinely on the floor;
- **the horizon**, `l = H^-T (0,0,1)`, the vanishing line of the floor. A point
  imaged past it cannot be on the floor at all, so `to_floor` returns NaN there
  and the caller says so instead of returning a number.

`solvePnP` with `SOLVEPNP_IPPE_SQUARE` was deliberately not used for this. It
returns an ambiguous solution for a planar target, and the foundation agent
measured it reporting 2 degrees for a true 20 degree tilt.

With one vertical reference of known height, the vertical vanishing point can be
rescaled so that one unit of it is one metre, after which every world point
satisfies

```
p  ~  H b~  +  h vz_h
```

which gives a height when the floor position is known, and a floor position when
the height is known. Measured against a camera whose answers are known in closed
form, heights come back to **under 1 cm** across the room from 0.15 m to 1.7 m.

**A single view cannot do both at once**, and that limitation is not worked
around. A point of unknown height is placed by assuming it is above the person's
foot contact, which is true to a few centimetres for a hip and false for an
outstretched hand, and the code records which of the two it is doing. Without a
vertical reference, every height is refused with a reason and the back-projection
still works.

### 4.5 Setting a camera up by walking through the room

Asking a ward to measure a floor rectangle with a tape is the step most likely to
be skipped or done badly, and a badly measured rectangle produces confidently
wrong metres.

`preempt.calibrate` does it from people instead. Every upright stance is a
vertical segment in the image; all of them meet at the vertical vanishing point;
any two at different depths put a point on the floor's horizon. With those and an
assumed stature, a camera with square pixels and a central principal point is
determined, because `K^-1 vz` is parallel to `K^T l`.

On a synthetic room it recovers a focal length of **752.8 px against a true 760**
and a camera height of **2.52 m against a true 2.55**, and heights across the room
to within a centimetre.

One thing about it is worth recording because it cost real time. The textbook
two-point horizon construction, cross the head line with the foot line for every
pair of people and fit a line to the crossings, is correct and, on real footage,
useless. When two people stand at similar depths their crossing runs off to a
hundred thousand pixels and a few pixels of keypoint noise swings it. On the UR
Fall camera the fitted horizon came out tilted 24 degrees and the focal length had
no real solution at all. The module now falls back to a one-dimensional search for
a level horizon, and takes a known focal length in preference to either, which is
what the UR Fall evaluation uses.

### 4.6 Where the seconds come from

Standing up from a seat is four phases. Flexion momentum: the trunk leans forward
and the centre of mass travels toward the feet. Momentum transfer: the weight
leaves the seat. Extension. Stabilisation.

A pressure pad under a mattress cannot know anything until the weight has gone,
which is the end of phase two. A camera watching the upper body arrive over the
feet can see phase one.

The measured quantity is `lean_offset`: the shoulders' displacement from the foot
contact perpendicular to the projective vertical, divided by their distance along
it. It is dimensionless, so it needs no scale and survives any camera position,
and it is signed, so leaning back onto a pillow drives it the other way and does
not produce a call. Sitting on the edge of a bed it is strongly negative, because
the feet are out in front and the shoulders are back over the mattress. Standing
up drives it toward zero, and that arrival *is* the momentum-transfer phase.

A call is raised when three things hold together, because each alone has an
innocent explanation:

1. the rate of that arrival is at or above 0.18 per second, **and** the total
   travel across the window is real rather than a wobble;
2. the shoulders have already come within 0.32 of being over the feet;
3. the feet are on the floor beside the furniture, and the person was sitting a
   moment ago.

The third condition is the one that removed the last false alarm in the synthetic
evaluation: somebody crossing the room past the foot of the bed can satisfy the
edge test and the lean test in the same instant, and nobody begins to stand up
while they are already walking.

Evidence is allowed a 0.35 second grace before the clock restarts, because one bad
frame is keypoint noise and not a change of mind, and without it the evidence
restarts several times during a real transfer and the call arrives a second late.

**Median lead time: 4.60 seconds**, range 4.07 to 5.60.

### 4.7 Unsteady gait

Scored over a four-second window of walking, and only of walking: the window is
cleared the moment the person stops being on their feet, because letting a
sit-to-stand into it put a metre of apparent sway into the next four seconds.

Sway is the RMS perpendicular distance of the torso's floor track from a line
fitted with `cv2.fitLine`. Fitting the line means walking round a corner is not
scored as sway, which a fixed-axis measure gets wrong. Measured: 2.3 to 3.8 cm for
a steady walk, 10.8 to 12.7 cm for an unsteady one.

A hand held out to a wall for more than 0.6 seconds is the second measure.

Step-time variability is computed, reported, and **deliberately not scored**. It is
the standard clinical statistic and it was in the score until the evaluation was
run properly: at fifteen hertz its measurement noise floor is about 0.3, which is
the size of the effect, and it produced two false alarms in twenty quiet sequences
on its own. Section 6 of [evaluation.md](evaluation.md) has the numbers.

### 4.8 On the floor, geometrically

This test was wrong twice and it is worth recording both, because both looked
right and both failed silently.

**Version one asked whether the head was low, in metres.** A height needs a base
point the top is above, and a person lying down has their head a body length
*sideways* from their feet, so the height came out between three and five metres.
On twelve real falls it fired zero times.

**Version two asked how long the body was when flattened onto the floor.** That
saturates: standing and lying both gave about 2.2 m.

**Version three, which survives real footage**, asks where the head lands when it
is back-projected as if it were on the floor. Someone on the floor: 0.99 to 1.69 m
from their feet, which is a person. Someone standing: 2.5 to 5.3 m, or past the
horizon and therefore nowhere at all. Measured on UR Fall, cleanly separated, with
the fraction of joints that back-project sensibly as a second condition and the
bed and chair zones excluding the obvious confusion.

### 4.9 Hazards

Three, each visible from a ceiling corner and each classical OpenCV with no model
behind it:

- **a walking frame out of reach**: the distance in metres from where the person
  is sitting to the aid's registered zone, on the floor plane;
- **clutter on the route**: `absdiff` against the room as the ward signed it off,
  Otsu threshold, `morphologyEx`, `connectedComponentsWithStats`, keeping only
  components whose lowest point back-projects into the walking corridor with a
  real floor footprint above 400 square centimetres;
- **a wet-floor sign**: HSV `inRange` for a saturated yellow, then `convexHull`,
  `approxPolyDP` and a solidity and aspect test for an A-frame standing on the
  route.

A hazard never raises a state on its own, because a bag on the floor is not an
emergency while the person is asleep. It raises the escalation one rung when a
state is already in play, and never into urgent: a walking frame parked across the
room is a reason to send someone sooner, not a reason to tell a ward that someone
has fallen.

---

## 5. Privacy by construction

The claim is narrow and checkable: **in strict mode, no bytes derived from camera
pixels are written to disk, sent over the network, or held after the pose has been
read.** Not anonymised. Not blurred. Nothing leaves.

How it is enforced rather than promised:

1. `PoseEstimator.estimate` is the only code that touches an image. It returns
   keypoints. The caller then calls `guard.release(frame)`, which overwrites the
   buffer in place, so until numpy frees it the bytes in that page are not a
   picture of a patient.
2. Every write to disk goes through `PrivacyGuard.emit`, which requires a
   `Provenance`. `CAMERA` in strict mode raises. There is no flag on the call that
   overrides it.
3. `SYNTHETIC` artefacts are drawn by `render.py` from keypoints and polygons onto
   a blank canvas. **The renderer's signatures do not accept an image**, and a
   test asserts that by reflection, so there is no code path from a camera frame
   to a saved file.
4. `diagnostic` mode exists for a bench and raises unless an environment variable
   is set that the deployed service never sets.

### The code path, rather than the claim

There is one place where a frame becomes keypoints, and one line where it stops
existing. From `pipeline.Pipeline.run_video`:

```python
with stage("pose:total"):
    pose = self.estimator.estimate(image, frame.index, time_s)
view = self.view.grade(image, time_s, person_seen=pose is not None)
hazards = self.hazards.scan(image, time_s, ...)
# Everything the camera gave us has now been read. Destroy it.
self.guard.release(image)
```

and `PrivacyGuard.release`:

```python
def release(self, image: np.ndarray) -> None:
    if self.mode == "strict":
        try:
            image[...] = 0
        except (ValueError, TypeError):   # a read-only view; nothing to do
            pass
    else:
        self.ledger.frames_retained += 1
```

`image[...] = 0` is an in-place write to the one buffer, not a rebinding. There is
one frame in memory at a time, it is overwritten before the next decode, and no
reference to it survives the loop.

`tests/test_privacy.py` runs the whole pipeline with the real models over a real
video whose every frame is unique high-entropy content, then asserts zero camera
bytes persisted, zero frames retained, correlation below 0.35 between every
written file and every source frame, and that asking to write a frame raises.

The interface shows the same ledger the tests assert on: the seventeen keypoints
with their coordinates and scores, next to the figure they produced, above a line
saying how many camera bytes were written. The privacy claim is not a shield icon.
It is the raw payload, printed.

### The cost of privacy, published rather than hidden

Privacy has a price here and the product states it rather than absorbing it.

A ward will pause the camera for washing, toileting and dressing, and it should:
those are the activities where camera acceptance is lowest, and there is a button
for it. They are also, from section 1, among the activities where a large share of
falls happen. So a paused camera is a **first-class state** rather than an
absence. Every run reports `blind_by_choice_s` separately from `faulty_s`, the
interface shows it, and the record carries a `BLIND_BY_CHOICE` refusal whose
message says the window is **unknowable, not absent**.

The consequence has to be said plainly: **calls missed while the camera is paused
cannot be counted, by anyone, including us.** A detection rate measured over a
shift that was blind for three hours is not the same quantity as one measured over
a shift that watched all night, and this product publishes the denominator so that
the difference is visible rather than quietly folded in.

**One compression figure that makes the point.** A 1280x720 BGR frame is 2.76 MB.
The seventeen keypoints kept from it are 408 bytes. The ratio is about 6,800 to 1,
and what is discarded is all of the identity.

---

## 6. AWS deployment

Container to ECR, ECR to App Runner, 2 vCPU and 4 GB, always on, HTTPS with no
load balancer. `docs/costs.md` has the resources, the rates and the two decisions
worth recording, including why this is in us-east-2 rather than us-east-1.

Both ONNX models, seven bundled pose tracks and the default room are inside the
image, so the endpoint works from a cold start with **no network at runtime**. The
container answers `/healthz` two seconds after start and a bundled sample returns
in under a second.

The demonstration is a pose track rather than a video, and that is a product
decision rather than a shortcut. A pose track is exactly what a Preempt device
holds after the camera stage, so replaying one replays what the device really
keeps. Shipping a video of a patient in a bed would contradict the only claim the
product makes. A judge's own video upload still runs the whole pipeline.

---

## 7. Limitations

**Pose estimation on a person already on the floor is the binding constraint.**
Over 561 ground-truth lying frames, a person was detected in 93 per cent but only
68 per cent produced six or more usable joints. A detector trained mostly on
upright people sees a person lying down as an unfamiliar blob. Coasting on the
last box and a grace period on the floor state recover some of it; what would fix
it is a detector trained with supine and prone people in it.

**One camera, one person.** There is no multi-person tracking. A room with a
visitor and a patient in it will track the larger of the two. That is a real gap
for a shared bay, and it is a design choice for a side room.

**The metres are only as good as the setup.** With four measured floor points the
geometry is exact to a centimetre. With the walking-person calibration it is as
good as the assumed stature: assume 1.75 m for a 1.60 m patient and every height
reads about 9 per cent high.

**Heights fail when the feet are not on the floor.** Every height is measured from
the point where the body meets the floor. On a bed that point is a mattress, and
the heights inflate. The engine detects this from the zones a nurse drew and falls
back rather than reporting an inflated number.

**Bathrooms are not covered and this is where a large share of falls happen.**
Section 1 explains why. Preempt is a bedroom-and-bay product and the honest answer
about the bathroom is that a camera should not be in it, so a different sensing
modality is needed there.

**And the same is true of every minute the camera is paused.** Section 5 has the
measurement; the limitation is that nobody, including this evaluation, can say
what was missed in those windows.

**The false-alarm rate has a small denominator.** Zero false alarms in 1.21 hours
of observed quiet is real but it is 1.21 hours. Nothing here is a bed-night.

**No outcome evidence exists.** No deployed-system outcome trial could be found
for any camera safety product in this domain. This has not been shown to reduce
falls on a real ward, and neither has anything comparable.

**Crouching reads like lying.** Two of ten activity sequences with no lying frames
said "on the floor" when the subject bent down. The bound that separates them is
narrow and it is written where somebody tuning the product will read it.

---

## 8. Responsible use, and the questions a ward would ask

**"Can anyone watch the camera?"** No. There is no video output, no recording and
no stream. The only images the system produces are drawn from keypoints.

**"Does it do face recognition?"** No, and it cannot. The pipeline has no face
model, no embedding and no identity of any kind. The seventeen keypoints include
the nose, eyes and ears as *positions*, which is enough to tell where a head is
and not remotely enough to tell whose it is. Nothing in this repository may add a
face model to this product.

**"What happens to the video?"** It is destroyed after the pose is read, in the
same function call. In strict mode the buffer is overwritten. The interface prints
the byte count.

**"Who gets told, and in what order?"** The person in the room first, quietly.
Then the station. An urgent call only when someone is on the floor. A nudge that
nobody answers becomes a station call after twenty seconds, which is what makes
the quiet rung safe.

**"What if a patient says no?"** Then the camera comes out of that room. Consent
has to be real, which means it has to be refusable without argument and revisited,
because capacity changes. A monitoring system installed over an objection is worse
than none, because the ward loses the trust it needs for everything else.

**"What about a patient who cannot consent?"** A best-interests decision under the
usual framework, documented, with the family involved, and revisited. The
strongest thing this design offers that conversation is that there is nothing to
show anybody: the question is not "who may watch this person" but "may the ward be
told when this person starts to get up", which is a far easier question for a
family to answer.

**"Will staff be measured on it?"** They must not be. Nothing here counts response
times or names individuals, and that restriction is architectural: the record
carries a room and a state, never a member of staff. A system that watches staff
gets unplugged, and it should be.

**"What if it is wrong?"** It says which state it is in, the reasons that raised
it, and how certain it is, in words. It says "observed movement, not a prediction
of a fall" on the screen. And when it cannot see, it says that instead of showing
a calm green light, which is the failure that would matter most.

**"Is it a medical device?"** It is a monitoring and alerting aid. It makes no
diagnosis, gives no treatment advice and takes no clinical action. It tells a
human being to walk down the corridor and look. Anything beyond that would need a
regulatory route this project does not claim to have.

---

## 9. Reproducing this

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e packages/visioncore -e packages/servicekit -e products/preempt
products/preempt/models/fetch.sh
.venv/bin/python -m pytest products/preempt/tests -q          # 102 tests
.venv/bin/python -m preempt.cli evaluate --seeds 5 --quiet-loops 16
products/preempt/eval/fetch_urfall.sh 12
.venv/bin/python -m preempt.cli urfall products/preempt/eval/data --room products/preempt/eval/urfall-room.json
```

Pinned dependencies are in `constraints.txt` and in each package's
`pyproject.toml`. The Docker build asserts the OpenCV major version, and
`/version` on the live endpoint prints what is running.
