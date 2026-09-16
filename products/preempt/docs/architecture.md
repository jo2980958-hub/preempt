# Architecture

Two diagrams: what happens to a frame, and what runs on AWS.

## 1. The pipeline

The important thing in this diagram is the red box. Every stage to its right
works on seventeen (x, y, score) triples, and no stage to its right can be handed
an image, because none of their signatures accept one.

```mermaid
flowchart TB
    subgraph device["On the device, in the room"]
        CAM[["Camera"]] --> DEC["cv2.VideoCapture<br/>decode, decimate to 15 Hz"]
        DEC --> VIEW["View grade<br/>cv2.calcHist · cv2.Laplacian<br/>cv2.ORB + BFMatcher"]
        DEC --> DET["Person box<br/>YOLOX-tiny in cv2.dnn<br/>Apache-2.0, ONNX"]
        DET --> POSE["17 keypoints<br/>RTMPose-t in cv2.dnn<br/>SimCC decode"]
        POSE --> DESTROY["<b>Frame destroyed</b><br/>buffer zeroed in place"]
        DET -. "detector lost them" .-> COAST["Coast on the last box<br/>up to 6 frames"]
        COAST --> POSE
    end

    DESTROY --> KP(["17 x (x, y, score)<br/>1 632 bytes a frame"])

    subgraph reason["Reasoning, on pose alone"]
        KP --> KIN["Kinematics<br/>floor homography · single-view heights<br/>cv2.KalmanFilter on the floor track"]
        KIN --> EXIT["Bed and chair exit<br/>state machine"]
        KIN --> GAIT["Gait window, 4 s<br/>cv2.fitLine · cv2.dft · cv2.filter2D"]
        KIN --> FLOOR["On the floor<br/>back-projection onto the floor plane"]
    end

    subgraph hazard["Hazards, every 2 s"]
        DEC --> HAZ["cv2.absdiff · Otsu · morphologyEx<br/>connectedComponentsWithStats<br/>cv2.inRange · approxPolyDP"]
    end

    EXIT --> RISK{{"Risk state<br/>settled · watch · rising soon<br/>unsteady · on the floor"}}
    GAIT --> RISK
    FLOOR --> RISK
    HAZ --> RISK
    VIEW --> RISK

    RISK --> LADDER["Escalation ladder"]
    LADDER --> NUDGE["Nudge<br/>in-room prompt"]
    LADDER --> STATION["Station call"]
    LADDER --> URGENT["Urgent<br/>someone is on the floor"]
    VIEW -. "cannot see" .-> MAINT["Maintenance notice<br/>and no clinical call"]

    RISK --> RENDER["Evidence<br/>stick figure on a room plan<br/>cv2.FontFace · drawn from keypoints"]
    RENDER --> REC[("RunRecord JSON<br/>+ two PNGs")]

    style DESTROY fill:#B0174A,color:#fff,stroke:#B0174A
    style KP fill:#EFEDF7,stroke:#3B3390
    style device fill:#F7F6FB,stroke:#5A5780
```

### Why each OpenCV 5 call is where it is

| Stage | OpenCV 5 | Why this and not something else |
|---|---|---|
| Decode | `VideoCapture` | OpenCV 5 returns **-1** for an unsupported property where 4.x returned 0; `visioncore.imageio` treats anything below zero as missing. |
| Detect | `dnn.readNetFromONNX(engine=ENGINE_NEW)` | `readNetFromCaffe` and `readNetFromDarknet` are **gone** in OpenCV 5. Everything is ONNX now. The new graph engine is 1.6x faster on YOLOX than the classic one at the same call site. |
| Pose | the same net, SimCC head | Two outputs, so `net.forward(names)` rather than `net.forward()`. |
| Floor | `findHomography`, `perspectiveTransform` | Not `solvePnP` with `SOLVEPNP_IPPE_SQUARE`, which returns an ambiguous solution: the foundation agent measured it reporting 2 degrees for a true 20 degree tilt. |
| Zones | `pointPolygonTest` | Signed distance, so "0.4 m outside the bed" is one call. |
| Track | `KalmanFilter` | Keypoints are noisy at the centimetre level and a raw difference of two noisy positions at 15 Hz is almost pure noise. A filter also predicts through a dropout where a difference spikes. |
| Gait | `fitLine`, `getGaussianKernel` + `filter2D`, `dft` | The walking line is fitted, so walking round a corner is not scored as sway. |
| View | `calcHist`, `Laplacian`, `ORB` + `BFMatcher` | Darkness, a blocked lens and a knocked camera are three different failures and get three different names. |
| Hazards | `absdiff`, Otsu `threshold`, `morphologyEx`, `connectedComponentsWithStats`, `inRange`, `approxPolyDP`, `convexHull` | Classical, explainable, and every number traceable to a pixel count and a homography. |
| Evidence | `FontFace` + `putText`, `polylines`, `fillPoly` | OpenCV 5's TrueType text engine, so a ward screenshot does not look like a 1990s machine-vision demo. |

## 2. The geometry, in one picture

A single homography between the image and the floor gives a metric frame. With a
vertical reference rescaled so one unit is one metre, `p ~ H·b + h·vz`, which
gives a height when the floor position is known and a floor position when the
height is known.

```mermaid
flowchart LR
    SETUP["Four floor points<br/>+ one vertical of known height"] --> H["H : floor metres to pixels"]
    SETUP --> VZ["vz scaled so<br/>one unit is one metre"]
    H --> HORIZON["Horizon<br/>l = H⁻ᵀ(0,0,1)"]
    H --> BACK["Back-projection<br/>pixel to floor metres"]
    VZ --> HEIGHT["Height above the floor<br/>from a base on the floor"]
    VZ --> SHADOW["Floor shadow<br/>of a point of known height"]
    HORIZON --> REFUSE["Above the horizon:<br/>refuse, do not guess"]
    BACK --> ONFLOOR["Head back-projects<br/>0.55 to 1.75 m from the feet<br/>= on the floor"]
    HEIGHT --> POSTURE["Hip height<br/>= seated or standing"]
    SHADOW --> TORSO["Torso position on the floor<br/>= the gait track"]

    ALT["No vertical reference?"] --> NOHEIGHT["Heights refused with a reason.<br/>Back-projection still exact."]
    style REFUSE fill:#EFEDF7,stroke:#B0174A
    style NOHEIGHT fill:#EFEDF7,stroke:#B0174A
```

A single view cannot place a point of unknown height, and cannot measure the
height of a point whose floor position is unknown. That is a real limitation and
it is not worked around: a point of unknown height is placed by assuming it is
above the person's foot contact, which is true to a few centimetres for a hip and
false for an outstretched hand, and the code says which of the two it is doing.

## 3. AWS

```mermaid
flowchart TB
    JUDGE(["Judge's browser"]) -->|HTTPS| AR

    subgraph aws["AWS, us-east-2, everything tagged Project=opencv26"]
        AR["App Runner<br/><b>opencv26-preempt</b><br/>2 vCPU · 4 GB · always on<br/>auto TLS, no load balancer"]
        ECR[("ECR<br/>opencv26/preempt<br/>651 MB image<br/>lifecycle: keep 10")]
        ROLE["IAM role<br/>opencv26-apprunner-ecr-access"]
        LOGS["CloudWatch Logs<br/>application + service"]
        ECR --> AR
        ROLE -.->|pull| ECR
        AR --> LOGS
    end

    subgraph image["Inside the image, no network needed at runtime"]
        UV["OpenCV 5.0.0.93 headless<br/>pinned, asserted at build"]
        M1["rtmpose-t-body7.onnx<br/>13 MB · Apache-2.0"]
        M2["yolox_tiny.onnx<br/>20 MB · Apache-2.0"]
        S["7 bundled pose tracks<br/>+ the default room"]
    end
    AR --- image

    style AR fill:#3B3390,color:#fff
    style image fill:#F7F6FB,stroke:#5A5780
```

**No load balancer, deliberately.** An idle ALB is about $16 a month for nothing,
and App Runner gives HTTPS and a stable hostname without one.

**No S3 in the request path, deliberately.** Everything a run produces is either a
few kilobytes of JSON or two drawn PNGs, and the privacy argument is easier to
make about a service that writes nothing durable at all. Uploads live in the
container's `/tmp` and expire with the job.

**us-east-2 rather than us-east-1**, because this account is limited to two App
Runner services per region and both us-east-1 slots were already taken by other
work. Preempt is the only one of the five entries outside us-east-1, so a
project-wide diagram would show four services in one region and this one in
another. Recorded in `docs/costs.md`.

## 4. The request, end to end

```mermaid
sequenceDiagram
    participant B as Browser
    participant A as App Runner
    participant J as Job store
    participant P as Pipeline

    B->>A: POST /api/samples/bed-exit-steady
    A->>J: create job, write the track to /tmp
    A-->>B: 202 {job_id, events_url}
    B->>A: GET /api/jobs/{id}/events  (SSE)
    J->>P: run on a worker thread
    loop every 20 samples
        P-->>B: progress
    end
    P->>P: draw two evidence PNGs from keypoints
    P-->>J: RunRecord
    J-->>B: end
    B->>A: GET /api/jobs/{id}
    A-->>B: the RunRecord
    B->>A: GET /api/jobs/{id}/evidence/risk-state.png
```

A judge's own video takes the same path and runs YOLOX and RTMPose over it.
`/version` reports the OpenCV version, the git sha and the model licences, so
what ran is checkable rather than asserted.
