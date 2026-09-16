# Preempt: a call before the fall.
#
# Two stages so the runtime image carries no build tools. OpenCV 5 is pinned in
# visioncore's pyproject.toml and again in this product's - do not add an
# unpinned opencv line anywhere, because `pip install opencv-python` resolves to
# 4.14.x, which shipped after 5.0.0 and would fail the competition's core
# requirement. The build asserts the version rather than trusting it.
#
# The two ONNX models are baked in. That is deliberate: the endpoint has to work
# from a cold start with no network, and 33 MB of Apache-2.0 weights is a cheap
# price for a judge seeing the loop in seconds. Nothing from Ultralytics is here
# and nothing from Ultralytics may be added: AGPL-3.0 section 13 makes a hosted
# demo a source-disclosure event.
#
# Build from the REPO ROOT so the shared packages are in the context:
#   infra/ecr.sh preempt --context . --dockerfile products/preempt/Dockerfile

FROM python:3.13-slim-bookworm AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl unzip ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY constraints.txt ./
COPY packages/visioncore ./packages/visioncore
COPY packages/servicekit ./packages/servicekit
COPY products/preempt ./products/preempt

RUN pip install --constraint constraints.txt \
      ./packages/visioncore ./packages/servicekit ./products/preempt \
 && python -c "import cv2, sys; v=cv2.__version__; sys.exit(0 if v.startswith('5.') else f'got OpenCV {v}, need 5.x')"

# RTMPose-t (Apache-2.0, OpenMMLab) and YOLOX-tiny (Apache-2.0, Megvii).
RUN products/preempt/models/fetch.sh

# ---------------------------------------------------------------------------

FROM python:3.13-slim-bookworm AS runtime

# opencv-python-headless still wants libGL's friends for a few video codecs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    OPENCV26_UPLOAD_DIR=/tmp/uploads \
    PREEMPT_SAMPLES_DIR=/opt/preempt/samples \
    PREEMPT_ROOM=/opt/preempt/rooms/default.json \
    PREEMPT_RTMPOSE_URI=/opt/preempt/models/rtmpose-t-body7.onnx \
    OPENCV26_YOLOX_URI=/opt/preempt/models/yolox_tiny.onnx

COPY --from=build /build/products/preempt/models/rtmpose-t-body7.onnx /opt/preempt/models/
COPY --from=build /build/products/preempt/models/yolox_tiny.onnx /opt/preempt/models/
COPY products/preempt/samples /opt/preempt/samples
COPY products/preempt/rooms /opt/preempt/rooms
COPY products/preempt/web /opt/preempt/web

ARG GIT_SHA=unknown
ENV OPENCV26_GIT_SHA=${GIT_SHA}

RUN useradd --create-home --uid 10001 app && mkdir -p /tmp/uploads && chown app /tmp/uploads
USER app
WORKDIR /home/app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "preempt.main:app", "--host", "0.0.0.0", "--port", "8000"]
