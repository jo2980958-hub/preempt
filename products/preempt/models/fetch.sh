#!/usr/bin/env bash
# Fetch the two ONNX models Preempt runs, into this directory.
#
# Both are permissively licensed and both load in `cv2.dnn` with no second
# runtime. Neither is committed: they are 33 MB of binary and the licences are
# better served by a URL and a checksum than by a copy.
#
#   RTMPose-t (body7, 256x192)  Apache-2.0, OpenMMLab / MMPose
#   YOLOX-tiny (416x416)        Apache-2.0, Megvii
#
# Ultralytics' pose models are AGPL-3.0, whose section 13 makes a hosted demo a
# source-disclosure event. They are not used here and must not be added.
set -euo pipefail
cd "$(dirname "$0")"

RTMPOSE_ZIP="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip"
YOLOX_URL="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx"

RTMPOSE_SHA="a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f"
YOLOX_SHA="427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"

if [[ ! -f rtmpose-t-body7.onnx ]]; then
  echo "fetching RTMPose-t (Apache-2.0, OpenMMLab)"
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/rtmpose.zip" "$RTMPOSE_ZIP"
  unzip -q -o "$tmp/rtmpose.zip" -d "$tmp"
  find "$tmp" -name end2end.onnx -exec cp {} rtmpose-t-body7.onnx \;
  rm -rf "$tmp"
fi

if [[ ! -f yolox_tiny.onnx ]]; then
  echo "fetching YOLOX-tiny (Apache-2.0, Megvii)"
  curl -fsSL -o yolox_tiny.onnx "$YOLOX_URL"
  echo "$YOLOX_SHA  yolox_tiny.onnx" | sha256sum -c -
fi

ls -l ./*.onnx
