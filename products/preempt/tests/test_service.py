"""The web service: the routes a judge and the UI actually hit."""

from __future__ import annotations

import cv2
import pytest
from fastapi.testclient import TestClient

from preempt.service import create


@pytest.fixture(scope="module")
def client():
    with TestClient(create()) as test_client:
        yield test_client


def test_health_and_version_report_opencv_5(client):
    assert client.get("/healthz").json()["status"] == "ok"
    version = client.get("/version").json()
    assert version["opencv_version"].startswith("5."), version["opencv_version"]
    assert version["product"]["slug"] == "preempt"
    assert "Apache-2.0" in version["models"]["pose"]["licence"]


def test_the_index_page_is_the_products_own(client):
    body = client.get("/").text
    assert "Preempt" in body
    assert "<html" in body.lower()


def test_the_bundled_samples_are_listed_with_descriptions(client):
    samples = client.get("/api/samples").json()["samples"]
    assert samples, "no bundled samples, so a cold start shows a judge nothing"
    names = {s["name"] for s in samples}
    assert "bed-exit-steady" in names
    assert "curtain-drawn" in names, "the failure case must be one of the samples"
    for sample in samples:
        assert sample["description"], f"{sample['name']} has no description"


def test_running_a_bundled_sample_produces_a_call_with_a_lead_time(client):
    started = client.post("/api/samples/bed-exit-steady")
    assert started.status_code == 202
    job_id = started.json()["job_id"]
    result = _await_job(client, job_id)
    metrics = result["metrics"]
    assert metrics["peak_state"] == "rising soon"
    assert metrics["lead_time_s"] > 1.5
    assert metrics["privacy"]["camera_bytes_persisted"] == 0
    assert metrics["privacy"]["frames_retained"] == 0
    kinds = {r["kind"] for r in result["results"]}
    assert {"risk", "calls", "hazards", "timeline", "keypoint-ledger"} <= kinds


def test_the_failure_sample_returns_a_refusal_rather_than_an_error(client):
    job_id = client.post("/api/samples/curtain-drawn").json()["job_id"]
    result = _await_job(client, job_id)
    codes = {r["code"] for r in result["refusals"]}
    assert "VIEW_UNUSABLE" in codes
    assert result["refused"] is True
    assert result["metrics"]["view"]["usable_fraction"] < 1.0


def test_the_evidence_endpoint_serves_the_drawn_cards(client):
    job_id = client.post("/api/samples/bed-exit-steady").json()["job_id"]
    result = _await_job(client, job_id)
    assert result["evidence"], "no evidence was attached"
    for item in result["evidence"]:
        response = client.get(item["uri"])
        assert response.status_code == 200
        assert response.content[:4] == b"\x89PNG"


def test_an_unknown_sample_is_a_clean_not_found(client):
    response = client.post("/api/samples/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_an_unsupported_upload_is_refused_with_the_accepted_list(client):
    response = client.post(
        "/api/jobs",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"params": "{}"},
    )
    assert response.status_code == 415
    assert ".mp4" in response.json()["error"]["details"]["accepted"]


def test_an_uploaded_video_runs_the_whole_pipeline(client, tmp_path):
    import numpy as np

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (320, 240))
    rng = np.random.default_rng(5)
    for _ in range(12):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    response = client.post(
        "/api/jobs",
        files={"file": ("clip.mp4", path.read_bytes(), "video/mp4")},
        data={"params": "{}"},
    )
    assert response.status_code == 202
    result = _await_job(client, response.json()["job_id"])
    assert result["metrics"]["privacy"]["camera_bytes_read"] > 0
    assert result["metrics"]["privacy"]["camera_bytes_persisted"] == 0


def _await_job(client, job_id: str, tries: int = 400) -> dict:
    import time

    for _ in range(tries):
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] == "done":
            return payload["result"]
        if payload["status"] == "failed":
            raise AssertionError(f"job failed: {payload['error']}")
        time.sleep(0.05)
    raise AssertionError("job did not finish")
