"""Cloud API security, persistence and honest monitoring tests."""
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from iot_cloud.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    monkeypatch.setenv("SMARTPLANT_IOT_INGEST_TOKEN", "test-ingest-token-never-deploy")
    monkeypatch.setenv("SMARTPLANT_IOT_READ_TOKEN", "test-read-token-never-deploy")
    monkeypatch.setenv("SMARTPLANT_IOT_DB_PATH", str(tmp_path / "scans.db"))


def payload(**overrides):
    source = dict(
        scan_id=str(uuid4()),
        captured_at_ms=1795215000000,
        selected_crop="Potato",
        top_label="Strawberry | Leaf Scorch",
        top_probability=0.884,
        top_five=[
            {"label": "Strawberry | Leaf Scorch", "probability": 0.884},
            {"label": "Potato | Early Blight", "probability": 0.062},
            {"label": "Strawberry | Healthy", "probability": 0.049},
        ],
    )
    source.update(overrides)
    return source


def headers(write=True):
    return {"Authorization": "Bearer " + ("test-ingest-token-never-deploy" if write else "test-read-token-never-deploy")}


def test_health_public_without_private_data():
    assert client.get("/health").json() == {"service": "smartplant-iot", "status": "ok",
                                           "mode": "metadata-only"}


def test_unauthorized_requests_rejected():
    assert client.post("/v1/scans", json=payload()).status_code == 401
    assert client.get("/v1/scans").status_code == 401
    assert client.get("/v1/summary", headers=headers(write=True)).status_code == 401
    assert client.post("/v1/scans", headers=headers(write=False), json=payload()).status_code == 401


def test_missing_auth_configuration_fails_closed(monkeypatch):
    monkeypatch.delenv("SMARTPLANT_IOT_INGEST_TOKEN")
    assert client.post("/v1/scans", headers=headers(), json=payload()).status_code == 503


def test_metadata_only_and_input_bounds():
    assert client.post("/v1/scans", headers=headers(), json=payload(image_base64="secret image")).status_code == 422
    assert client.post("/v1/scans", headers=headers(), json=payload(gps="do not collect")).status_code == 422
    assert client.post("/v1/scans", headers=headers(), json=payload(top_probability=1.2)).status_code == 422
    assert client.post("/v1/scans", headers=headers(), json=payload(top_label="Potato | Healthy")).status_code == 422


def test_mismatch_flag_and_idempotency():
    data = payload()
    first = client.post("/v1/scans", headers=headers(), json=data)
    assert first.status_code == 201
    assert first.json()["saved"] and first.json()["review_flag"] == "crop_mismatch"
    again = client.post("/v1/scans", headers=headers(), json=data)
    assert again.status_code == 201
    assert again.json()["duplicate"] and not again.json()["saved"]
    result = client.get("/v1/scans", headers=headers(write=False))
    assert result.status_code == 200
    scans = result.json()["scans"]
    assert len(scans) == 1
    assert scans[0]["crop_mismatch"]
    assert scans[0]["top_five"][0]["probability"] == pytest.approx(0.884)
    assert "image" not in scans[0]
    assert "location" not in scans[0]
    summary = client.get("/v1/summary", headers=headers(write=False)).json()
    assert summary["total_scans"] == 1
    assert summary["crop_mismatch_count"] == 1
    assert summary["review_count"] == 1


def test_agreement_and_delete():
    data = payload(
        selected_crop="Potato", top_label="Potato | Healthy",
        top_probability=0.9,
        top_five=[
            {"label": "Potato | Healthy", "probability": 0.9},
            {"label": "Potato | Early Blight", "probability": 0.05},
        ]
    )
    resp = client.post("/v1/scans", headers=headers(), json=data)
    assert resp.status_code == 201
    assert resp.json()["review_flag"] == "none"
    deleted = client.delete("/v1/scans/" + data["scan_id"], headers=headers(write=False))
    assert deleted.status_code == 200 and deleted.json()["deleted"]
    assert client.get("/v1/summary", headers=headers(write=False)).json()["total_scans"] == 0


def test_limit_validation():
    assert client.get("/v1/scans?limit=10000", headers=headers(write=False)).status_code == 422
