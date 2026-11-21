"""SmartPlant IoT metadata API (prototype, not a public production deployment).

Data flow: user-approved Android summary -> HTTPS POST -> token-authenticated SQLite
-> read-token protected dashboard. Leaf photos never enter this service.
"""
from __future__ import annotations

import hmac
import json
import os
import sqlite3
import time
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

app = FastAPI(title="SmartPlant AI IoT Monitoring API", version="0.1.0",
              description="Opt-in metadata ingestion. No image upload or disease diagnosis.")


class ScoredLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=160)
    probability: float = Field(ge=0.0, le=1.0)


class ScanIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scan_id: UUID
    captured_at_ms: int = Field(gt=0)
    selected_crop: str | None = Field(default=None, max_length=80)
    top_label: str = Field(min_length=1, max_length=160)
    top_probability: float = Field(ge=0.0, le=1.0)
    top_five: list[ScoredLabel] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_scores(self):
        if self.top_label != self.top_five[0].label:
            raise ValueError("top_label must equal first ranked label")
        if abs(self.top_probability - self.top_five[0].probability) > 1e-5:
            raise ValueError("top_probability must equal first ranked probability")
        scores = [x.probability for x in self.top_five]
        if any(scores[i] < scores[i+1] for i in range(len(scores)-1)):
            raise ValueError("top_five must be in descending probability order")
        if sum(scores) > 1.001:
            raise ValueError("top_five probabilities must not exceed 1")
        return self


def _auth(secret_name: str, authorization: str | None) -> None:
    expected = os.environ.get(secret_name, "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Authentication not configured")
    scheme, _, credential = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(credential.strip(), expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _db_path() -> Path:
    return Path(os.environ.get("SMARTPLANT_IOT_DB_PATH", "iot_cloud/data/scans.db"))


def _connect():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans(
            scan_id TEXT PRIMARY KEY,
            captured_at_ms INTEGER NOT NULL,
            received_at_ms INTEGER NOT NULL,
            selected_crop TEXT,
            top_label TEXT NOT NULL,
            top_probability REAL NOT NULL,
            crop_mismatch INTEGER NOT NULL,
            review_flag TEXT NOT NULL,
            top_five_json TEXT NOT NULL
        )
    """)
    return conn


def _label_crop(label: str) -> str:
    return label.split("|", 1)[0].strip()


def _review_flag(top_label: str, crop_mismatch: bool) -> str:
    if crop_mismatch:
        return "crop_mismatch"
    # Monitoring flag only. It is NOT a confirmed disease alert.
    if "healthy" not in top_label.lower():
        return "review_prediction"
    return "none"


def _row_dict(row: sqlite3.Row) -> dict:
    record = dict(row)
    record["crop_mismatch"] = bool(record["crop_mismatch"])
    record["top_five"] = json.loads(record.pop("top_five_json"))
    return record


@app.get("/health")
def health():
    return {"service": "smartplant-iot", "status": "ok", "mode": "metadata-only"}


@app.post("/v1/scans", status_code=201)
def ingest_scan(scan: ScanIn, authorization: str | None = Header(default=None)):
    _auth("SMARTPLANT_IOT_INGEST_TOKEN", authorization)
    crop = scan.selected_crop if scan.selected_crop not in ("", "Auto / unknown") else None
    mismatch = bool(crop and crop.lower() != _label_crop(scan.top_label).lower())
    flagged = _review_flag(scan.top_label, mismatch)
    with _connect() as db:
        cursor = db.execute(
            """INSERT OR IGNORE INTO scans
            (scan_id, captured_at_ms, received_at_ms, selected_crop, top_label,
             top_probability, crop_mismatch, review_flag, top_five_json)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (str(scan.scan_id), scan.captured_at_ms, int(time.time() * 1000),
             crop, scan.top_label, scan.top_probability, int(mismatch), flagged,
             json.dumps([x.model_dump() for x in scan.top_five])),
        )
        inserted = cursor.rowcount == 1
        db.commit()
    return {"scan_id": str(scan.scan_id), "saved": inserted, "duplicate": not inserted,
            "review_flag": flagged}


@app.get("/v1/scans")
def list_scans(limit: int = Query(default=100, ge=1, le=500),
               authorization: str | None = Header(default=None)):
    _auth("SMARTPLANT_IOT_READ_TOKEN", authorization)
    with _connect() as db:
        records = db.execute(
            "SELECT * FROM scans ORDER BY received_at_ms DESC, scan_id DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"scans": [_row_dict(row) for row in records]}


@app.get("/v1/summary")
def get_summary(authorization: str | None = Header(default=None)):
    _auth("SMARTPLANT_IOT_READ_TOKEN", authorization)
    with _connect() as db:
        row = db.execute("""
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(crop_mismatch),0) AS crop_mismatch_count,
                   COALESCE(SUM(CASE WHEN review_flag != 'none' THEN 1 ELSE 0 END),0)
                        AS review_count
            FROM scans
        """).fetchone()
        top = db.execute("""
            SELECT top_label, COUNT(*) as count FROM scans
            GROUP BY top_label ORDER BY count DESC LIMIT 10
        """).fetchall()
    return {"total_scans": row["total"], "crop_mismatch_count": row["crop_mismatch_count"],
            "review_count": row["review_count"],
            "top_predicted_labels": [dict(x) for x in top]}


@app.delete("/v1/scans/{scan_id}")
def delete_scan(scan_id: UUID, authorization: str | None = Header(default=None)):
    _auth("SMARTPLANT_IOT_READ_TOKEN", authorization)
    with _connect() as db:
        cursor = db.execute("DELETE FROM scans WHERE scan_id = ?", (str(scan_id),))
        db.commit()
    return {"deleted": cursor.rowcount == 1, "scan_id": str(scan_id)}
