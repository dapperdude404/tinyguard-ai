# SmartPlant AI — IoT Cloud Monitoring (Optional Phase 3 MVP)

**Status: draft implementation, not deployed. No public cloud endpoint or APK is published.**

This extends, rather than replaces, the existing 29-class offline MobileNetV3-Small Android Edge AI application.

## Data flow

```text
Android phone camera (local leaf image)
    -> offline ONNX CNN inference (no network required)
    -> local prediction + private scan history
    -> USER EXPLICITLY TAPS "Sync latest scan"
    -> HTTPS POST /v1/scans (Bearer ingest token)
    -> SQLite scan metadata store
    -> GET /v1/scans + /v1/summary (separate read token)
    -> Streamlit monitoring dashboard (server-side API requests)
```

**No image content, GPS, contact information, raw pixels or identifying device ID is sent.** The JSON payload holds a random scan UUID, capture timestamp, optional user-selected crop, the CNN top predicted label, top score, and the top five original class scores.

## Why this is demonstrably IoT + Edge AI

The Android phone is the sensing/edge-computing node. It captures plant images, runs the model locally, and—only after consent—communicates structured monitoring observations via network to a remote server. The dashboard aggregates those observations. Offline inference and local history continue even when cloud connectivity is unavailable. A phone can serve as the IoT endpoint; ESP32/Raspberry Pi hardware is not necessary.

## Start the API locally for development

Python 3.11 recommended:

```bash
python -m pip install -r requirements-iot.txt
export SMARTPLANT_IOT_INGEST_TOKEN="<generate-a-strong-random-token>"
export SMARTPLANT_IOT_READ_TOKEN="<generate-a-different-strong-token>"
export SMARTPLANT_IOT_DB_PATH="./iot_cloud/data/scans.db"
uvicorn iot_cloud.api:app --host 127.0.0.1 --port 8000
```

On Windows PowerShell use `$env:SMARTPLANT_IOT_INGEST_TOKEN="..."` and analogous commands. Generate tokens with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. **Never commit either token to GitHub.** The default cloud API deliberately refuses scan operations if the necessary token is not configured.

Run the dashboard in a second terminal:

```bash
python -m pip install streamlit pandas
streamlit run iot_cloud/dashboard.py
```

Enter `http://localhost:8000` and the **read token** in the dashboard for local development. This dashboard uses server-side Python requests to the API, so browsers do not need CORS credentials.

### Example API request (metadata only)

```bash
curl -X POST "http://localhost:8000/v1/scans" \
  -H "Authorization: Bearer YOUR_INGEST_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"scan_id":"23598f0f-1055-45f3-92d5-08d0bb739791","captured_at_ms":1795215000000,"selected_crop":"Potato","top_label":"Potato | Early Blight","top_probability":0.995,"top_five":[{"label":"Potato | Early Blight","probability":0.995},{"label":"Tomato | Late Blight","probability":0.005}]}'
```

This is a sample request, **not a model accuracy claim**.

## Android testing with a remote HTTPS API

The app intentionally rejects plain HTTP. A physical phone cannot sync to a laptop's local `localhost` API. For full IoT functionality, the owner must deploy the API behind a trusted public or private HTTPS endpoint, configure environment variables and persistent storage, and share the **ingest token** directly with the Android app's explicit sync dialog. The Android app does not store the token; it remembers only the API base URL.

**Do not deploy this experimental service as-is to a public address without additional controls**, including per-device credentials, token rotation, audit logs, rate limiting, persistent encrypted database backups, deletion/retention policy, and operational monitoring. SQLite is suitable for a classroom demonstration but a multi-user production deployment should use a managed database.

## API security details

- Requests require separate `SMARTPLANT_IOT_INGEST_TOKEN` (write) and `SMARTPLANT_IOT_READ_TOKEN` (read/delete); missing tokens fail closed.
- Raw images and unknown JSON fields are rejected by Pydantic's `extra="forbid"`.
- Exactly one scan UUID identifies a record; replaying the same ID is idempotent.
- Limit top-five labels and validate the probabilities.
- Dashboard only reads the metadata using the read-only purpose token. The API's read token currently also permits deletion; grant this only to trusted administrative users.
- Monitoring flags mean **review suggested**, not verified plant disease.
- A scan can be removed with `DELETE /v1/scans/{uuid}` using the read/admin token.
- User-triggered sync is not a persistent offline queue: if the phone has no internet, the user must retry while the scan is still open.

## Testing and acceptance gates

```bash
python -m pytest -q tests/test_iot_cloud.py tests/test_iot_android.py
```

Before merging, check all GitHub tests, Android compilation and model parity. Before deploying, demonstrate: offline image prediction still works with airplane mode; no traffic occurs without pressing Sync; HTTPS + bad token fails; good token creates only metadata; dashboard displays the record and crop mismatch alert; repeat sync does not duplicate; delete removes the record. Record actual screenshots and network testing evidence.

This is a **networked monitoring extension**, not a second cloud disease classifier.
