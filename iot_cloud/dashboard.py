"""SmartPlant IoT dashboard. Run separately: streamlit run iot_cloud/dashboard.py

Connects server-side to a separately deployed FastAPI instance using its READ token.
No photos, GPS, device identifier, credentials or raw leaf pixels are stored in dashboard.
"""
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SmartPlant | IoT Monitoring", page_icon="🌿", layout="wide")
st.title("🌿 SmartPlant IoT Cloud Monitoring")
st.caption("Smart agriculture research · opt-in scan metadata only · no leaf photographs")
st.info("AI runs on the Android phone. This dashboard shows records voluntarily synced over HTTPS. "
        "Predictions and monitoring flags are not confirmed diagnoses.")

with st.sidebar:
    st.header("Secure connection")
    base = st.text_input("IoT API base URL", placeholder="https://your-secure-api.example.com")
    token = st.text_input("Dashboard read token", type="password")
    limit = st.slider("Number of recent records", 10, 200, 50)
    refresh = st.button("Refresh", use_container_width=True)

if not base or not token:
    st.warning("Enter your HTTPS IoT API address and its read-only dashboard token to view authorized results.")
    st.stop()

url = base.rstrip("/")
if not url.startswith("https://") and not url.startswith("http://localhost:"):
    st.error("A secure HTTPS API is required (localhost is allowed during local development).")
    st.stop()

headers = {"Authorization": "Bearer " + token}
try:
    summary_response = requests.get(url + "/v1/summary", headers=headers, timeout=10)
    scans_response = requests.get(url + "/v1/scans", headers=headers,
                                  params={"limit": limit}, timeout=10)
    summary_response.raise_for_status()
    scans_response.raise_for_status()
    summary = summary_response.json()
    scans = scans_response.json()["scans"]
except (requests.RequestException, KeyError, ValueError) as exc:
    st.error("Unable to read authorized monitoring data. Check API availability and credentials.")
    st.caption(type(exc).__name__)
    st.stop()

a, b, c = st.columns(3)
a.metric("Authorized scans", summary["total_scans"])
b.metric("Crop mismatches", summary["crop_mismatch_count"])
c.metric("Review flags", summary["review_count"])

if not scans:
    st.success("Connected. No synced scans yet.")
    st.stop()

df = pd.DataFrame(scans)
df["Time (UTC)"] = df["received_at_ms"].apply(
    lambda ts: datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"))
df["Crop review"] = df["crop_mismatch"].map({True: "Mismatch", False: "No mismatch"})
df["Probability (%)"] = (df["top_probability"] * 100).round(1)
st.subheader("Recent scans")
st.dataframe(df[["Time (UTC)", "selected_crop", "top_label", "Probability (%)", "Crop review",
                 "review_flag"]].rename(columns={"selected_crop": "Selected crop",
                                                 "top_label": "Model predicted category",
                                                 "review_flag": "Review flag"}),
             use_container_width=True, hide_index=True)

st.subheader("Most frequent predicted categories")
counts = pd.DataFrame(summary.get("top_predicted_labels", []))
if not counts.empty:
    st.bar_chart(counts.set_index("top_label")["count"], horizontal=True)

mismatches = df[df["crop_mismatch"]]
if not mismatches.empty:
    st.warning(f"{len(mismatches)} displayed scan(s) have a mismatch between the user's selected crop "
               "and the model's top predicted crop. Review the source photographs locally.")

st.caption("Data are research observations. Classification probabilities are not calibrated confidence; "
           "a flag only requests human review. Use the authenticated API's DELETE endpoint to remove a scan.")
