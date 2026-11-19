# TinyGuard AI — Deployment and Classroom Readiness

## Cloud deployment
1. Verify the GitHub repository has `app.py`, `requirements.txt`, `tinyguard/core.py`, and `.streamlit/config.toml` on `main`.
2. Open https://share.streamlit.io and connect the GitHub repository `dapperdude404/tinyguard-ai`.
3. Select `main`, application path `app.py`, and deploy.
4. Open the resulting URL in a private browser window to check public access.
5. Test all four dashboard tabs, a synthetic dataset, the CSV upload, and CSV download.
6. Keep screenshots or a local fallback ready if the classroom network is unavailable.

## Local verification
```bash
python -m venv .venv
# Activate the virtual environment for your operating system
pip install -r requirements-dev.txt
python -m pytest -q
python benchmark.py --seeds 3 --output results/benchmark.csv
streamlit run app.py
```

## Optional TinyML export
```bash
pip install 'tensorflow>=2.16'
python export_tflite.py
```

The exported TFLite artifact requires quantized-inference threshold recalibration, microcontroller operator compatibility testing, and device-specific flash/RAM/latency/power measurements before any TinyML deployment claim.

## Research presentation checklist
- Describe the synthetic dataset and injected anomalies accurately.
- Show a normal and abnormal input in the live demo.
- Explain that anomaly scores are not calibrated probabilities.
- Use benchmark results computed from the checked-in code; do not invent accuracy claims.
- State explicitly that the sklearn dashboard model and exported Keras model are different models.
- Identify the target microcontroller and obtain actual hardware measurements for a stronger M.Tech evaluation.

## Current limitations
The GitHub source is not itself a deployed Streamlit app. The original ZIP, DOCX and PPTX have not been committed to the repository. No live browser smoke test or on-device TinyML benchmark has been verified.
