# 🌿 SmartPlant AI — M.Tech AI Research Edition

**A modern, explainable plant-leaf image analysis and disease-classification research prototype.**

[Original TinyGuard anomaly detection project (preserved)](https://github.com/dapperdude404/tinyguard-ai/tree/archive/tinyguard-anomaly-v1)

## What works now
- Mobile-friendly Streamlit dashboard with image upload and preprocessing
- Visible green/yellow/brown pixel indicators (descriptive, **not a diagnosis**)
- HOG texture and RGB histogram feature extraction
- Reproducible Random Forest training on a supplied labeled leaf dataset
- Ranked disease-class predictions **when a trained model is installed**
- Automated tests, research methodology and classroom presentation script

**Important:** No trained disease classifier or disease dataset is bundled. The app deliberately does not invent a disease prediction without a model. This is a functional research prototype, not a validated agricultural diagnostic product.

## Run online or locally
Use Python 3.11.

```bash
python -m venv .venv
# Activate your environment
pip install -r requirements.txt
streamlit run app.py
```

## Train on real labeled images
Organize your dataset as:

```text
data/train/
  Healthy/
    leaf_001.jpg
    leaf_002.jpg
  Tomato_Early_blight/
    leaf_003.jpg
  Tomato_Late_blight/
    leaf_004.jpg
```

Each class needs **at least five** images; significantly more are required for meaningful research.

```bash
python train.py --data data/train --output models/plant_model.joblib
```

This writes `models/plant_model.joblib` and `models/plant_model.metrics.json`. The `models/` folder is gitignored: only load trusted model artifacts, and avoid uploading third-party datasets without checking licensing.

For a deployed classifier, make a trusted model artifact available to the deployed environment through a controlled release/build workflow. The public app runs in educational image-inspection mode until then.

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Deploy
Open [Streamlit Community Cloud](https://share.streamlit.io), select `dapperdude404/tinyguard-ai`, branch `main`, entrypoint `app.py`. The existing Streamlit deployment may automatically rebuild from the updated main branch; confirm the app status and logs.

## Classroom and research
- [Research methodology](docs/SMARTPLANT_RESEARCH.md)
- [8–10 minute classroom presentation script](docs/SMARTPLANT_CLASSROOM.md)
- [Preserved TinyGuard project](https://github.com/dapperdude404/tinyguard-ai/tree/archive/tinyguard-anomaly-v1)

## Research ethics
Validation accuracy on a random image split can be misleading if photographs of the same leaf appear in both partitions. Use independent plant/farm-aware test sets and report macro F1 and confusion matrices. Predictions and color indicators are not agricultural treatment advice.

## Version
SmartPlant AI V1 research baseline. Former TinyGuard code remains in the archival branch and some legacy files on main; they are not part of the new app entrypoint.
