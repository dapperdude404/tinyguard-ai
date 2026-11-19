# 🛡️ TinyGuard AI — M.Tech AI Research Edition

**Online anomaly detection dashboard and reproducible TinyML-oriented research prototype** for multivariate IoT sensor streams.

## Features
- Streamlit monitoring dashboard and manual anomaly-injection demo
- Deterministic four-sensor synthetic data generator and CSV upload
- Isolation Forest and compact sklearn MLP reconstruction model
- Chronological train/test split, training-normal threshold, precision, recall, F1, ROC AUC and confusion matrix
- Multi-seed comparative benchmark exported to CSV
- Optional **separately trained** Keras INT8 TensorFlow Lite export script
- Pytest unit tests, GitHub Actions CI, research and classroom guides

## Install and run
Requires Python 3.11 or a compatible supported Python version.

```bash
python -m venv .venv
# Activate your environment
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

To produce research comparison data:

```bash
python benchmark.py --seeds 3 --output results/benchmark.csv
```

Optional INT8 export (TensorFlow is deliberately not a required Streamlit Cloud dependency):

```bash
pip install 'tensorflow>=2.16'
python export_tflite.py
```

## Publish the web app
Go to https://share.streamlit.io, connect this repository, choose branch `main`, and set the entrypoint to `app.py`. See [deployment checklist](docs/DEPLOYMENT_CHECKLIST.md).

## Classroom and research documents
- [Presentation script](docs/CLASS_DEMO.md)
- [Research methodology](docs/RESEARCH_GUIDE.md)
- [Deployment and validation checklist](docs/DEPLOYMENT_CHECKLIST.md)

## Important limitations
Synthetic sensor anomalies do not establish real-world detection accuracy. The dashboard uses sklearn models; the optional INT8 export trains a **different Keras model**. TFLite export alone is not microcontroller deployment. Quantized thresholds, operator compatibility, RAM, flash, inference latency, and power must be measured on target hardware. Predictions are not certified industrial safety alerts.

**Status:** Source code is committed. A public Streamlit deployment, physical device benchmarks, and uploading original DOCX/PPTX binaries remain separate tasks.
