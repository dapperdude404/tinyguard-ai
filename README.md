# TinyGuard AI — M.Tech AI Research Edition

TinyML-oriented IoT sensor anomaly detection with a Streamlit dashboard, Isolation Forest, compact autoencoder, evaluation tools, synthetic datasets, and an optional TensorFlow Lite quantization pipeline.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Deployment

Deploy `app.py` from the `main` branch on https://share.streamlit.io after all source files are committed.

## Research integrity

Synthetic data is not a validated industrial dataset. The sklearn autoencoder used for the dashboard is distinct from the optional TensorFlow Lite export model. Device latency, power, and RAM must be measured on hardware before claiming embedded TinyML performance.

## Classroom package

The original complete ZIP, DOCX research report, and PPTX presentation are available in the ChatGPT conversation. Upload remaining source and binary presentation files before deploying this repository.
