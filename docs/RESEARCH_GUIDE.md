# Research Guide — TinyGuard AI

## Proposed title
Explainable Lightweight Anomaly Detection for IoT Sensor Streams: A Reproducible TinyML-Oriented Study

## Research question
How does a compact reconstruction autoencoder compare with Isolation Forest for anomaly detection in synthetic multivariate sensor streams, and what tradeoffs arise when deploying a separately trained int8 quantized model?

## Experimental protocol
- Inputs: temperature, vibration, humidity, current.
- Generate reproducible simulated normal signals and injected anomalies with fixed seeds.
- Preserve chronological order; use first 65% for training and remaining 35% for evaluation.
- Fit scaling on normal training observations only.
- Compare Isolation Forest and compact sklearn MLP reconstruction error.
- Select each anomaly threshold using the 97th percentile of normal training scores.
- Report precision, recall, F1, ROC AUC, and confusion matrix, noting class imbalance.
- Repeat across multiple seeds and anomaly intensities to quantify variance.
- Evaluate on independently sourced real IoT data before making external-validity claims.

## TinyML quantization
Run `pip install 'tensorflow>=2.16'` then `python export_tflite.py` to train a **separate** Keras autoencoder and export an int8 TFLite artifact. Record bytes, tensor quantization parameters, operator compatibility and recalibrated anomaly threshold. Benchmark on an identified microcontroller for flash size, peak RAM, latency, power, and predictive quality. Quantized model performance must be evaluated independently.

## Threats to validity
Synthetic anomalies are simple and separable; normal data may drift; sensor noise differs across deployments; percentile thresholds affect false alarms; and the exported TFLite model is not identical to the dashboard's sklearn model.

## Suggested paper structure
Abstract; Introduction; Related Work; Dataset and Methodology; Models and Quantization; Evaluation; Discussion; Limitations; Conclusion; References.

## Citation guidance
Cite the actual versions of scikit-learn, Streamlit, TensorFlow Lite, and any datasets used. Do not invent references, performance numbers or benchmark results.
