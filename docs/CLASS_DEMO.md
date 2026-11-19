# TinyGuard AI — Classroom Demonstration

## 8–10 minute presentation

1. **Problem (1 min):** Explain why resource-constrained IoT systems need anomaly detection.
2. **Architecture (1 min):** Four sensors → preprocessing → lightweight detector → anomaly score → dashboard.
3. **Data (1 min):** Explain the reproducible synthetic dataset and its limitations.
4. **Models (2 min):** Compare Isolation Forest and a compact autoencoder.
5. **Live demonstration (2 min):** Open the **Live demo** tab; submit normal readings, then raise temperature to 85 and vibration to 0.9. Note that outcomes depend on the fitted threshold.
6. **Evaluation (1 min):** Show precision, recall, F1, ROC AUC and confusion matrix on the chronological holdout.
7. **TinyML distinction (1 min):** Explain that the separate optional int8 TFLite model is not the sklearn model. Hardware deployment and benchmarking are future work.
8. **Questions (1 min):** Discuss false positives, threshold selection, concept drift and embedded limitations.

## Deployment

Publish the main branch at https://share.streamlit.io with entrypoint `app.py`. Confirm it loads before the presentation. Prepare screenshots as a backup in case classroom internet fails.

## Academic honesty

Do not claim hardware execution, industrial validation, calibrated probabilities, or measured latency/RAM/power without evidence.
