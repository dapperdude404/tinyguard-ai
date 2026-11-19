# SmartPlant AI — Classroom Presentation Script (8–10 minutes)

## Slide 1 — Title (30 sec)
SmartPlant AI: Explainable Plant Leaf Disease Classification — M.Tech Artificial Intelligence.

## Slide 2 — Problem (1 min)
Farmers need accessible crop-health screening, but leaf symptoms can be visually similar. This tool is an educational classifier prototype, not a substitute for agronomy expertise.

## Slide 3 — Objectives (1 min)
Build image upload, preprocessing, interpretable features, a trained model interface, and a reproducible evaluation pipeline.

## Slide 4 — Architecture (1 min)
Leaf image → RGB/resize → HOG texture + RGB histogram → trained Random Forest → class probabilities → Streamlit dashboard.

## Slide 5 — Demonstration (2 min)
Upload a clear leaf photograph. Show the normalized image and the color indicators. Explain these are **not diagnoses**. If a trained model has been installed, show its ranked class predictions; otherwise explain the training step and missing model artifact.

## Slide 6 — Training and evaluation (1 min)
Explain labeled dataset folders, stratified holdout and the importance of an independent test set. Discuss accuracy, macro F1 and confusion matrix.

## Slide 7 — Limitations (1 min)
Controlled images vs real farms, lighting, domain shift, model calibration, lack of pretrained weights in this repository.

## Slide 8 — Future work (1 min)
MobileNet transfer learning, Grad-CAM, multi-crop training, mobile deployment, TensorFlow Lite and external validation.

## Slide 9 — Conclusion / Questions (1 min)
Explain that transparency and reproducibility are key to an academically defensible AI project.

## Live demo fallback
Save screenshots before class in case internet is unavailable. Run locally with `streamlit run app.py` if necessary.
