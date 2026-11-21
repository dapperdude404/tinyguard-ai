# SmartPlant AI Phase 2 — Crop-aware error analysis (experimental)

**Status:** development branch only; not merged or released.

## Motivation

A photographed potato plant was classified by the Android CNN as **Strawberry | Leaf Scorch (88.4%)**. This observed error motivates a transparent crop-aware review UI. It does not prove background leakage or any particular failure mechanism.

## Phase 2 implementation

1. Preserve the existing 29-class MobileNetV3-Small ONNX model; no training weights are changed.
2. Let the user optionally select a **known crop** from the crop names actually listed in the model labels, or select `Auto / unknown`.
3. Run all 29 class scores and preserve original probabilities (NO crop reweighting or renormalization).
4. If the top predicted class belongs to another crop, show a prominent mismatch warning with the user-selected crop, original AI-predicted crop, and the top three scores *within the user-selected crop*.
5. Keep original overall top-five ranking so users can inspect misclassification honestly.
6. Add a branded vector launcher icon and dashboard logo.

**Important:** The crop selector is supplied by the user. It is not automatic plant-species recognition, a new hierarchical classifier or calibrated disease diagnosis.

## Reproducible failure-case analysis

Keep a test table recording: expected crop, expected disease label (if independently confirmed), top predicted class, top probability, other class probabilities, source image ID, lighting, background, phone model, and comparison with Streamlit inference. Do not publicly claim ground-truth disease merely from visual appearance.

## Model research roadmap

- Build an independent, plant-/farm-separated evaluation set and audit near-duplicate photos.
- Establish a full cross-crop confusion matrix and per-class recall and macro F1.
- Compare performance on the same leaf under different backgrounds; treat background reliance as a hypothesis to test.
- Train a separate, validated crop-classification stage if the evidence supports a true hierarchical architecture.
- Investigate selective classification/rejection using calibrated probabilities and out-of-distribution samples rather than arbitrary confidence thresholds.
- Validate ONNX-vs-PyTorch and PIL-vs-Android preprocessing parity on real photos.
- Add Grad-CAM heatmaps after model transparency is independently evaluated.

## Test gates

1. Unit tests pass for crop-name parsing, mismatch detection, and original probabilities.
2. ONNX conversion parity and Android debug APK compilation pass in GitHub CI.
3. On-device camera/gallery, crop selector, warnings, scan history, airplane mode and launcher icon visually checked.
4. Record potato-versus-strawberry error case with known crop = Potato.
5. No merge or public release until the above evidence is reviewed.
