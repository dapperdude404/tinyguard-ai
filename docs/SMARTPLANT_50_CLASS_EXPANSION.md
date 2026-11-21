# SmartPlant AI 50-class expansion — research implementation plan

**Status:** Source-code preparation and test automation only. As of this branch, no new dataset was supplied, no 50-class training was run, no field accuracy was verified, and no app was released or deployed.

## Objective and scope

Retain **all 29 originally deployed MobileNetV3-Small class labels and their numeric positions**; append **21 candidates** across eight new crops to total **50 unique classes**. An optional, SHA-pinned trained checkpoint can later power Streamlit and offline Android ONNX inference.

This expansion does not cure the known failure cases automatically:
- A potato leaf was incorrectly classified as **Strawberry | Leaf Scorch** at 88.4% model softmax probability.
- A non-leaf fabric image was classified as **Cherry | Healthy** with 100% displayed model probability.
- A previously high test accuracy was not independently established because source-level dataset overlap was a concern.

These remain **test cases**, not evidence of field performance. Never use a single softmax score as calibrated diagnostic confidence.

## Proposed new class catalog — 21 categories

| Crop | New categories | Added |
| --- | --- | ---: |
| Mango | Healthy; Anthracnose; Powdery Mildew | 3 |
| Guava | Healthy; Anthracnose; Leaf Spot | 3 |
| Banana | Healthy; Black Sigatoka; Yellow Sigatoka | 3 |
| Rice | Healthy; Brown Spot; Leaf Blast | 3 |
| Wheat | Healthy; Leaf Rust; Powdery Mildew | 3 |
| Citrus | Healthy; Canker | 2 |
| Maize | Healthy; Northern Leaf Blight | 2 |
| Papaya | Healthy; Leaf Curl | 2 |
| **New categories** | **Total** | **21** |
| **All categories** | **Existing 29 + proposed 21** | **50** |

**Dataset gate:** These names are *research candidates*. Verify that labels can be assigned reliably and that relevant image rights, provenance and taxonomy are defensible. If actual old labels collide with proposed labels, label-plan generation **fails** rather than silently changing a class. Do not relabel images just to reach 50.

## Phase A — construct and audit a real dataset

1. Use only your own or appropriately licensed photos, labeled by qualified sources. Do not combine train/test copies, resized variants or photos of the same plant across partitions.
2. Obtain the **SHA-pinned original 29-class** checkpoint first. The existing downloader can retrieve it: 
   ```bash
   python -c "from smartplant.model_download import ensure_cnn_checkpoint; print(ensure_cnn_checkpoint('models/smartplant_mobilenetv3.pt'))"
   ```
3. Generate the full 50-label catalog **from the checkpoint's actual original class order**:
   ```bash
   python scripts/prepare_expanded_labels.py --checkpoint models/smartplant_mobilenetv3.pt --output data/expanded/class_plan.json
   ```
4. Organize independently split photos by safe class ID, not `|`-containing filenames:
   ```text
   data/expanded/
     class_plan.json
     train/c000/leaf01.jpg ... train/c049/...
     val/c000/leaf02.jpg   ... val/c049/...
     test/c000/leaf03.jpg  ... test/c049/...
     groups.csv             # strongly recommended: relative_path,group_id
   ```
5. `class_plan.json` lists the readable label for each ID. Classes `c000`–`c028` are exactly the legacy names; `c029`–`c049` are the new candidates.
6. Audit every split:
   ```bash
   python scripts/audit_expanded_dataset.py --dataset data/expanded --plan data/expanded/class_plan.json --groups-csv data/expanded/groups.csv
   ```
   Defaults require at least 20 training, 5 validation and 5 test images **per class** to catch mistakes, but are far below scientific sample-size requirements. Aim for **hundreds of independently sourced images per category** when possible. The audit rejects exact duplicates and dHash near-duplicates across splits, and rejects cross-split plant/farm group IDs. dHash is only a screen: manually verify visually similar images and provenance. If `groups.csv` is unavailable, the report flags group independence as **unverified**, not passed.

A useful `groups.csv` line is `train/c029/mango_leaf_001.jpg,farm12-tree07`. Every image must have a group; no group may appear in more than one split.

## Phase B — fine-tune MobileNetV3-Small

```bash
python scripts/train_expanded_cnn.py \
  --dataset data/expanded --plan data/expanded/class_plan.json \
  --base-checkpoint models/smartplant_mobilenetv3.pt \
  --groups-csv data/expanded/groups.csv \
  --output models/smartplant_50.pt \
  --epochs 25 --patience 6 --seed 42 --batch-size 32 --lr 0.0001
```

The warm-start initializer verifies the original checkpoint SHA-256; loads the existing backbone and final 29 logits; copies the **old classifier's weights/bias rows 0..28 into the new 50-way head**; randomly initializes only rows 29..49. All layers are then fine-tuned jointly on all 50 classes, mitigating—but not guaranteeing prevention of—catastrophic forgetting. Avoid training only on new crop images.

- Default input: RGB 160×160 via short-side resize 176 + center crop; ImageNet mean/std normalization.
- Train augmentation: random resized crop, horizontal flip, slight color jitter. Validation/test are deterministic.
- Optimizer: AdamW; cosine learning-rate schedule; inverse-sqrt frequency weighted cross-entropy.
- Seed: 42; PyTorch deterministic cuDNN flags (hardware/software may still influence bitwise reproducibility).
- Checkpoint selection: **validation macro-F1**; early stopping; **test evaluated only once** after selection.
- Outputs (not tracked in Git): `models/smartplant_50.pt`, `models/smartplant_50.metrics.json`, and `models/smartplant_50.confusion.csv`.
- Report: accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, class support and old/new class subset accuracy. Record crop mismatch and non-leaf samples separately on an external test set.
- Report to your professor honestly: whether per-farm or per-plant groups were supplied, what source rights were documented, limitations and failure cases. **Training output alone never certifies agricultural accuracy.**

## Phase C — export to Android ONNX after training

The exporter supports the current 29-class model **by default** and an arbitrary verified label count up to 128. For a 50-class checkpoint, explicitly supply its SHA-256 from training:

```bash
python scripts/export_smartplant_onnx.py \
  --checkpoint models/smartplant_50.pt \
  --expected-sha256 YOUR_VERIFIED_64_CHARACTER_SHA256 \
  --output android-edge/app/src/main/assets
```

The exporter rejects untrusted hashes, checks label indices, verifies ONNX shape and compares three synthetic tensor outputs against PyTorch (1e-3 tolerance). The manifest binds both the ONNX binary and labels JSON with SHA-256 digests and records the class count. **Synthetic parity tests verify conversion, not crop accuracy.**

The Android `EdgeClassifier.kt` derives output dimensionality from `labels.json` and the signed-by-digest manifest. The crop dropdown uses the labels list. The fixed 160×160 preprocessing is retained. Do not replace a verified 29-class Android asset with a new 50-class binary before testing camera/gallery, offline inference, metadata sync behavior if applicable, and the same real photos on Python and Android.

## Phase D — enable optional Streamlit 50-class checkpoint

Default behavior **remains the deployed SHA-pinned 29-class model**. For a locally available, newly trained 50-class model on a controlled test deployment, configure both:

```bash
export SMARTPLANT_EXPANDED_MODEL_PATH=/absolute/path/to/models/smartplant_50.pt
export SMARTPLANT_EXPANDED_MODEL_SHA256=YOUR_VERIFIED_64_CHARACTER_SHA256
streamlit run app.py
```

The file must actually exist on the target host and pass the full SHA check. The code does not automatically upload a trained binary, retrieve new weights from Drive, merge a branch or replace production. If the expanded model fails validation, the app warns rather than pretending to have 50 classes. When no expanded path is set, it continues loading the original 29-class pinned model.

## Release gates

- [ ] Confirm original checkpoint's 29 labels against the catalog and no collision with 21 new categories.
- [ ] Obtain licensed, source-traceable labeled images for **all 50 classes**.
- [ ] Pass exact/perceptual duplicates and farm-/plant-group checks across train/val/test.
- [ ] Train + record hyperparameters/seed/early-stopping and independently evaluate macro-F1, class metrics and confusion matrix.
- [ ] Re-test the known potato vs strawberry and non-leaf fabric cases, plus external field photos.
- [ ] Export new checkpoint with its exact SHA-256 and check PyTorch/ONNX synthetic AND real-leaf parity.
- [ ] Build Android APK with new 50-class assets and prove camera/gallery and airplane-mode inference still work.
- [ ] Test Streamlit in a **separate** development deployment using an explicitly configured new model path and SHA.
- [ ] Obtain approval **before** any merge or deployment.

**No trained 50-class model, new dataset, accuracy result or Android 50-class APK exists solely because these scripts were committed.**
