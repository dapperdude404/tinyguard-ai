# SmartPlant AI — Android Offline Edge AI, Phase 1

**Status: experimental development branch; not merged, deployed, or verified on an Android device.**

## Scope
- Reuse the trusted 29-class MobileNetV3-Small weights from SmartPlant AI.
- Export an ONNX model and label mapping with SHA-256 integrity manifest.
- Verify PyTorch vs ONNX logits on three synthetic inputs.
- Native Kotlin Android app: take a photo with installed camera app or choose a gallery photo; run ONNX Runtime **entirely on device**, without a network request.
- Show ranked top-five class probabilities.

## Architecture
Phone camera/gallery → Android bitmap → resize shortest side to 176, center crop 160, RGB CHW ImageNet normalization → ONNX Runtime CPU → 29 logits → softmax → ranked results.

The Android project deliberately declares **no INTERNET permission**. This is a different execution path from the existing Streamlit cloud app. The model binary is not stored in GitHub.

## Build preparation
1. Download your trusted `best_model_safe.pt` to `models/smartplant_mobilenetv3.pt` locally. Expected SHA-256: `f41551f9eb0a084856f07ad154c1b53739253f0d614a37056bfa77e7e37fc313`.
2. Use Python 3.11 and install `torch torchvision onnx onnxruntime pillow gdown`.
3. Run `python scripts/export_smartplant_onnx.py --checkpoint models/smartplant_mobilenetv3.pt`.
4. Confirm parity tests pass and `android-edge/app/src/main/assets/` contains `smartplant.onnx`, `labels.json`, `model_manifest.json`.
5. Open `android-edge` in Android Studio with JDK 17, sync Gradle, and run on an Android 8.0+ device.
6. Test camera capture, gallery selection, airplane mode, repeat inference, orientation, low memory, and corrupted model rejection.

## Important limitations and validation gates
- **Not yet built or run** in Android Studio. Dependencies and SDK availability require verification.
- ONNX parity is for raw normalized tensors. Android Bitmap resizing and PyTorch/PIL resizing may differ slightly: validate preprocessing parity on the **same real leaf images** before reporting numerical equivalence.
- The app currently uses CPU ONNX Runtime, not Android NNAPI/GPU/NPU acceleration.
- Grad-CAM, continuous camera preview, real-time video inference, sensor connectivity, disease treatment advice, and calibrated uncertainty are **not implemented**.
- Camera uses the system camera via Activity Result `TakePicture`, not CameraX. No additional IoT hardware is required.
- The trained model's real-world accuracy remains unverified; class probabilities are not diagnostic confidence.
- Do not merge until the Android build, model parity, device camera flow, offline test and real-leaf comparison have been demonstrated.


## CI verification (updated after run #44894401549)
- The first run's `static` job passed all **5** tests.
- The first `android-compile` job failed **before compilation**, because `android-actions/setup-android@v3` invoked a removed SDK package named `tools`.
- The development workflow now installs supported Android platform/build-tools packages using the runner SDK, and adds a new `onnx-parity` job. This downloads the pinned checkpoint, compares ONNX and PyTorch logits on three synthetic inputs, and hands generated assets to the Android debug build.
- If compilation succeeds, CI uploads `smartplant-edge-debug-UNTESTED` for short-lived inspection. The artifact **must not be treated as production-ready** without device and airplane-mode tests.
- The exported `.onnx`, labels and manifest are ignored by Git. The working Streamlit project on `main` remains untouched.
- **CI outcomes after these changes have not yet been independently confirmed**; do not infer a passing Android build from the earlier static-test success.

## Suggested acceptance tests
| Test | Expected |
| --- | --- |
| SHA-256 mismatch | Export or Android load fails closed |
| ONNX vs PyTorch | Three deterministic synthetic samples within 1e-3 absolute/relative tolerance |
| Labels | Exactly 29; index order preserved |
| Camera capture | Full-resolution photo opens and yields top-five probabilities |
| Airplane mode | Inference still works after app and assets are installed |
| Real-leaf parity | Compare same photo against Streamlit prediction and investigate discrepancies |
| App restart | Inference still works without network |
| Unknown/blurred leaf | App does not present a guaranteed diagnosis |

## GitHub workflow
All work stays on `feature/smartplant-android-edge-phase1` and in a draft pull request. No changes to `main` or Streamlit production.

## Verified CI results — 21 November 2026

Workflow: [Android Edge Phase 1 Checks, run 44897501674](https://github.com/dapperdude404/tinyguard-ai/actions/runs/44897501674), based on commit `f6faf03002567f01e59daba0809f96cc0b45bb4b`.

- **Static:** passed (five tests).
- **ONNX parity:** passed after checksum-verified download of the trusted 29-class checkpoint; maximum absolute raw-logit differences on three seeded synthetic tensors were approximately 1.24e-5, 1.24e-5 and 1.62e-5 in the preceding run.
- **Android Gradle:** `assembleDebug` succeeded after correcting the non-null camera `Uri` and SDK path handling.
- **Debug APK:** [smartplant-edge-debug-UNTESTED](https://github.com/dapperdude404/tinyguard-ai/actions/runs/44897501674/artifacts/14704438646), uploaded and retained until approximately 28 November 2026 (according to GitHub artifact metadata).
- **Full SmartPlant AI tests:** passed in workflow run 44897501668.

**Release status:** Build readiness established; this is not a tested or published Android release. On-device installation, phone camera/gallery operations, offline/airplane-mode inference, image-preprocessing parity against Streamlit, memory/performance and crash recovery are still required before releasing or merging.
