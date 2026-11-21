# SmartPlant AI — Android V2 Professional UI

**Status:** Development only. No merge, public release, or production deployment intended.

## Design and scope

V2 builds on the verified 29-class MobileNetV3-Small ONNX Android engine; it does not change weights, labels, inference code or Streamlit production.

- Native Kotlin green-themed, scrollable dashboard with clear capture and results sections.
- Android system-window insets prevent content underlapping status/navigation bars.
- System camera photo capture and gallery image selection (no camera hardware other than the Android phone).
- Gallery/camera image reading on a worker thread with bounded bitmap decoding and EXIF orientation correction.
- Top predicted category, model class probability, top-five bars, visible reminder not to interpret probabilities as medical/agronomic certainty.
- Local scan history (up to 20 records: label, probability, timestamp) saved via SharedPreferences; no leaf image bytes in history. Clear-history confirmation dialog.
- No INTERNET permission in manifest. ONNX Runtime processes photographs locally, using an integrity-checked bundled model.
- Research note: photo previews only exist in memory, but temporary camera JPEGs are created in the app's cache for camera-app integration.

## Source

- UI: `android-edge/app/src/main/java/ai/smartplant/edge/MainActivity.kt`
- Inference: `android-edge/app/src/main/java/ai/smartplant/edge/EdgeClassifier.kt`
- ONNX export / integrity: `scripts/export_smartplant_onnx.py`
- App version: `0.2.0` debug development build.

## Tests and release gates

1. Run existing PyTorch-to-ONNX parity checks on the trusted checkpoint.
2. Run Python checks, Android Gradle `assembleDebug` and inspect GitHub Actions logs.
3. Install a debug APK built **with** generated model assets, and test on a physical Android phone.
4. Confirm the top bar is fully visible below the status area and the bottom history is scrollable above navigation.
5. Test camera and gallery, portrait rotation, invalid images, large images and repeated scans.
6. Enable airplane mode and verify camera/gallery inference and history persist through app restart.
7. Compare the **same real leaf photos** against Streamlit; preprocessing may differ subtly because Android Bitmap scaling and torchvision resize are not pixel-identical.
8. Check memory usage, crash reports and readability on narrow screens before marking V2 ready.

**Existing model class probabilities are not independent diagnostic validation.** No Grad-CAM, treatments, background sync, real-time video, microcontroller or sensor integration is claimed.
