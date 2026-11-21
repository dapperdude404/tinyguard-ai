from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "android-edge/app/src/main/java/ai/smartplant/edge"
RES = ROOT / "android-edge/app/src/main/res"

def test_crop_review_preserves_original_probability():
    code = (SRC / "CropReview.kt").read_text(encoding="utf-8")
    assert "withinCrop" in code
    assert "it.probability" in code
    assert "mismatch" in code
    assert "equals(selectedCrop, ignoreCase = true)" in code

def test_phase2_ui_keeps_top_five_and_selectable_crop():
    code = (SRC / "MainActivity.kt").read_text(encoding="utf-8")
    assert "cropSpinner" in code
    assert "topK = 29" in code
    assert "showPredictions(predictions.take(5), assessment)" in code
    assert "Crop mismatch:" in code

def test_launcher_icon_registered():
    manifest = (ROOT / "android-edge/app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")
    assert '@mipmap/ic_smartplant' in manifest
    assert (RES / "mipmap-anydpi-v26/ic_smartplant.xml").exists()
    assert (RES / "drawable/ic_smartplant_foreground.xml").exists()

def test_offline_classifier_independent_of_opt_in_network():
    manifest = (ROOT / "android-edge/app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")
    assert "android.permission.INTERNET" in manifest
    classifier = (SRC / "EdgeClassifier.kt").read_text()
    assert "CloudSync" not in classifier
