"""Fast source checks for the SmartPlant Android V2 screen and privacy invariants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "android-edge/app/src/main"
KOTLIN = APP / "java/ai/smartplant/edge/MainActivity.kt"

def test_dashboard_has_camera_gallery_and_inference():
    source = KOTLIN.read_text()
    assert "ActivityResultContracts.TakePicture()" in source
    assert "ActivityResultContracts.GetContent()" in source
    assert "EdgeClassifier(applicationContext).use" in source

def test_safe_android_window_insets():
    source = KOTLIN.read_text()
    assert "ViewCompat.setOnApplyWindowInsetsListener" in source
    assert "WindowInsetsCompat.Type.systemBars()" in source

def test_probabilities_and_local_history():
    source = KOTLIN.read_text()
    assert "ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal)" in source
    assert 'getSharedPreferences("smartplant_edge_scans", MODE_PRIVATE)' in source
    assert "JSONArray" in source
    assert "setPositiveButton" in source

def test_camera_exif_and_bounded_decode():
    source = KOTLIN.read_text()
    assert "ExifInterface" in source
    assert "inSampleSize = sample" in source
    assert "FileProvider.getUriForFile" in source

def test_no_unnecessary_camera_or_storage_permissions():
    manifest = (APP / "AndroidManifest.xml").read_text()
    assert "android.permission.INTERNET" in manifest
    assert 'android:usesCleartextTraffic="false"' in manifest
    assert "android.permission.READ_EXTERNAL_STORAGE" not in manifest
    assert "android.permission.CAMERA" not in manifest

def test_model_remains_integrity_verified():
    source = (APP / "java/ai/smartplant/edge/EdgeClassifier.kt").read_text()
    assert 'MessageDigest.getInstance("SHA-256")' in source
    assert "onnxruntime" in source
