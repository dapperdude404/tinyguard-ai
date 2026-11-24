"""Fast validation of offline Android export invariants (no model binary needed)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_export_script_compiles():
    source = (ROOT / "scripts/export_smartplant_onnx.py").read_text(encoding="utf-8")
    ast.parse(source)

def test_android_opt_in_tls_permission():
    manifest = (ROOT / "android-edge/app/src/main/AndroidManifest.xml").read_text()
    assert "android.permission.INTERNET" in manifest
    assert 'android:usesCleartextTraffic="false"' in manifest

def test_manifest_uses_private_file_provider():
    manifest = (ROOT / "android-edge/app/src/main/AndroidManifest.xml").read_text()
    assert 'android:exported="false"' in manifest
    assert "FileProvider" in manifest

def test_android_preprocess_and_class_count():
    code = (ROOT / "android-edge/app/src/main/java/ai/smartplant/edge/EdgeClassifier.kt").read_text()
    assert "160 * 160" in code
    assert "labels.size == 29" in code
    assert "SHA-256" in code
    assert "ImageNet" in code

def test_export_hash_is_pinned():
    source = (ROOT / "scripts/export_smartplant_onnx.py").read_text()
    assert "f41551f9eb0a084856f07ad154c1b53739253f0d614a37056bfa77e7e37fc313" in source
    assert "weights_only=True" in source
