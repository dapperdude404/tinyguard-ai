"""Validate opt-in network feature without weakening offline CNN."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android-edge/app/src/main/java/ai/smartplant/edge"
MANIFEST = (ROOT / "android-edge/app/src/main/AndroidManifest.xml").read_text()


def test_network_permission_is_explicit_and_tls_only():
    assert 'android.permission.INTERNET' in MANIFEST
    assert 'android:usesCleartextTraffic="false"' in MANIFEST
    sync = (JAVA / "CloudSync.kt").read_text()
    assert "protocol == \"https\"" in sync
    assert "HttpsURLConnection" in sync


def test_only_user_consented_metadata_is_sent():
    activity = (JAVA / "MainActivity.kt").read_text()
    assert "Send once" in activity
    assert "No leaf images are sent" in activity
    assert "CloudSync.send" in activity
    assert "lastScanPayload" in activity
    sync = (JAVA / "CloudSync.kt").read_text()
    assert '"image"' not in sync\n    assert "JSONObject" in sync
    assert "Authorization" in sync


def test_local_inference_remains_without_network():
    activity = (JAVA / "MainActivity.kt").read_text()
    assert "EdgeClassifier(applicationContext).use" in activity
    assert "ActivityResultContracts.TakePicture()" in activity
    assert "ActivityResultContracts.GetContent()" in activity
