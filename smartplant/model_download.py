"""Install the explicitly approved SmartPlant model from a public Drive file.

The SHA-256 pin prevents an unexpected or changed checkpoint from being loaded.
"""
import hashlib
import os
from pathlib import Path
import tempfile

MODEL_ID = "1oF2W3-CelHeIxYWJLOL8m1yFRviIpWVZ"
MODEL_SHA256 = "77a85ecacfa812b3d1ee0d9bdb709b226eef8f910fb8713543dd7e4a892b3ac3"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cnn_checkpoint(path):
    target = Path(path)
    if target.is_file():
        if _sha256(target) != MODEL_SHA256:
            raise ValueError("Installed checkpoint SHA-256 mismatch; refusing to load it.")
        return target

    file_id = os.getenv("SMARTPLANT_MODEL_DRIVE_ID", MODEL_ID).strip()
    if not file_id:
        return None

    import gdown
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        downloaded = gdown.download(id=file_id, output=str(temporary), quiet=True)
        if not downloaded or temporary.stat().st_size < 1024:
            raise RuntimeError("Model download failed; check Drive sharing and quotas.")
        if _sha256(temporary) != MODEL_SHA256:
            raise ValueError("Downloaded checkpoint SHA-256 mismatch; refusing to install.")
        import torch
        checkpoint = torch.load(temporary, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != "mobilenet_v3_small":
            raise ValueError("Unexpected checkpoint architecture.")
        if not isinstance(checkpoint.get("state_dict"), dict) or not isinstance(checkpoint.get("labels"), dict):
            raise ValueError("Missing checkpoint weights or labels.")
        if len(checkpoint["labels"]) != 29 or checkpoint.get("input_size") != 160:
            raise ValueError("Unexpected class count or input size.")
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)
