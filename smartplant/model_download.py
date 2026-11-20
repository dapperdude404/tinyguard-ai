"""Download and validate a trusted SmartPlant checkpoint before inference."""
import os
from pathlib import Path
import tempfile

MODEL_ID = "1iDoD3nxQNv9qgTm6HYH4_sUAxEwbpp59"
MODEL_SHA256 = "f41551f9eb0a084856f07ad154c1b53739253f0d614a37056bfa77e7e37fc313"


def _validate(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        raise ValueError("Checkpoint SHA-256 mismatch; refusing to load.")
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")
    if checkpoint.get("architecture") not in (None, "mobilenet_v3_small"):
        raise ValueError("Unexpected checkpoint architecture.")
    if not isinstance(checkpoint.get("state_dict"), dict) or not isinstance(checkpoint.get("labels"), dict):
        raise ValueError("Missing weights or labels.")
    if len(checkpoint["labels"]) != 29:
        raise ValueError("Expected 29 classes.")
    return checkpoint


def ensure_cnn_checkpoint(path):
    target = Path(path)
    if target.is_file():
        try:
            _validate(target)
            return target
        except (ValueError, RuntimeError, EOFError, OSError):
            # A previous empty or corrupt placeholder must not suppress download.
            pass

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
        _validate(temporary)
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)
