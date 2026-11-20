"""Download an explicitly configured, publicly accessible trusted CNN checkpoint."""
from pathlib import Path
import os
import tempfile

def ensure_cnn_checkpoint(path):
    target = Path(path)
    if target.is_file():
        return target
    file_id = os.getenv("SMARTPLANT_MODEL_DRIVE_ID", "").strip()
    if not file_id:
        return None
    import gdown
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".pt", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        downloaded = gdown.download(id=file_id, output=str(temporary), quiet=True)
        if not downloaded or temporary.stat().st_size < 1024:
            raise RuntimeError("Google Drive model download failed. Check public sharing permissions.")
        # Validate structure before placing the checkpoint in the live model path.
        import torch
        checkpoint = torch.load(temporary, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict) or not isinstance(checkpoint.get("labels"), dict):
            raise ValueError("Downloaded checkpoint does not contain expected state_dict and labels.")
        if len(checkpoint["labels"]) != 29:
            raise ValueError("Unexpected number of CNN labels.")
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)
