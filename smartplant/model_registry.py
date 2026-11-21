"""Explicitly select a verified CNN checkpoint without changing production fallback.

The 29-class checkpoint remains SHA-pinned and fetched as before. An optional
trained expansion is selected ONLY via an explicit on-disk path + SHA-256 digest.
No model download, deployment, or runtime weights switch is implied by this code.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from smartplant.model_download import ensure_cnn_checkpoint


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_cnn_checkpoint() -> tuple[str | None, str]:
    path = os.environ.get("SMARTPLANT_EXPANDED_MODEL_PATH", "").strip()
    if path:
        expected = os.environ.get("SMARTPLANT_EXPANDED_MODEL_SHA256", "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Expanded checkpoint requires SMARTPLANT_EXPANDED_MODEL_SHA256")
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Configured expanded checkpoint not found: {resolved}")
        if file_sha256(resolved) != expected:
            raise ValueError("Expanded checkpoint SHA-256 mismatch")
        return str(resolved), "expanded"
    legacy = "models/smartplant_mobilenetv3.pt"
    verified = ensure_cnn_checkpoint(legacy)
    return (str(verified), "legacy") if verified is not None else (None, "legacy")
