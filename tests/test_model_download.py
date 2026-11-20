"""Offline tests for checkpoint recovery and download failures."""
from pathlib import Path

import pytest

from smartplant import model_download


def test_existing_valid_checkpoint_is_reused(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    path.write_bytes(b"valid")
    monkeypatch.setattr(model_download, "_validate", lambda candidate: None)
    assert model_download.ensure_cnn_checkpoint(path) == path


def test_invalid_placeholder_can_be_replaced(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    path.write_bytes(b"\r\n")
    def validate(candidate):
        if Path(candidate).read_bytes() == b"\r\n":
            raise EOFError("empty placeholder")
    monkeypatch.setattr(model_download, "_validate", validate)
    import sys
    from types import SimpleNamespace
    def download(id, output, quiet):
        Path(output).write_bytes(b"x" * 2048)
        return output
    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=download))
    assert model_download.ensure_cnn_checkpoint(path) == path
    assert path.stat().st_size == 2048


def test_download_failure_keeps_placeholder(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    path.write_bytes(b"\r\n")
    monkeypatch.setattr(model_download, "_validate", lambda candidate: (_ for _ in ()).throw(EOFError()))
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=lambda **kwargs: None))
    with pytest.raises(RuntimeError, match="download failed"):
        model_download.ensure_cnn_checkpoint(path)
    assert path.read_bytes() == b"\r\n"
