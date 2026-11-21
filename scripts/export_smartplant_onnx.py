"""Export a SHA-pinned MobileNetV3-Small checkpoint (29 or 50 labels) to ONNX.

Default is the existing verified 29-class model, preserving current Android CI.
For the newly TRAINED model you MUST supply its computed SHA-256 explicitly:
python scripts/export_smartplant_onnx.py --checkpoint models/smartplant_50.pt \
  --expected-sha256 <training-output-64-hex-digest>
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

EXPECTED_SHA256 = "f41551f9eb0a084856f07ad154c1b53739253f0d614a37056bfa77e7e37fc313"


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_names(checkpoint: dict) -> list[str]:
    labels = checkpoint.get("labels")
    if not isinstance(labels, dict) or not 2 <= len(labels) <= 128:
        raise ValueError("Checkpoint must include 2..128 indexed class labels")
    if not all(isinstance(name, str) and name.strip() and isinstance(i, int)
               for name, i in labels.items()):
        raise ValueError("Invalid checkpoint labels")
    if set(labels.values()) != set(range(len(labels))):
        raise ValueError("Class indices must be consecutive from zero")
    return [name for name, _ in sorted(labels.items(), key=lambda entry: entry[1])]


def export(checkpoint_path: str | Path, output_dir: str | Path,
           expected_sha256: str = EXPECTED_SHA256) -> dict:
    import torch
    import onnx
    import onnxruntime as ort
    from torchvision import models

    checkpoint_path = Path(checkpoint_path)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("Expected SHA-256 must be a complete 64-character hex string")
    actual_checkpoint_hash = sha256(checkpoint_path)
    if actual_checkpoint_hash != expected_sha256.lower():
        raise ValueError("Checkpoint SHA-256 mismatch; refusing to export untrusted weights")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError("Unexpected checkpoint architecture")
    if checkpoint.get("input_size") != 160 or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Expected 160-pixel input size and state_dict")
    names = ordered_names(checkpoint)
    model = models.mobilenet_v3_small(weights=None)
    last = model.classifier[-1]
    model.classifier[-1] = torch.nn.Linear(last.in_features, len(names))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "smartplant.onnx"
    sample = torch.randn(1, 3, 160, 160)
    torch.onnx.export(
        model, sample, str(target), input_names=["image"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(str(target))
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    assert tuple(session.get_inputs()[0].shape) == (1, 3, 160, 160)
    for seed in (1, 7, 23):
        torch.manual_seed(seed)
        tensor = torch.randn(1, 3, 160, 160)
        with torch.inference_mode():
            reference = model(tensor).numpy()
        actual = session.run(None, {session.get_inputs()[0].name: tensor.numpy()})[0]
        if actual.shape != (1, len(names)):
            raise RuntimeError("ONNX output shape does not match class labels")
        error = float(np.max(np.abs(reference - actual)))
        print(f"Parity seed={seed}: max_abs_logit_error={error:.6g}")
        if not np.allclose(reference, actual, atol=1e-3, rtol=1e-3):
            raise RuntimeError("PyTorch / ONNX parity check failed")

    labels_path = output_dir / "labels.json"
    labels_path.write_text(json.dumps(names, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema": "smartplant-onnx-manifest-v2",
        "model": target.name,
        "sha256": sha256(target),
        "labels": labels_path.name,
        "labels_sha256": sha256(labels_path),
        "class_count": len(names),
        "input_shape": [1, 3, 160, 160],
        "preprocess": "RGB, resize shortest side 176 bilinear, center crop 160, ImageNet normalize",
        "output": f"{len(names)} logits; apply softmax",
        "source_checkpoint_sha256": actual_checkpoint_hash,
    }
    (output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(names)}-class ONNX: {target} ({target.stat().st_size} bytes)")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/smartplant_mobilenetv3.pt")
    parser.add_argument("--output", default="android-edge/app/src/main/assets")
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256,
                        help="Mandatory explicit new digest when exporting new trained checkpoint")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.expected_sha256)


if __name__ == "__main__":
    main()
