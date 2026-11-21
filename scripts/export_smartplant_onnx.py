"""Export the trusted 29-class CNN to ONNX for offline Android inference.

Install: pip install torch torchvision onnx onnxruntime pillow gdown
Usage: python scripts/export_smartplant_onnx.py --checkpoint models/smartplant_mobilenetv3.pt
"""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256 = "f41551f9eb0a084856f07ad154c1b53739253f0d614a37056bfa77e7e37fc313"

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def export(checkpoint_path, output_dir):
    import torch
    import onnx
    import onnxruntime as ort
    import numpy as np
    from torchvision import models

    checkpoint_path = Path(checkpoint_path)
    if sha256(checkpoint_path) != EXPECTED_SHA256:
        raise ValueError("Checkpoint hash mismatch; do not export untrusted weights")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "mobilenet_v3_small" or checkpoint.get("input_size") != 160:
        raise ValueError("Unexpected architecture or input size")
    labels = checkpoint["labels"]
    if not isinstance(labels, dict) or len(labels) != 29 or set(labels.values()) != set(range(29)):
        raise ValueError("Expected exactly 29 indexed labels")
    names = [name for name, index in sorted(labels.items(), key=lambda item: item[1])]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 29)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "smartplant.onnx"
    sample = torch.randn(1, 3, 160, 160)
    torch.onnx.export(model, sample, str(path), input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for seed in (1, 7, 23):
        torch.manual_seed(seed)
        x = torch.randn(1, 3, 160, 160)
        with torch.inference_mode():
            expected = model(x).numpy()
        actual = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
        error = float(np.max(np.abs(expected - actual)))
        print(f"Parity seed={seed}: max_abs_logit_error={error:.6g}")
        if not np.allclose(expected, actual, atol=1e-3, rtol=1e-3):
            raise RuntimeError("ONNX/PyTorch parity check failed")
    (output_dir / "labels.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    (output_dir / "model_manifest.json").write_text(json.dumps({
        "model": "smartplant.onnx", "sha256": sha256(path), "labels": "labels.json",
        "input_shape": [1, 3, 160, 160], "preprocess": "RGB, resize shortest side 176 bilinear, center crop 160, ImageNet normalize",
        "output": "29 logits; apply softmax", "source_checkpoint_sha256": EXPECTED_SHA256
    }, indent=2), encoding="utf-8")
    print("Exported:", path, "bytes:", path.stat().st_size)
    print("Copy smartplant.onnx, labels.json, model_manifest.json to android-edge/app/src/main/assets/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/smartplant_mobilenetv3.pt")
    parser.add_argument("--output", default="android-edge/app/src/main/assets")
    args = parser.parse_args()
    export(args.checkpoint, args.output)
