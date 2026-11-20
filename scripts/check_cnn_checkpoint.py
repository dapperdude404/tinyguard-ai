"""Verify the installed trusted SmartPlant MobileNetV3 checkpoint before deployment."""
from pathlib import Path
import sys

def main():
    path = Path("models/smartplant_mobilenetv3.pt")
    if not path.is_file():
        print(f"Missing checkpoint: {path}")
        return 2
    import torch
    from PIL import Image
    from smartplant.cnn import load_cnn, predict_cnn
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") not in (None, "mobilenet_v3_small"):
        print("Unexpected architecture:", checkpoint.get("architecture"))
        return 3
    model_and_labels = load_cnn(str(path))
    if model_and_labels is None:
        return 4
    _, labels = model_and_labels
    if len(labels) != 29:
        print("Unexpected number of classes:", len(labels))
        return 5
    results = predict_cnn(Image.new("RGB", (160, 160), (50, 150, 50)), model_and_labels)
    assert results and all(0 <= p <= 1 for _, p in results)
    print(f"PASS: loaded {len(labels)} classes and completed inference. Top category: {results[0][0]}")
    print("Note: synthetic-image inference is a compatibility test, not a disease accuracy evaluation.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
