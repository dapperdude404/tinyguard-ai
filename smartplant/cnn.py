"""Optional MobileNetV3 inference. No prediction is made without a trained checkpoint."""
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=2)
def load_cnn(checkpoint_path):
    if not Path(checkpoint_path).is_file():
        return None
    import torch
    from torchvision import models
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "labels" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("Unsupported checkpoint: expected labels and state_dict.")
    labels = checkpoint["labels"]
    if not isinstance(labels, dict) or not 2 <= len(labels) <= 128:
        raise ValueError("Checkpoint must have 2..128 class labels.")
    if not all(isinstance(name, str) and name.strip() and isinstance(index, int)
               for name, index in labels.items()):
        raise ValueError("Invalid label metadata.")
    if set(labels.values()) != set(range(len(labels))):
        raise ValueError("Labels must use unique consecutive indices.")
    if checkpoint.get("architecture") not in (None, "mobilenet_v3_small"):
        raise ValueError("Unexpected CNN architecture.")
    if checkpoint.get("input_size") not in (None, 160):
        raise ValueError("Unexpected model input size.")
    names = [label for label, _ in sorted(labels.items(), key=lambda item: item[1])]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(names))
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as exc:
        raise ValueError("Checkpoint architecture does not match MobileNetV3-Small.") from exc
    model.eval()
    return model, names

def predict_cnn(image, model_and_labels, top_k=5):
    import torch
    from torchvision import transforms
    model, names = model_and_labels
    transform = transforms.Compose([
        transforms.Resize(176), transforms.CenterCrop(160),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406], [.229,.224,.225])
    ])
    with torch.inference_mode():
        probabilities = model(transform(image.convert("RGB")).unsqueeze(0)).softmax(1)[0]
    values, indices = probabilities.topk(min(top_k, len(names)))
    return [(names[int(i)], float(v)) for v, i in zip(values, indices)]
