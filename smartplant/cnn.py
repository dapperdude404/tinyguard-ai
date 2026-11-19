"""Optional MobileNetV3 inference. No prediction is made without a trained checkpoint."""
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=2)
def load_cnn(checkpoint_path):
    if not Path(checkpoint_path).is_file():
        return None
    import torch
    from torchvision import models
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]
    names = [label for label, _ in sorted(labels.items(), key=lambda item: item[1])]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(names))
    model.load_state_dict(checkpoint["state_dict"])
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
