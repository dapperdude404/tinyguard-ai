"""Transparent image features, trained classifier and reproducible evaluation."""
from dataclasses import dataclass
from pathlib import Path
import io
import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from skimage.feature import hog

SIZE = (128, 128)
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}

def decode_image(content):
    if len(content) > 12 * 1024 * 1024:
        raise ValueError("Maximum upload size is 12 MB")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image = ImageOps.exif_transpose(image)
            if image.width * image.height > 24_000_000:
                raise ValueError("Image resolution is too large")
            return image.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid image: {exc}") from exc

def extract_features(image):
    arr = np.asarray(image.convert("RGB").resize(SIZE), dtype=np.float32) / 255
    gray = rgb2gray(arr)
    edges = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), feature_vector=True)
    hist = np.concatenate([np.histogram(arr[:, :, c], bins=16, range=(0, 1), density=True)[0]
                           for c in range(3)])
    return np.concatenate([edges, hist, arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))]).astype(np.float32)

def leaf_observations(image):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green = (g > r * 1.15) & (g > b * 1.08)
    yellow = (r > .35) & (g > .3) & (b < .75 * ((r + g) / 2))
    brown = (r > g * 1.18) & (g > b * 1.08) & (r < .8)
    return {"green_pixels_pct": round(float(green.mean() * 100), 1),
            "yellow_pixels_pct": round(float(yellow.mean() * 100), 1),
            "brown_pixels_pct": round(float(brown.mean() * 100), 1)}

@dataclass
class PlantModel:
    classifier: object
    labels: list
    validation_accuracy: float
    dataset_size: int

def fit_model(images, labels, seed=42):
    if len(images) != len(labels):
        raise ValueError("Image and label counts differ")
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or min(counts) < 5:
        raise ValueError("Need at least 2 classes with 5 images each")
    x = np.stack([extract_features(im) for im in images])
    xtrain, xval, ytrain, yval = train_test_split(
        x, labels, test_size=.25, random_state=seed, stratify=labels)
    model = RandomForestClassifier(n_estimators=120, class_weight="balanced",
                                   random_state=seed, n_jobs=-1)
    model.fit(xtrain, ytrain)
    predictions = model.predict(xval)
    return PlantModel(model, list(model.classes_),
                      float(accuracy_score(yval, predictions)), len(images)), {
        "report": classification_report(yval, predictions, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(yval, predictions, labels=model.classes_).tolist(),
        "labels": list(model.classes_),
        "validation_count": len(yval)
    }

def predict(model, image):
    probabilities = model.classifier.predict_proba(extract_features(image)[None, :])[0]
    ranking = np.argsort(probabilities)[::-1]
    return [(str(model.classifier.classes_[i]), float(probabilities[i])) for i in ranking]

def load_dataset(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("Dataset directory not found")
    images, labels = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        for file in sorted(folder.iterdir()):
            if file.suffix.lower() not in SUPPORTED:
                continue
            try:
                images.append(decode_image(file.read_bytes()))
                labels.append(folder.name)
            except ValueError:
                continue
    return images, labels

def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)  # Load only a trusted, locally produced artifact.
