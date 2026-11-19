"""Train a transparent baseline on a locally provided labeled image dataset.

Dataset layout:
data/train/Healthy/*.jpg
data/train/Tomato_Early_blight/*.jpg
data/train/Tomato_Late_blight/*.jpg
"""
import argparse
import json
from pathlib import Path
from smartplant.core import fit_model, load_dataset, save_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train")
    parser.add_argument("--output", default="models/plant_model.joblib")
    args = parser.parse_args()
    images, labels = load_dataset(args.data)
    model, metrics = fit_model(images, labels)
    save_model(model, args.output)
    path = Path(args.output).with_suffix(".metrics.json")
    path.write_text(json.dumps(metrics, indent=2))
    print(f"Trained on {model.dataset_size} images, validation accuracy: {model.validation_accuracy:.3f}")
    print(f"Model: {args.output}; metrics: {path}")
    print("Warning: random image split can overestimate generalization if duplicate or related leaves cross splits.")

if __name__ == "__main__":
    main()
