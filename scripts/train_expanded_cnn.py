"""Fine-tune MobileNetV3-Small from SHA-pinned 29-class weights on all 50 classes.

Requires a real, legally obtained, provenance-audited image dataset.
Never uses synthetic images to report accuracy or claims a trained model exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from smartplant.expanded_data import audit_dataset
from smartplant.expanded_labels import (
    LEGACY_CLASS_COUNT, TARGET_CLASS_COUNT, pinned_legacy_checkpoint_names, read_plan
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fine_tune_model(legacy_checkpoint_path: Path, plan: dict) -> nn.Module:
    """Copy all old class-head rows 0..28; initialize only new rows 29..49."""
    original_names = pinned_legacy_checkpoint_names(legacy_checkpoint_path)
    new_names = [entry["name"] for entry in plan["classes"]]
    if original_names != new_names[:LEGACY_CLASS_COUNT]:
        raise ValueError("Expanded class plan changes legacy label ordering")
    checkpoint = torch.load(legacy_checkpoint_path, map_location="cpu", weights_only=True)
    model = models.mobilenet_v3_small(weights=None)
    original_layer = model.classifier[-1]
    model.classifier[-1] = nn.Linear(original_layer.in_features, LEGACY_CLASS_COUNT)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    old_head = model.classifier[-1]
    new_head = nn.Linear(old_head.in_features, TARGET_CLASS_COUNT)
    with torch.no_grad():
        new_head.weight[:LEGACY_CLASS_COUNT].copy_(old_head.weight)
        new_head.bias[:LEGACY_CLASS_COUNT].copy_(old_head.bias)
    model.classifier[-1] = new_head
    return model


def _dataset(folder: Path, is_training: bool):
    prep = [
        transforms.Resize(176),
        transforms.CenterCrop(160),
    ]
    if is_training:
        prep = [
            transforms.RandomResizedCrop(160, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15, hue=.03),
        ]
    transform = transforms.Compose([
        *prep, transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    ds = datasets.ImageFolder(folder, transform=transform)
    expected_ids = [f"c{i:03d}" for i in range(TARGET_CLASS_COUNT)]
    if ds.classes != expected_ids:
        raise ValueError(f"Dataset folder indexing differs from class plan: {folder}")
    return ds


def _evaluate(model, loader, device, names: list[str]) -> tuple[dict, np.ndarray]:
    model.eval()
    all_true, all_pred = [], []
    with torch.inference_mode():
        for x, labels in loader:
            logits = model(x.to(device))
            predictions = logits.argmax(1).cpu().tolist()
            all_true.extend(labels.tolist())
            all_pred.extend(predictions)
    if not all_true:
        raise ValueError("Evaluation dataset is empty")
    prec, recall, f1, support = precision_recall_fscore_support(
        all_true, all_pred, labels=list(range(len(names))), zero_division=0
    )
    matrix = confusion_matrix(all_true, all_pred, labels=list(range(len(names))))
    report = {
        "samples": len(all_true),
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "legacy_accuracy": float(np.mean([
            p == y for y, p in zip(all_true, all_pred) if y < LEGACY_CLASS_COUNT
        ])),
        "new_crop_accuracy": float(np.mean([
            p == y for y, p in zip(all_true, all_pred) if y >= LEGACY_CLASS_COUNT
        ])),
        "per_class": [{
            "label": names[i], "precision": float(prec[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i])
        } for i in range(len(names))]
    }
    return report, matrix


def _write_matrix(path: Path, matrix: np.ndarray, names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true / predicted", *names])
        for label, row in zip(names, matrix):
            writer.writerow([label, *[int(x) for x in row]])


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.lr <= 0:
        raise ValueError("Invalid training hyperparameters")
    seed_everything(args.seed)
    plan = read_plan(args.plan)
    old_names = pinned_legacy_checkpoint_names(args.base_checkpoint)
    names = [item["name"] for item in plan["classes"]]
    if names[:LEGACY_CLASS_COUNT] != old_names:
        raise ValueError("Expanded plan does not preserve original 29-class order")
    provenance = audit_dataset(
        args.dataset, args.plan, (args.min_train, args.min_val, args.min_test),
        groups_csv=args.groups_csv,
    )
    if not args.groups_csv:
        print("WARNING: No plant/farm groups.csv; source-level split independence unverified.")
    root = Path(args.dataset)
    train_ds = _dataset(root / "train", is_training=True)
    val_ds = _dataset(root / "val", is_training=False)
    test_ds = _dataset(root / "test", is_training=False)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = fine_tune_model(Path(args.base_checkpoint), plan).to(device)
    counts = np.bincount([idx for _, idx in train_ds.samples], minlength=TARGET_CLASS_COUNT)
    class_weights = torch.tensor(np.sqrt(counts.max() / counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    best_score, best_epoch, no_improve = -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, samples = 0.0, 0
        for image, target in train_loader:
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image), target)
            loss.backward()
            optimizer.step()
            count = target.size(0)
            samples += count
            loss_sum += float(loss.detach().item()) * count
        scheduler.step()
        val_metrics, _ = _evaluate(model, val_loader, device, names)
        summary = {
            "epoch": epoch,
            "train_loss": loss_sum / max(samples, 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(summary)
        print(json.dumps(summary))
        if val_metrics["macro_f1"] > best_score + 1e-6:
            best_score, best_epoch, no_improve = val_metrics["macro_f1"], epoch, 0
            torch.save({
                "architecture": "mobilenet_v3_small",
                "input_size": 160,
                "labels": {name: i for i, name in enumerate(names)},
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "training": {"seed": args.seed, "epoch": best_epoch,
                             "best_val_macro_f1": best_score},
            }, output)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Stopping early at epoch {epoch} (best epoch {best_epoch})")
                break

    # Only evaluate test once, after model selection on validation split.
    safe_best = torch.load(output, map_location="cpu", weights_only=True)
    model.load_state_dict(safe_best["state_dict"], strict=True)
    model.to(device)
    test_metrics, confusion = _evaluate(model, test_loader, device, names)
    report = {
        "status": "research-evaluation; not independently verified field accuracy",
        "model": "MobileNetV3-Small, 50 classes",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "test": test_metrics,
        "split_audit": provenance,
        "hyperparameters": {
            "epochs_requested": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "patience": args.patience, "device": str(device),
            "weighted_cross_entropy": "inverse sqrt train-class frequency",
        },
        "history": history,
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    out_report = output.with_suffix(".metrics.json")
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    confusion_path = output.with_suffix(".confusion.csv")
    _write_matrix(confusion_path, confusion, names)
    print(f"Saved SHA-pinnable weights-only checkpoint: {output}")
    print(f"SHA-256: {report['checkpoint_sha256']}")
    print(f"Test metrics: {out_report}; confusion: {confusion_path}")
    print("Results are not independent field-performance evidence until source-group and field tests pass.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/expanded")
    parser.add_argument("--plan", default="data/expanded/class_plan.json")
    parser.add_argument("--base-checkpoint", default="models/smartplant_mobilenetv3.pt")
    parser.add_argument("--output", default="models/smartplant_50.pt")
    parser.add_argument("--groups-csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--min-train", type=int, default=20)
    parser.add_argument("--min-val", type=int, default=5)
    parser.add_argument("--min-test", type=int, default=5)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
