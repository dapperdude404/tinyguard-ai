"""Reproducible 29→50 label plan; preserves deployed class order.

Class names are research hypotheses until licensed, correctly labeled images exist.
Never fabricate an expanded model or alter the deployed class mapping.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

LEGACY_CLASS_COUNT = 29
TARGET_CLASS_COUNT = 50

# Candidate new crop/disease combinations: 5x3 + 3x2 = 21.
# These are NOT verified diagnoses or proof that a labeled dataset exists.
NEW_LABELS = (
    "Mango | Healthy",
    "Mango | Anthracnose",
    "Mango | Powdery Mildew",
    "Guava | Healthy",
    "Guava | Anthracnose",
    "Guava | Leaf Spot",
    "Banana | Healthy",
    "Banana | Black Sigatoka",
    "Banana | Yellow Sigatoka",
    "Rice | Healthy",
    "Rice | Brown Spot",
    "Rice | Leaf Blast",
    "Wheat | Healthy",
    "Wheat | Leaf Rust",
    "Wheat | Powdery Mildew",
    "Citrus | Healthy",
    "Citrus | Canker",
    "Maize | Southern Leaf Blight",
    "Maize | Southern Rust",
    "Papaya | Healthy",
    "Papaya | Leaf Curl",
)


def validate_class_names(names: list[str], expected: int | None = None) -> list[str]:
    if not isinstance(names, list) or not names or not all(isinstance(s, str) for s in names):
        raise ValueError("Class names must be a nonempty list of strings")
    if any(not name.strip() or len(name) > 160 for name in names):
        raise ValueError("Empty or oversized class name")
    normalized = [s.strip().casefold() for s in names]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate class name detected")
    if expected is not None and len(names) != expected:
        raise ValueError(f"Expected {expected} classes, got {len(names)}")
    return names


def canonical_crop_disease(name: str) -> tuple[str, str]:
    """Prevent semantic crop aliases from creating duplicate training targets."""
    crop, sep, condition = name.partition("|")
    if not sep or not condition.strip():
        raise ValueError(f"Crop/disease label lacks a | separator: {name}")
    crop_key = re.sub(r"[^a-z0-9]", "", crop.casefold())
    disease_key = re.sub(r"[^a-z0-9]", "", condition.casefold())
    if crop_key in {"cornmaize", "corn", "maize"}:
        crop_key = "maize"
    return crop_key, disease_key


def reject_alias_collisions(names: list[str]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for name in names:
        key = canonical_crop_disease(name)
        if key in seen:
            raise ValueError(f"Semantic alias collision: {seen[key]} vs {name}")
        seen[key] = name


def build_plan(legacy_names: list[str]) -> dict:
    validate_class_names(legacy_names, LEGACY_CLASS_COUNT)
    new = list(NEW_LABELS)
    validate_class_names(new, 21)
    combined = legacy_names + new
    validate_class_names(combined, TARGET_CLASS_COUNT)
    reject_alias_collisions(combined)
    classes = [
        {"id": f"c{index:03d}", "index": index, "name": name,
         "source": "legacy" if index < LEGACY_CLASS_COUNT else "new"}
        for index, name in enumerate(combined)
    ]
    return {
        "schema": "smartplant-class-plan-v1",
        "target_count": TARGET_CLASS_COUNT,
        "legacy_count": LEGACY_CLASS_COUNT,
        "new_count": len(new),
        "classes": classes,
    }


def read_plan(path: str | Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema") != "smartplant-class-plan-v1":
        raise ValueError("Unexpected class-plan schema")
    classes = plan.get("classes")
    if not isinstance(classes, list) or len(classes) != TARGET_CLASS_COUNT:
        raise ValueError("Expected exactly 50 class-plan entries")
    names = [item["name"] for item in classes]
    validate_class_names(names, TARGET_CLASS_COUNT)
    reject_alias_collisions(names)
    for i, item in enumerate(classes):
        if (item["id"] != f"c{i:03d}" or item["index"] != i or
                item["source"] != ("legacy" if i < LEGACY_CLASS_COUNT else "new")):
            raise ValueError("Class-plan indices / IDs are inconsistent")
    if names[LEGACY_CLASS_COUNT:] != list(NEW_LABELS):
        raise ValueError("New category catalog differs from the reviewed plan")
    if plan.get("target_count") != TARGET_CLASS_COUNT or plan.get("legacy_count") != LEGACY_CLASS_COUNT:
        raise ValueError("Class-plan counts inconsistent")
    return plan


def pinned_legacy_checkpoint_names(path: str | Path) -> list[str]:
    """Refuse even to deserialize if bytes differ from trusted legacy digest."""
    import torch
    from smartplant.model_download import MODEL_SHA256
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("Legacy checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    labels = checkpoint.get("labels")
    if not isinstance(labels, dict) or len(labels) != LEGACY_CLASS_COUNT:
        raise ValueError("Legacy checkpoint must carry 29 indexed labels")
    if set(labels.values()) != set(range(LEGACY_CLASS_COUNT)):
        raise ValueError("Legacy label indices must be 0..28")
    names = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
    return validate_class_names(names, LEGACY_CLASS_COUNT)
