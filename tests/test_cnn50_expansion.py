"""Reproducibility and class-order tests. No fabricated classification metrics."""
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from smartplant.expanded_labels import (
    NEW_LABELS, TARGET_CLASS_COUNT, build_plan, read_plan, validate_class_names
)
from smartplant.expanded_data import audit_dataset
from smartplant.model_registry import selected_cnn_checkpoint
from scripts.export_smartplant_onnx import ordered_names


LEGACY = [f"Legacy Crop {i} | Label {i}" for i in range(29)]


def test_50_class_plan_preserves_all_legacy_indices(tmp_path):
    plan = build_plan(LEGACY)
    assert len(plan["classes"]) == TARGET_CLASS_COUNT == 50
    assert [x["name"] for x in plan["classes"][:29]] == LEGACY
    assert [x["name"] for x in plan["classes"][29:]] == list(NEW_LABELS)
    assert [x["index"] for x in plan["classes"]] == list(range(50))
    assert plan["classes"][49]["id"] == "c049"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert read_plan(path) == plan


def test_requested_crops_and_distribution():
    counts = Counter(name.split(" | ", 1)[0] for name in NEW_LABELS)
    assert counts == {
        "Mango": 3, "Guava": 3, "Banana": 3, "Rice": 3,
        "Wheat": 3, "Citrus": 2, "Maize": 2, "Papaya": 2
    }


def test_duplicate_or_collision_fails_closed():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_class_names(["Mango | Healthy", " mango | healthy "])
    with pytest.raises(ValueError, match="Duplicate"):
        build_plan(LEGACY[:-1] + ["Mango | Healthy"])
    with pytest.raises(ValueError, match="Expected 29"):
        build_plan(LEGACY[:-1])


def test_corn_maize_alias_collisions_cannot_be_trained_twice():
    # Both of these old labels already exist in the deployed 29-class model.
    old = LEGACY[:27] + ["Corn (Maize) | Healthy", "Corn (Maize) | Northern Leaf Blight"]
    plan = build_plan(old)
    assert "Maize | Southern Rust" in [x["name"] for x in plan["classes"]]
    assert "Maize | Southern Leaf Blight" in [x["name"] for x in plan["classes"]]
    assert "Maize | Healthy" not in [x["name"] for x in plan["classes"][29:]]
    with pytest.raises(ValueError, match="Semantic alias collision"):
        build_plan(LEGACY[:28] + ["Corn (Maize) | Southern Rust"])


def test_corrupt_mapping_is_rejected(tmp_path):
    plan = build_plan(LEGACY)
    plan["classes"][38]["index"] = 0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="inconsistent"):
        read_plan(path)


def test_dynamic_checkpoint_label_validation():
    labels = {f"Crop {i}": i for i in range(50)}
    assert ordered_names({"labels": labels}) == list(labels)
    labels["Broken"] = 100
    with pytest.raises(ValueError, match="consecutive"):
        ordered_names({"labels": labels})


def _make_dataset(tmp_path: Path):
    root = tmp_path / "expanded"
    root.mkdir()
    plan_path = root / "class_plan.json"
    plan_path.write_text(json.dumps(build_plan(LEGACY)), encoding="utf-8")
    groups = []
    for split_index, split in enumerate(("train", "val", "test")):
        for index in range(50):
            directory = root / split / f"c{index:03d}"
            directory.mkdir(parents=True)
            rnd = random.Random(index * 71 + split_index * 27007)
            pixels = bytes(rnd.randrange(256) for _ in range(12 * 12 * 3))
            image = Image.frombytes("RGB", (12, 12), pixels)
            path = directory / "leaf.png"
            image.save(path)
            groups.append([path.relative_to(root).as_posix(), f"{split}-{index}"])
    csv_path = root / "groups.csv"
    with csv_path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["relative_path", "group_id"])
        writer.writerows(groups)
    return root, plan_path, csv_path


def test_audit_valid_50_classes_with_groups(tmp_path):
    root, plan, groups = _make_dataset(tmp_path)
    audit = audit_dataset(root, plan, (1, 1, 1), groups)
    assert audit["class_count"] == 50
    assert audit["counts"] == {"train": 50, "val": 50, "test": 50}
    assert audit["provenance_group_audit"] == "passed"


def test_exact_duplicates_across_splits_are_rejected(tmp_path):
    root, plan, _ = _make_dataset(tmp_path)
    shutil.copy2(root / "train/c000/leaf.png", root / "val/c000/leaf.png")
    with pytest.raises(ValueError, match="Exact duplicate"):
        audit_dataset(root, plan, (1, 1, 1), near_duplicate_distance=0)


def test_missing_labeled_images_are_rejected(tmp_path):
    root, plan, groups = _make_dataset(tmp_path)
    (root / "test/c049/leaf.png").unlink()
    with pytest.raises(ValueError, match="requires"):
        audit_dataset(root, plan, (1, 1, 1), groups)


def test_provenance_cross_split_rejected(tmp_path):
    root, plan, groups = _make_dataset(tmp_path)
    content = groups.read_text()
    # Two distinct images cannot be treated as independent if same source plant.
    content = content.replace("val/c000/leaf.png,val-0", "val/c000/leaf.png,train-0")
    groups.write_text(content)
    with pytest.raises(ValueError, match="Source group crosses"):
        audit_dataset(root, plan, (1, 1, 1), groups)


def test_opt_in_expanded_checkpoint_requires_sha(monkeypatch, tmp_path):
    path = tmp_path / "expanded.pt"
    path.write_bytes(b"checkpoint placeholder for checksum-only selection")
    monkeypatch.setenv("SMARTPLANT_EXPANDED_MODEL_PATH", str(path))
    monkeypatch.delenv("SMARTPLANT_EXPANDED_MODEL_SHA256", raising=False)
    with pytest.raises(ValueError, match="SHA256"):
        selected_cnn_checkpoint()
    monkeypatch.setenv("SMARTPLANT_EXPANDED_MODEL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        selected_cnn_checkpoint()
    monkeypatch.setenv("SMARTPLANT_EXPANDED_MODEL_SHA256",
                       hashlib.sha256(path.read_bytes()).hexdigest())
    assert selected_cnn_checkpoint() == (str(path), "expanded")

def test_unreviewed_image_formats_are_rejected(tmp_path):
    root, plan, _ = _make_dataset(tmp_path)
    # ImageFolder also supports BMP, so the audit must never silently skip it.
    image = Image.new("RGB", (12, 12), color=(12, 53, 102))
    image.save(root / "train/c000/extra.bmp")
    with pytest.raises(ValueError, match="Unreviewed file types"):
        audit_dataset(root, plan, (1, 1, 1))


def test_nested_data_are_rejected(tmp_path):
    root, plan, _ = _make_dataset(tmp_path)
    (root / "test/c005/hidden").mkdir()
    with pytest.raises(ValueError, match="Unexpected nested folders"):
        audit_dataset(root, plan, (1, 1, 1))
