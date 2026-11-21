"""Audit externally supplied datasets before research training/evaluation.

This module NEVER downloads data or manufactures ground truth labels.
Directory names are stable cNNN IDs rather than class names, avoiding invalid
filename characters and preserving the original 29 label positions.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps

from smartplant.expanded_labels import read_plan

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")


def difference_hash(path: Path) -> int:
    """64-bit image difference hash; only a screening aid, not identity proof."""
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L").resize((9, 8))
        pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_groups(root: Path, path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    group_by_path: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not {"relative_path", "group_id"}.issubset(reader.fieldnames or []):
            raise ValueError("groups.csv requires relative_path,group_id headers")
        for row in reader:
            rel = row["relative_path"].replace("\\", "/")
            group = row["group_id"].strip()
            if not rel or not group or rel in group_by_path or rel.startswith("/") or ".." in Path(rel).parts:
                raise ValueError(f"Invalid / duplicate group entry: {rel}")
            group_by_path[rel] = group
    return group_by_path


def audit_dataset(root: str | Path, plan_path: str | Path,
                  min_per_split: tuple[int, int, int] = (20, 5, 5),
                  groups_csv: str | Path | None = None,
                  near_duplicate_distance: int = 5) -> dict:
    if not 0 <= near_duplicate_distance <= 7:
        raise ValueError("near_duplicate_distance must be 0..7")
    if len(min_per_split) != 3 or any(n < 1 for n in min_per_split):
        raise ValueError("min_per_split must contain three positive counts")
    root = Path(root).resolve()
    plan = read_plan(plan_path)
    ids = [item["id"] for item in plan["classes"]]
    groups = _read_groups(root, Path(groups_csv)) if groups_csv else None
    totals = Counter()
    file_hashes: dict[str, tuple[str, str]] = {}
    group_splits: dict[str, str] = {}
    # 8 independent hash-byte buckets make all pairs within <=7 bit flips comparable.
    buckets: dict[tuple[int, int], list[tuple[str, int, str]]] = {}
    reviewed: list[str] = []
    for split, minimum in zip(SPLITS, min_per_split):
        base = root / split
        if not base.is_dir():
            raise ValueError(f"Required dataset split missing: {base}")
        extraneous = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name not in ids)
        if extraneous:
            raise ValueError(f"Unexpected class IDs in {split}: {extraneous[:5]}")
        for class_id in ids:
            folder = base / class_id
            if not folder.is_dir():
                raise ValueError(f"Missing {split}/{class_id}: all 50 classes need images")
            entries = list(folder.iterdir())
            nested = [p.name for p in entries if p.is_dir()]
            if nested:
                raise ValueError(f"Unexpected nested folders in {split}/{class_id}: {nested[:5]}")
            unreviewed = [p.name for p in entries if p.is_file() and p.suffix.lower() not in IMAGE_EXTENSIONS]
            if unreviewed:
                raise ValueError(f"Unreviewed file types in {split}/{class_id}: {unreviewed[:5]}")
            files = sorted(p for p in entries
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if len(files) < minimum:
                raise ValueError(f"{split}/{class_id} has {len(files)} images, requires {minimum}")
            totals[split] += len(files)
            for file in files:
                relative = file.relative_to(root).as_posix()
                if groups is not None:
                    if relative not in groups:
                        raise ValueError(f"Missing provenance group: {relative}")
                    group = groups[relative]
                    other_split = group_splits.get(group)
                    if other_split is not None and other_split != split:
                        raise ValueError(f"Source group crosses {other_split} and {split}: {group}")
                    group_splits[group] = split
                digest = sha256_file(file)
                previous = file_hashes.get(digest)
                if previous and previous[0] != split:
                    raise ValueError(f"Exact duplicate across splits: {previous[1]} and {relative}")
                file_hashes[digest] = (split, relative)
                phash = difference_hash(file)  # decodes / validates image
                candidates: set[tuple[str, int, str]] = set()
                for byte in range(8):
                    shift = byte * 8
                    candidates.update(buckets.get((byte, (phash >> shift) & 255), []))
                for other_split, other_hash, other_path in candidates:
                    if other_split != split and (phash ^ other_hash).bit_count() <= near_duplicate_distance:
                        raise ValueError(
                            f"Possible near-duplicate across splits (review images): "
                            f"{other_path} versus {relative}"
                        )
                record = (split, phash, relative)
                for byte in range(8):
                    buckets.setdefault((byte, (phash >> (byte * 8)) & 255), []).append(record)
                reviewed.append(relative)
    if groups is not None and set(groups) != set(reviewed):
        raise ValueError("groups.csv contains entries outside the audited image set")
    return {
        "counts": {split: totals[split] for split in SPLITS},
        "class_count": len(ids),
        "exact_duplicate_audit": "passed",
        "perceptual_hash_screening": "passed (64-bit dHash; NOT proof of no overlap)",
        "provenance_group_audit": "passed" if groups is not None else "not supplied",
        "near_duplicate_distance": near_duplicate_distance,
    }
