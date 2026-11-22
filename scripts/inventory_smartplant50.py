"""Quick, read-only inventory of the 50-class data folder.

This is a *structural* preflight only. Passing it does not verify image bytes,
taxonomy, duplicates, licensing, plant independence, or model performance.
Those require the separate strict dataset audit from the research model branch.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SPLITS = ("train", "val", "test")
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
MIN_COUNTS = {"train": 20, "val": 5, "test": 5}


def inventory_dataset(path: str | Path) -> dict:
    root = Path(path).expanduser()
    counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    issues: list[str] = []
    if not root.is_dir():
        return {"dataset": str(root), "present": False, "counts": counts,
                "gaps": ["Dataset directory is missing: " + str(root)],
                "ready_for_strict_audit": False,
                "note": "Structural inventory only; NOT ground-truth or scientific verification"}
    files_seen: set[str] = set()
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            issues.append(f"Missing {split}/ directory")
            continue
        other_folders = [p.name for p in split_dir.iterdir()
                         if p.is_dir() and p.name not in {f"c{i:03d}" for i in range(50)}]
        if other_folders:
            issues.append(f"Unexpected classes in {split}: {other_folders[:8]}")
        for number in range(50):
            cid = f"c{number:03d}"
            folder = split_dir / cid
            if not folder.is_dir():
                counts[split][cid] = 0
                issues.append(f"Missing {split}/{cid}")
                continue
            entries = list(folder.iterdir())
            images = [p for p in entries if p.is_file() and
                      not p.is_symlink() and p.suffix.casefold() in IMAGE_TYPES]
            invalid = [p.name for p in entries if p not in images]
            if invalid:
                issues.append(f"Unreviewed files or nested folders in {split}/{cid}: {invalid[:5]}")
            counts[split][cid] = len(images)
            if len(images) < MIN_COUNTS[split]:
                issues.append(f"{split}/{cid}: {len(images)} < {MIN_COUNTS[split]} minimum")
            for file in images:
                files_seen.add(file.relative_to(root).as_posix())
    provenance = root / "groups.csv"
    if not provenance.is_file():
        issues.append("Missing groups.csv (original source plant/farm provenance)")
    else:
        entries: dict[str, str] = {}
        split_by_group: dict[str, str] = {}
        with provenance.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"relative_path", "group_id"}.issubset(reader.fieldnames or []):
                issues.append("groups.csv needs relative_path,group_id columns")
            else:
                for row in reader:
                    rel = (row.get("relative_path") or "").replace("\\", "/").strip()
                    group = (row.get("group_id") or "").strip()
                    if not rel or rel in entries or not group:
                        issues.append("Invalid or duplicate groups.csv row")
                        continue
                    entries[rel] = group
                    split = rel.split("/", 1)[0]
                    previous = split_by_group.get(group)
                    if previous and previous != split:
                        issues.append(f"Plant/farm group spans multiple splits: {group}")
                    split_by_group[group] = split
                unmatched = files_seen.symmetric_difference(entries.keys())
                if unmatched:
                    issues.append(f"groups.csv mismatch: {len(unmatched)} missing/extra file records")
    return {
        "dataset": str(root),
        "present": True,
        "counts": counts,
        "minimums": MIN_COUNTS,
        "total_images": sum(sum(n.values()) for n in counts.values()),
        "gaps": issues,
        "ready_for_strict_audit": not issues,
        "note": "Structural inventory only; NOT ground-truth, duplicate, license or model verification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/expanded")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = inventory_dataset(args.dataset)
    print("Image count:", report.get("total_images", 0))
    print("Next-stage audit gate:", "STRUCTURALLY READY" if report["ready_for_strict_audit"] else "BLOCKED")
    for gap in report["gaps"][:100]:
        print("NEEDS DATA:", gap)
    if len(report["gaps"]) > 100:
        print("... additional data gaps:", len(report["gaps"]) - 100)
    print(report["note"])
    if args.require_complete and not report["ready_for_strict_audit"]:
        raise SystemExit("Dataset incomplete. No training started.")


if __name__ == "__main__":
    main()
