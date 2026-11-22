"""Inspect existing SmartPlant_Datasets ZIPs directly from mounted Google Drive.

No external downloads or manual ZIP uploads. No class is relabeled, split or
trained automatically. Optional --stage-dir extracts only safe image members to
a staging area, NOT to train/val/test.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import stat
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

KNOWN_ARCHIVES = (
    "Apple.zip", "Bell Pepper.zip", "Cherry.zip", "Corn (Maize).zip",
    "Grape.zip", "Peach.zip", "Potato.zip", "Strawberry.zip", "Tomato.zip",
)
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILES = 180_000
MAX_ARCHIVE_UNCOMPRESSED = 6 * 1024**3
MAX_IMAGE_UNCOMPRESSED = 30 * 1024**2
MAX_COMPRESSION_RATIO = 250
CSV_FIELDS = ("archive", "archive_label", "internal_folder", "images",
              "uncompressed_bytes", "example_member", "mapping_status")


def safe_member(raw_name: str) -> PurePosixPath:
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise ValueError("Unsafe archive member name")
    if raw_name.startswith("/") or re.match(r"^[A-Za-z]:", raw_name):
        raise ValueError("Absolute archive member path")
    parts = raw_name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        # A trailing slash is not a file and is ignored before this function.
        raise ValueError("Archive traversal or empty path component")
    return PurePosixPath(*parts)


def inspect_archive(path: Path, stage_dir: Path | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, dict] = {}
    count = 0
    size = 0
    seen_paths: set[str] = set()
    with ZipFile(path, "r") as archive:
        entries = archive.infolist()
        if len(entries) > MAX_FILES:
            raise ValueError(f"Archive contains too many members: {path.name}")
        for info in entries:
            if info.is_dir():
                continue
            member = safe_member(info.filename)
            if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF):
                raise ValueError(f"ZIP symlink not allowed: {info.filename}")
            if member.suffix.casefold() not in IMAGE_TYPES:
                # Metadata and platform files are intentionally NOT training images.
                continue
            casefolded = member.as_posix().casefold()
            if casefolded in seen_paths:
                raise ValueError(f"Duplicate ZIP path: {info.filename}")
            seen_paths.add(casefolded)
            if info.file_size > MAX_IMAGE_UNCOMPRESSED:
                raise ValueError(f"Oversized image: {info.filename}")
            if info.file_size > MAX_COMPRESSION_RATIO * max(info.compress_size, 1):
                raise ValueError(f"Suspicious compression ratio: {info.filename}")
            size += info.file_size
            count += 1
            if size > MAX_ARCHIVE_UNCOMPRESSED:
                raise ValueError(f"Uncompressed ZIP size exceeds safety limit: {path.name}")
            internal = member.parent.as_posix()
            group = rows.setdefault(internal, {
                "archive": path.name,
                "archive_label": path.stem,
                "internal_folder": internal,
                "images": 0,
                "uncompressed_bytes": 0,
                "example_member": member.as_posix(),
                "mapping_status": "UNVERIFIED — review against 29 checkpoint labels",
            })
            group["images"] += 1
            group["uncompressed_bytes"] += info.file_size
            if stage_dir is not None:
                # Staging is separate from approved train/val/test and cannot by
                # itself be used as a model ground-truth evaluation dataset.
                dest = stage_dir / path.stem / Path(*member.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    raise FileExistsError(dest)
                with archive.open(info, "r") as src, dest.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
    return {"archive": path.name, "image_count": count,
            "uncompressed_image_bytes": size, "class_folder_candidates": list(rows.values())}


def scan_drive_archives(source_dir: Path, report_dir: Path,
                        stage_dir: Path | None = None) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    archive_summaries = []
    missing = []
    for name in KNOWN_ARCHIVES:
        path = source_dir / name
        if not path.exists():
            missing.append(name)
            continue
        result = inspect_archive(path, stage_dir=stage_dir)
        archive_summaries.append({k: v for k, v in result.items()
                                  if k != "class_folder_candidates"})
        all_rows.extend(result["class_folder_candidates"])
    csv_path = report_dir / "legacy_zip_class_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row:
                                (row["archive"].casefold(), row["internal_folder"].casefold())))
    result = {
        "source_directory": str(source_dir),
        "source_archive_names": list(KNOWN_ARCHIVES),
        "found_archives": len(archive_summaries),
        "missing_archives": missing,
        "image_count": sum(x["image_count"] for x in archive_summaries),
        "archive_summaries": archive_summaries,
        "unverified_class_folders": len(all_rows),
        "images_staged": stage_dir is not None,
        "review_status": "NOT READY FOR TRAINING: verify labels, provenance, duplicates, rights, group split",
        "provenance": "Archive/ZIP folder names are NOT source plant/farm group IDs.",
    }
    report_path = report_dir / "legacy_zip_inventory.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Found", result["found_archives"], "of", len(KNOWN_ARCHIVES), "Drive crop archives")
    print("Image filenames indexed (NOT validated):", result["image_count"])
    print("Potential internal class folders:", len(all_rows))
    print("Reports:", report_path, "and", csv_path)
    if missing:
        print("MISSING:", ", ".join(missing))
    print(result["review_status"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",
                        default="/content/drive/MyDrive/SmartPlant_Datasets")
    parser.add_argument("--report-dir",
                        default="/content/drive/MyDrive/SmartPlant50/audit_reports")
    parser.add_argument("--stage-dir", default=None,
                        help="Optional staged image extraction; NOT train/val/test. "
                             "Use /content/smartplant50_legacy_staging for temporary files.")
    args = parser.parse_args()
    scan_drive_archives(Path(args.source_dir), Path(args.report_dir),
                        Path(args.stage_dir) if args.stage_dir else None)


if __name__ == "__main__":
    main()
