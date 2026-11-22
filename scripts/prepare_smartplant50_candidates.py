"""Generate class-folder review candidates from Drive ZIPs and quarantined sources.

No images extracted, uploaded or approved. Exact normalized label matches are
suggestions only; aliases and missing source-plant groups must be reviewed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import stat
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from scripts.scan_smartplant_drive_archives import IMAGE_TYPES, safe_member

FIELDS = [
    "source_archive", "archive_sha256", "internal_folder", "image_count",
    "suggested_class_id", "suggested_label", "matching", "reviewed_class_id",
    "license_checked", "taxonomy_checked", "source_group_csv", "review_note",
]
MAX_MEMBERS = 200_000


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def canonical(text: str) -> str:
    text = re.sub(r"(?i)\bcorn\s*\(maize\)", "maize", text)
    text = re.sub(r"(?i)\bcorn\b", "maize", text)
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def proposal_for_folder(folder: str, labels: list[str]) -> tuple[str, str, str]:
    # The rightmost path segment is the original provider's proposed class label.
    # e.g. Potato___Early_blight -> Potato | Early blight.
    last = folder.rsplit("/", 1)[-1]
    candidate = last.replace("___", " | ").replace("_", " ").strip()
    key = canonical(candidate)
    matches = [(index, name) for index, name in enumerate(labels)
               if canonical(name) == key]
    if len(matches) == 1:
        idx, label = matches[0]
        return f"c{idx:03d}", label, "EXACT_NORMALIZED_SUGGESTION_ONLY"
    return "", "", "UNRESOLVED_REVIEW_REQUIRED"


def find_archives(legacy_dir: Path, workspace: Path) -> list[Path]:
    result = []
    if legacy_dir.is_dir():
        result += [p for p in legacy_dir.glob("*.zip") if p.is_file() and not p.is_symlink()]
    quarantine = workspace / "source_quarantine"
    if quarantine.is_dir():
        result += [p for p in quarantine.glob("*/*.zip") if p.is_file() and not p.is_symlink()]
    return sorted(set(result))


def inspect_zip(path: Path, labels: list[str]) -> list[dict]:
    groups = defaultdict(int)
    with ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_MEMBERS:
            raise ValueError("Archive contains too many entries")
        for info in entries:
            if info.is_dir():
                continue
            member = safe_member(info.filename)
            if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF):
                raise ValueError("Archive contains symlink")
            if member.suffix.lower() not in IMAGE_TYPES:
                continue
            if info.file_size > 30 * 1024 * 1024:
                raise ValueError("Image member is too large")
            if info.file_size > 250 * max(info.compress_size, 1):
                raise ValueError("Suspicious compression ratio")
            groups[member.parent.as_posix()] += 1
    digest = fingerprint(path)
    result = []
    for folder, count in sorted(groups.items()):
        suggestion, name, kind = proposal_for_folder(folder, labels)
        result.append({
            "source_archive": str(path.resolve()),
            "archive_sha256": digest,
            "internal_folder": folder,
            "image_count": str(count),
            "suggested_class_id": suggestion,
            "suggested_label": name,
            "matching": kind,
            "reviewed_class_id": "",
            "license_checked": "no",
            "taxonomy_checked": "no",
            "source_group_csv": "",
            "review_note": "Unapproved. Provide license evidence, diagnosis review, "
                           "and source-plant/farm groups before verified import.",
        })
    return result


def generate_review(legacy_dir: Path, workspace: Path, plan: Path,
                    output: Path) -> dict:
    catalog = json.loads(plan.read_text(encoding="utf-8"))
    entries = catalog.get("classes")
    if not isinstance(entries, list) or len(entries) != 50:
        raise ValueError("Reviewed 50-class plan with indices required")
    labels = [item["name"] for item in entries]
    if any(item["index"] != i or item["id"] != f"c{i:03d}"
           for i, item in enumerate(entries)):
        raise ValueError("Class plan indices are inconsistent")
    rows = []
    archives = find_archives(legacy_dir, workspace)
    for archive in archives:
        rows.extend(inspect_zip(archive, labels))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Review CSV already exists. Never overwrite prior approvals.")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "archives_discovered": len(archives), "folders_review_required": len(rows),
        "exact_suggestions": sum(x["suggested_class_id"] != "" for x in rows),
        "approved": 0, "review_csv": str(output),
        "note": "Suggestions are not verified labels. No files imported.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-dir", default="/content/drive/MyDrive/SmartPlant_Datasets")
    p.add_argument("--workspace", default="/content/drive/MyDrive/SmartPlant50")
    p.add_argument("--plan", default="/content/drive/MyDrive/SmartPlant50/data/expanded/class_plan.json")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    workspace = Path(a.workspace)
    output = Path(a.output) if a.output else (
        workspace / "audit_reports" / "source_folder_review_candidates.csv"
    )
    result = generate_review(Path(a.legacy_dir), workspace, Path(a.plan), output)
    print(json.dumps(result, indent=2))
    print("Complete reviewed_class_id, evidence flags and source_group_csv after reviewing source data.")


if __name__ == "__main__":
    main()
