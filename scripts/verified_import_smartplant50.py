"""Fail-closed import of approved, checksum-pinned source images into a NEW candidate split.

Review is explicit and auditable: every ZIP folder must have a reviewed class ID,
licence and taxonomy evidence, and a per-original-photo source-plant/farm mapping.
No images are fabricated. No existing training, APK, production app, or checkpoint
is overwritten. Unapproved sources cannot enter the candidate dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import stat
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageOps

from scripts.prepare_smartplant50_candidates import FIELDS, fingerprint
from scripts.scan_smartplant_drive_archives import IMAGE_TYPES, safe_member

MINIMUM = {"train": 20, "val": 5, "test": 5}
MAX_IMAGE_SIZE = 30 * 1024 * 1024
MAX_IMAGES = 250_000


def row_approved(row: dict) -> bool:
    return (bool(row["reviewed_class_id"].strip()) and
            row["license_checked"].strip().lower() == "yes" and
            row["taxonomy_checked"].strip().lower() == "yes" and
            bool(row["source_group_csv"].strip()))


def split_for_group(group: str) -> str:
    # Deterministic by original plant/farm group ID, not random by image.
    h = hashlib.sha256(("smartplant50:v1:" + group).encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big") % 100
    return "train" if n < 80 else ("val" if n < 90 else "test")


def perceptual_hash(image: Image.Image) -> int:
    thumb = ImageOps.exif_transpose(image).convert("L").resize((9, 8))
    pixels = list(thumb.getdata())
    result = 0
    for r in range(8):
        for c in range(8):
            result = (result << 1) | int(pixels[r * 9 + c] > pixels[r * 9 + c + 1])
    return result


def validated_image(raw: bytes) -> tuple[str, int]:
    if not raw or len(raw) > MAX_IMAGE_SIZE:
        raise ValueError("Empty or oversized image")
    with Image.open(io.BytesIO(raw)) as image:
        if image.width < 64 or image.height < 64:
            raise ValueError("Image too small to assess reliably")
        if image.width * image.height > 100_000_000:
            raise ValueError("Excessive image pixels")
        image.load()
        if image.format not in ("PNG", "JPEG", "WEBP"):
            raise ValueError("Unsupported actual image format")
        return hashlib.sha256(raw).hexdigest(), perceptual_hash(image)


def group_manifest(path: Path) -> dict[str, str]:
    groups = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"member_path", "source_group_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Source group CSV needs member_path,source_group_id")
        for row in reader:
            member = safe_member(row["member_path"].strip()).as_posix()
            group = row["source_group_id"].strip()
            if not group or member in groups:
                raise ValueError("Missing source group or duplicate ZIP member")
            groups[member] = group
    return groups


def load_plan(plan_path: Path) -> dict[str, str]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    items = plan.get("classes", [])
    if len(items) != 50 or any(
        item["id"] != f"c{i:03d}" or item["index"] != i or not item["name"]
        for i, item in enumerate(items)
    ):
        raise ValueError("Invalid 50-class plan")
    labels = [item["name"] for item in items]
    if len(set(labels)) != 50:
        raise ValueError("Duplicate planned classes")
    return {item["id"]: item["name"] for item in items}


def verify_review_row(row: dict, allowed: set[str], names: dict[str, str]):
    archive_path = Path(row["source_archive"]).resolve(strict=True)
    if not archive_path.is_file() or archive_path.suffix.lower() != ".zip":
        raise ValueError("Approved source must be a ZIP archive")
    if not any(archive_path.is_relative_to(Path(root)) for root in allowed):
        raise ValueError("Archive outside approved Drive root folders")
    if not row["archive_sha256"] or fingerprint(archive_path) != row["archive_sha256"].lower():
        raise ValueError("Source archive checksum changed since review")
    class_id = row["reviewed_class_id"].strip()
    if class_id not in names:
        raise ValueError("Reviewed class ID not present in 50-class plan")
    suggested = row["suggested_class_id"].strip()
    if (row["matching"] == "EXACT_NORMALIZED_SUGGESTION_ONLY"
            and suggested != class_id):
        raise ValueError("Approved reviewed class conflicts with suggested mapping")
    groups_csv = Path(row["source_group_csv"].strip()).expanduser().resolve(strict=True)
    if not groups_csv.is_file() or not groups_csv.is_relative_to(Path(allowed_group_root(allowed))):
        raise ValueError("Group CSV must be inside the user's SmartPlant50 Drive workspace")
    return archive_path, groups_csv, class_id


def allowed_group_root(roots: set[str]) -> str:
    # Explicitly configured workspace root is always tagged with /SmartPlant50.
    workspaces = [r for r in roots if Path(r).name == "SmartPlant50"]
    if len(workspaces) != 1:
        raise ValueError("Workspace root missing or ambiguous")
    return workspaces[0]


def audit_and_plan(review_csv: Path, class_plan: Path, workspace: Path,
                   legacy_dir: Path, max_images: int = MAX_IMAGES) -> tuple[dict, list[dict]]:
    names = load_plan(class_plan)
    with review_csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Reviewed source CSV columns changed")
        rows = list(reader)
    allowed_roots = {str(workspace.resolve()), str(legacy_dir.resolve())}
    allowed_folder = Path(allowed_group_root(allowed_roots))
    checked: list[dict] = []
    hash_to_original: dict[str, tuple[str, str, str]] = {}
    hash_groups: dict[str, tuple[str, int, str]] = {}
    # A 64-bit dHash within <=5 bit flips must match in at least one of
    # 8 disjoint bytes. Use buckets to avoid O(N^2) image comparisons.
    near_buckets: dict[tuple[int, int], list[str]] = {}
    count = Counter()
    unapproved_count = 0
    repeated = set()
    total_bytes = 0
    for row in rows:
        if not row_approved(row):
            unapproved_count += 1
            continue
        archive, groups_csv, class_id = verify_review_row(row, allowed_roots, names)
        folder = row["internal_folder"]
        key = (str(archive), folder)
        if key in repeated:
            raise ValueError("Reviewed source folder repeated")
        repeated.add(key)
        groups = group_manifest(groups_csv)
        image_members = []
        with ZipFile(archive) as zf:
            for item in zf.infolist():
                if item.is_dir():
                    continue
                member = safe_member(item.filename).as_posix()
                if stat.S_ISLNK((item.external_attr >> 16) & 0xFFFF):
                    raise ValueError("Source archive contains a symbolic link")
                if Path(member).suffix.lower() not in IMAGE_TYPES:
                    continue
                if str(Path(member).parent).replace("\\", "/") != folder:
                    continue
                if item.file_size > MAX_IMAGE_SIZE:
                    raise ValueError("Source member exceeds image byte limit")
                image_members.append(item)
            if len(image_members) != int(row["image_count"]):
                raise ValueError("Reviewed ZIP folder image count no longer matches")
            if not image_members:
                raise ValueError("Approved ZIP folder has no images")
            expected = {x.filename for x in image_members}
            if expected != set(groups):
                raise ValueError("Source group CSV must cover exactly every image in approved folder")
            for item in image_members:
                member = item.filename
                total_bytes += item.file_size
                if total_bytes > 5 * 1024**3 or len(checked) >= max_images:
                    raise ValueError("Import exceeds configured byte/file budget")
                if item.file_size > 250 * max(item.compress_size, 1):
                    raise ValueError("Suspicious compressed image size")
                raw = zf.read(item)
                digest, dhash = validated_image(raw)
                source_group = groups[member]
                # Include independent source fingerprint in group ID to distinguish
                # same name from unrelated source datasets, without splitting photos
                # of the same actual plant within one archive across partitions.
                full_group = row["archive_sha256"] + ":" + source_group
                split = split_for_group(full_group)
                prev = hash_to_original.get(digest)
                if prev:
                    prev_class, prev_split, prev_source = prev
                    if prev_class != class_id or prev_split != split:
                        raise ValueError("Cross-class or cross-split exact duplicate detected")
                    # Same-class same-split duplicates are excluded from output.
                    continue
                candidates = set()
                for byte in range(8):
                    candidates.update(near_buckets.get((byte, (dhash >> (byte * 8)) & 255), []))
                for other_digest in candidates:
                    other_split, other_hash, other_class = hash_groups[other_digest]
                    distance = (dhash ^ other_hash).bit_count()
                    if other_split != split and distance <= 5:
                        raise ValueError("Likely near-duplicate image crosses splits")
                    if other_class != class_id and distance <= 1:
                        raise ValueError("Likely near-duplicate image has conflicting diagnosis")
                hash_to_original[digest] = (class_id, split, member)
                hash_groups[digest] = (split, dhash, class_id)
                for byte in range(8):
                    near_buckets.setdefault((byte, (dhash >> (byte * 8)) & 255), []).append(digest)
                checked.append({
                    "archive": str(archive), "member": member,
                    "archive_sha256": row["archive_sha256"], "original_sha256": digest,
                    "class_id": class_id, "label": names[class_id],
                    "source_group_id": full_group, "split": split,
                    "bytes": len(raw),
                })
                count[(split, class_id)] += 1
    missing = [f"{split}/{cid}: {count[(split, cid)]} < {minimum}"
               for cid in names for split, minimum in MINIMUM.items()
               if count[(split, cid)] < minimum]
    report = {
        "schema": "smartplant-approved-import-v1",
        "review_rows": len(rows), "unapproved_rows": unapproved_count,
        "reviewed_files": len(checked),
        "split_counts": {split: sum(count[(split, cid)] for cid in names)
                         for split in MINIMUM},
        "class_gaps": missing,
        "ready_for_full_50": not missing and unapproved_count == 0,
        "status": ("CANDIDATE_READY_FOR_FINAL_SCIENTIFIC_REVIEW"
                   if not missing and unapproved_count == 0
                   else "BLOCKED: incomplete class evidence / split sample minimums"),
        "safeguards": "Image decode, SHA-256, dHash, original source groups; not proof of diagnosis",
    }
    return report, checked


def commit_staging(entries: list[dict], report: dict, workspace: Path) -> Path:
    if not report["ready_for_full_50"]:
        raise ValueError("Refusing to import an incomplete, unapproved 50-class dataset")
    target_parent = workspace / "approved_staging"
    target_parent.mkdir(parents=True, exist_ok=True)
    target = target_parent / ("candidate_" + hashlib.sha256(
        json.dumps(entries, sort_keys=True).encode()).hexdigest()[:12])
    if target.exists():
        raise FileExistsError("The identical staging candidate already exists; no overwrite")
    temp = Path(tempfile.mkdtemp(prefix=".staging-", dir=target_parent))
    try:
        groups = []
        provenance = []
        archive_cache: dict[str, ZipFile] = {}
        try:
            for entry in entries:
                archive = entry["archive"]
                if archive not in archive_cache:
                    archive_cache[archive] = ZipFile(archive)
                data = archive_cache[archive].read(entry["member"])
                if hashlib.sha256(data).hexdigest() != entry["original_sha256"]:
                    raise ValueError("ZIP image changed after preflight")
                suffix = Path(entry["member"]).suffix.lower()
                dest = (temp / entry["split"] / entry["class_id"] /
                        (entry["original_sha256"] + suffix))
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    raise ValueError("Duplicate output asset detected")
                dest.write_bytes(data)
                groups.append({
                    "relative_path": dest.relative_to(temp).as_posix(),
                    "group_id": entry["source_group_id"],
                })
                provenance.append({
                    "relative_path": dest.relative_to(temp).as_posix(),
                    **entry,
                })
        finally:
            for f in archive_cache.values():
                f.close()
        with (temp / "groups.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path", "group_id"])
            writer.writeheader()
            writer.writerows(groups)
        (temp / "provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (temp / "import_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        temp.replace(target)
        return target
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review-csv", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--workspace", default="/content/drive/MyDrive/SmartPlant50")
    p.add_argument("--legacy-dir", default="/content/drive/MyDrive/SmartPlant_Datasets")
    p.add_argument("--commit", action="store_true", help="Write fully verified data to new staging only")
    a = p.parse_args()
    workspace = Path(a.workspace)
    report, entries = audit_and_plan(
        Path(a.review_csv), Path(a.plan), workspace, Path(a.legacy_dir)
    )
    report_path = workspace / "audit_reports" / "verified_import_preflight.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2)[:10000])
    if a.commit:
        print("New, untouched staging folder:", commit_staging(entries, report, workspace))
    else:
        print("DRY RUN: no imports written. --commit only stages when all 50 classes pass.")


if __name__ == "__main__":
    main()
