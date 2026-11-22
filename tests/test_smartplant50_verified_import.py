"""Offline tests for verified import, source-plant grouping and safe partial staging."""
import csv
import hashlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from scripts.prepare_smartplant50_candidates import FIELDS, inspect_zip
from scripts.verified_import_smartplant50 import (
    audit_and_plan, commit_staging, split_for_group,
)


def png(seed: int) -> bytes:
    image = Image.new("RGB", (96, 96))
    rng = random.Random(seed)
    points = [tuple(rng.randrange(256) for _ in range(3)) for _ in range(96 * 96)]
    image.putdata(points)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def fixture(root: Path):
    workspace = root / "SmartPlant50"
    quarantine = workspace / "source_quarantine" / "demo"
    quarantine.mkdir(parents=True)
    zip_path = quarantine / "approved.zip"
    members = {
        "Citrus___Healthy/leaf1.png": png(1),
        "Citrus___Healthy/leaf2.png": png(2),
        "Citrus___Healthy/leaf3.png": png(3),
    }
    with ZipFile(zip_path, "w") as z:
        for k, v in members.items():
            z.writestr(k, v)
    class_plan = workspace / "data/expanded/class_plan.json"
    class_plan.parent.mkdir(parents=True)
    class_plan.write_text(json.dumps({"classes": [
        {"id": f"c{i:03d}", "index": i, "name": "Citrus | Healthy" if i == 44
         else f"Crop{i} | Disease{i}"} for i in range(50)
    ]}))
    review_path = workspace / "audit_reports/review.csv"
    review_path.parent.mkdir(parents=True)
    rows = inspect_zip(zip_path, json.loads(class_plan.read_text())["classes"] and [
        "Citrus | Healthy" if i == 44 else f"Crop{i} | Disease{i}" for i in range(50)
    ])
    assert len(rows) == 1
    rows[0]["reviewed_class_id"] = "c044"
    rows[0]["license_checked"] = "yes"
    rows[0]["taxonomy_checked"] = "yes"
    groups = workspace / "audit_reports/source_groups.csv"
    with groups.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["member_path", "source_group_id"])
        for k, v in members.items():
            writer.writerow([k, k.split("/")[-1]])
    rows[0]["source_group_csv"] = str(groups)
    with review_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return workspace, zip_path, class_plan, review_path, rows


class VerifiedImportTests(unittest.TestCase):
    def test_incomplete_but_approved_data_may_stage_and_not_train(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            workspace, zip_path, plan, review, _ = fixture(root)
            report, checked = audit_and_plan(review, plan, workspace, root / "SmartPlant_Datasets")
            self.assertEqual(len(checked), 3)
            self.assertGreater(len(report["class_gaps"]), 0)
            self.assertFalse(report["ready_for_full_50"])
            staged = commit_staging(checked, report, workspace)
            self.assertTrue(staged.is_dir())
            self.assertTrue((staged / "groups.csv").exists())
            self.assertTrue((staged / "provenance.json").exists())
            self.assertFalse((workspace / "data/expanded/train/c044").exists())
            with self.assertRaises(FileExistsError):
                commit_staging(checked, report, workspace)

    def test_licence_unapproved_is_not_imported(self):
        with tempfile.TemporaryDirectory() as d:
            workspace, _, plan, review, rows = fixture(Path(d))
            rows[0]["license_checked"] = "no"
            with review.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            report, checked = audit_and_plan(review, plan, workspace, Path(d) / "SmartPlant_Datasets")
            self.assertEqual(checked, [])
            with self.assertRaisesRegex(ValueError, "No approved"):
                commit_staging(checked, report, workspace)

    def test_archive_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            workspace, _, plan, review, rows = fixture(Path(d))
            rows[0]["archive_sha256"] = "0" * 64
            with review.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "checksum"):
                audit_and_plan(review, plan, workspace, Path(d) / "SmartPlant_Datasets")

    def test_missing_source_group_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            workspace, _, plan, review, rows = fixture(Path(d))
            groups = Path(rows[0]["source_group_csv"])
            groups.write_text("member_path,source_group_id\nCitrus___Healthy/leaf1.png,plant-A\n")
            with self.assertRaisesRegex(ValueError, "cover exactly"):
                audit_and_plan(review, plan, workspace, Path(d) / "SmartPlant_Datasets")

    def test_hash_based_split_is_stable(self):
        self.assertEqual(split_for_group("farm1-plant1"), split_for_group("farm1-plant1"))
        self.assertIn(split_for_group("farm1-plant1"), {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
