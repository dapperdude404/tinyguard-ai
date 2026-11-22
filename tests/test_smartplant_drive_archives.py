"""Unit tests for *metadata-only* Drive ZIP intake and safe optional staging."""
import csv
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_STORED
import stat

from scripts.scan_smartplant_drive_archives import (
    safe_member, inspect_archive, scan_drive_archives
)


class DriveArchiveTests(unittest.TestCase):
    def test_one_archive_scanned_without_extraction_or_label_claims(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = root / "SmartPlant_Datasets"
            raw.mkdir()
            with ZipFile(raw / "Potato.zip", "w", ZIP_STORED) as z:
                z.writestr("train/Potato___Early_blight/photo1.jpg", b"fake-photo-test")
                z.writestr("train/Potato___Early_blight/photo2.jpg", b"fake-photo-test2")
                z.writestr("notes.md", b"not an image")
            report = scan_drive_archives(raw, root / "reports")
            self.assertEqual(report["found_archives"], 1)
            self.assertEqual(report["image_count"], 2)
            self.assertFalse(report["images_staged"])
            self.assertTrue(report["missing_archives"])
            with (root / "reports/legacy_zip_class_candidates.csv").open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertIn("UNVERIFIED", rows[0]["mapping_status"])
            self.assertEqual(rows[0]["images"], "2")
            self.assertFalse((root / "data" / "train").exists())

    def test_staged_data_do_not_become_train_test_splits(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = root / "Potato.zip"
            with ZipFile(raw, "w") as z:
                z.writestr("Potato___Early_blight/img.jpg", b"placeholder-only")
            report = inspect_archive(raw, stage_dir=root / "staging")
            self.assertEqual(report["image_count"], 1)
            staged = root / "staging" / "Potato" / "Potato___Early_blight" / "img.jpg"
            self.assertTrue(staged.exists())
            self.assertFalse((root / "train").exists())

    def test_reject_path_traversal(self):
        with self.assertRaises(ValueError):
            safe_member("../steal.jpg")
        with self.assertRaises(ValueError):
            safe_member("/tmp/steal.jpg")
        with self.assertRaises(ValueError):
            safe_member("C:/steal.jpg")
        with self.assertRaises(ValueError):
            safe_member(r"folder\\evil.jpg")

    def test_zip_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / "Apple.zip"
            with ZipFile(archive, "w") as z:
                entry = ZipInfo("Apple___Healthy/file.jpg")
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(entry, b"../../secrets")
            with self.assertRaisesRegex(ValueError, "symlink"):
                inspect_archive(archive)

    def test_duplicate_member_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / "Peach.zip"
            with ZipFile(archive, "w") as z:
                z.writestr("folder/image.jpg", b"a")
                z.writestr("folder/IMAGE.JPG", b"b")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                inspect_archive(archive)


if __name__ == "__main__":
    unittest.main()
