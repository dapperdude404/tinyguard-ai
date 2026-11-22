"""Smoke checks for read-only structural dataset inventory; fake files never train a CNN."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.inventory_smartplant50 import inventory_dataset, MIN_COUNTS


def build_toy_structure(root):
    """Only zero-byte filename placeholders, not valid training photos."""
    rows = []
    for split in ("train", "val", "test"):
        for n in range(50):
            subdir = root / split / f"c{n:03d}"
            subdir.mkdir(parents=True, exist_ok=True)
            filename = subdir / f"{split}_{n}.jpg"
            filename.touch()
            rows.append((filename.relative_to(root).as_posix(), f"plant-{split}-{n}"))
    with (root / "groups.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["relative_path", "group_id"])
        writer.writerows(rows)


class StructuralInventoryTests(unittest.TestCase):
    def test_missing_folder_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            report = inventory_dataset(Path(d) / "nonexistent")
            self.assertFalse(report["ready_for_strict_audit"])
            self.assertFalse(report["present"])

    def test_structural_completion_requires_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            build_toy_structure(root)
            with patch.dict(MIN_COUNTS, {"train": 1, "val": 1, "test": 1}):
                report = inventory_dataset(root)
            self.assertTrue(report["ready_for_strict_audit"])
            self.assertEqual(report["total_images"], 150)
            self.assertIn("Structural inventory only", report["note"])

    def test_cross_split_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            build_toy_structure(root)
            groups = root / "groups.csv"
            content = groups.read_text()
            groups.write_text(content.replace("plant-val-0", "plant-train-0"))
            with patch.dict(MIN_COUNTS, {"train": 1, "val": 1, "test": 1}):
                report = inventory_dataset(root)
            self.assertFalse(report["ready_for_strict_audit"])
            self.assertTrue(any("spans multiple splits" in gap for gap in report["gaps"]))

    def test_unreviewed_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            build_toy_structure(root)
            (root / "train" / "c001" / "unknown.bmp").touch()
            with patch.dict(MIN_COUNTS, {"train": 1, "val": 1, "test": 1}):
                report = inventory_dataset(root)
            self.assertFalse(report["ready_for_strict_audit"])
            self.assertTrue(any("Unreviewed" in gap for gap in report["gaps"]))


if __name__ == "__main__":
    unittest.main()
