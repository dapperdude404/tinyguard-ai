"""Built-in unittest: validates source-list structure and honest readiness gating."""
import csv
import tempfile
import unittest
from pathlib import Path
from scripts.check_smartplant_sources import inspect, COLUMNS

REGISTER = Path(__file__).resolve().parents[1] / "dataset_sources/50class_source_register.csv"


class SourceRegisterTests(unittest.TestCase):
    def test_complete_21_class_plan_but_no_approved_sources(self):
        result = inspect(REGISTER)
        self.assertEqual(result["total_new_classes"], 21)
        self.assertEqual(result["evidence_ready"], 0)
        self.assertEqual(len(result["gaps"]), 21)

    def test_reject_unjustified_approval(self):
        with REGISTER.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        rows[0]["stage"] = "approved"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "lacks required evidence"):
                inspect(path)

    def test_reject_wrong_label_index(self):
        with REGISTER.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        rows[1]["class_id"] = "c042"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "order mismatch"):
                inspect(path)


if __name__ == "__main__":
    unittest.main()
