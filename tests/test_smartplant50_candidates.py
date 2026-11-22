"""Offline tests for no-guess class mapping and source ZIP review candidates."""
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.prepare_smartplant50_candidates import (
    proposal_for_folder, generate_review,
)


class CandidateReviewTests(unittest.TestCase):
    def test_exact_names_remain_unapproved_suggestions(self):
        labels = ["Potato | Early Blight", "Corn (Maize) | Healthy"]
        self.assertEqual(proposal_for_folder("Potato___Early_Blight", labels),
                         ("c000", "Potato | Early Blight", "EXACT_NORMALIZED_SUGGESTION_ONLY"))
        self.assertEqual(proposal_for_folder("Corn_(Maize)___Healthy", labels),
                         ("c001", "Corn (Maize) | Healthy", "EXACT_NORMALIZED_SUGGESTION_ONLY"))

    def test_unmatched_disease_is_never_guessed(self):
        names = ["Guava | Leaf Spot", "Banana | Yellow Sigatoka"]
        self.assertEqual(proposal_for_folder("Guava___Dot", names),
                         ("", "", "UNRESOLVED_REVIEW_REQUIRED"))
        self.assertEqual(proposal_for_folder("Banana___Sigatoka", names),
                         ("", "", "UNRESOLVED_REVIEW_REQUIRED"))

    def test_existing_review_file_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old_dir = root / "SmartPlant_Datasets"
            old_dir.mkdir()
            with ZipFile(old_dir / "Potato.zip", "w") as z:
                z.writestr("Potato___Early_Blight/a.jpg", b"dummy-file-for-candidate-inventory")
            plan = root / "plan.json"
            plan.write_text(json.dumps({"classes": [
                {"id": f"c{i:03d}", "index": i,
                 "name": "Potato | Early Blight" if i == 0 else f"Crop {i} | Class {i}"}
                for i in range(50)
            ]}))
            report = root / "SmartPlant50/audit_reports/review.csv"
            outcome = generate_review(old_dir, root / "SmartPlant50", plan, report)
            self.assertEqual(outcome["exact_suggestions"], 1)
            self.assertEqual(outcome["approved"], 0)
            self.assertIn("reviewed_class_id", report.read_text())
            with self.assertRaises(FileExistsError):
                generate_review(old_dir, root / "SmartPlant50", plan, report)


if __name__ == "__main__":
    unittest.main()
