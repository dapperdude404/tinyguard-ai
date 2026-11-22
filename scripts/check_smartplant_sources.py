"""Validate the 21-class *source-evidence register*, not the image dataset.

python scripts/check_smartplant_sources.py [--require-ready]
Approved source evidence is NOT proof of independent test accuracy.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

EXPECTED = [
    "Mango | Healthy", "Mango | Anthracnose", "Mango | Powdery Mildew",
    "Guava | Healthy", "Guava | Anthracnose", "Guava | Leaf Spot",
    "Banana | Healthy", "Banana | Black Sigatoka", "Banana | Yellow Sigatoka",
    "Rice | Healthy", "Rice | Brown Spot", "Rice | Leaf Blast",
    "Wheat | Healthy", "Wheat | Leaf Rust", "Wheat | Powdery Mildew",
    "Citrus | Healthy", "Citrus | Canker",
    "Maize | Southern Leaf Blight", "Maize | Southern Rust",
    "Papaya | Healthy", "Papaya | Leaf Curl",
]
COLUMNS = ("class_id", "label", "source_url", "stage", "license_verified",
           "taxonomy_verified", "source_grouped", "original_image_count", "notes")


def inspect(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError("Source CSV header/schema mismatch")
        rows = list(reader)
    if len(rows) != 21:
        raise ValueError("Exactly 21 proposed class rows are required")
    issues = []
    for i, row in enumerate(rows):
        expected_id = f"c{i+29:03d}"
        if row["class_id"] != expected_id or row["label"] != EXPECTED[i]:
            raise ValueError(f"Class plan order mismatch at {expected_id}")
        if row["stage"] not in ("candidate", "needs-review", "approved"):
            raise ValueError("Invalid stage for " + expected_id)
        for flag in ("license_verified", "taxonomy_verified", "source_grouped"):
            if row[flag] not in ("yes", "no"):
                raise ValueError(f"Invalid {flag} flag in {expected_id}")
        try:
            count = int(row["original_image_count"])
        except ValueError:
            raise ValueError("Invalid image count for " + expected_id) from None
        if count < 0:
            raise ValueError("Negative image count for " + expected_id)
        if row["source_url"] and not row["source_url"].startswith("https://"):
            raise ValueError("Source URL must be HTTPS")
        # A candidate never counts as ready merely because flags have been edited.
        # Approval must be explicit; actual image files are checked separately.
        evidence_ready = (row["stage"] == "approved" and
                          row["license_verified"] == "yes" and
                          row["taxonomy_verified"] == "yes" and
                          row["source_grouped"] == "yes" and count >= 30 and
                          bool(row["source_url"]))
        if row["stage"] == "approved" and not evidence_ready:
            raise ValueError("Approved class lacks required evidence: " + expected_id)
        if not evidence_ready:
            issues.append(f"{expected_id} — {row['label']}: source, verified label, "
                          "license, grouping or original image count incomplete")
    return {"total_new_classes": len(rows),
            "stages": dict(Counter(row["stage"] for row in rows)),
            "evidence_ready": len(rows) - len(issues), "gaps": issues}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="dataset_sources/50class_source_register.csv")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    result = inspect(args.registry)
    print(f"New categories: {result['total_new_classes']} | Evidence ready: "
          f"{result['evidence_ready']}/{result['total_new_classes']}")
    print(f"Stages: {result['stages']}")
    for gap in result["gaps"]:
        print("NEEDS REVIEW: " + gap)
    if args.require_ready and result["gaps"]:
        raise SystemExit("Training source-evidence gate NOT passed; do not start training.")


if __name__ == "__main__":
    main()
