"""Offline tests for publisher API metadata, authorization and checksummed quarantine."""
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.acquire_smartplant50_sources as mod


class Response:
    def __init__(self, chunks):
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, chunk_size=1024 * 1024):
        return iter(self.chunks)


class SourceAcquisitionTests(unittest.TestCase):
    def test_google_and_untrusted_urls_not_automatically_followed(self):
        self.assertTrue(mod.approved_https_url("https://api.data.mendeley.com/datasets/foo"))
        self.assertTrue(mod.approved_https_url("https://s3.amazonaws.com/verylonghash"))
        self.assertFalse(mod.approved_https_url("http://api.data.mendeley.com"))
        self.assertFalse(mod.approved_https_url("https://127.0.0.1/steal"))
        self.assertFalse(mod.approved_https_url("https://evil.org/dataset"))
        self.assertFalse(mod.approved_https_url("https://api.data.mendeley.com.evil.org/"))

    def test_license_is_fail_closed(self):
        self.assertTrue(mod.find_cc_by_4({"data_licence": {"name": "CC BY 4.0"}}))
        self.assertTrue(mod.find_cc_by_4({"data_licence": {"url": "https://creativecommons.org/licenses/by/4.0/"}}))
        self.assertFalse(mod.find_cc_by_4({"data_licence": {"name": "CC BY-NC 4.0"}}))
        self.assertFalse(mod.find_cc_by_4({}))
        self.assertFalse(mod.find_cc_by_4({"data_licence": {"name": "All rights reserved"}}))

    def test_filename_and_checksum_are_checked(self):
        with self.assertRaises(ValueError):
            mod.normalise_file({"id": "fa0b1842-a45b-46ad-98de-e949b4c47ce3",
                                "filename": "../private.zip", "content_details": {"size": 4}})
        with self.assertRaises(ValueError):
            mod.normalise_file({"id": "fa0b1842-a45b-46ad-98de-e949b4c47ce3",
                                "filename": "images.zip",
                                "content_details": {"size": 4, "sha256_hash": "banana"}})

    def test_discovery_deduplicates_datasets(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = Path(d) / "register.csv"
            with csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["class_id", "source_url"])
                writer.writeheader()
                writer.writerow({"class_id": "c029", "source_url":
                    "https://data.mendeley.com/datasets/7ghdbftp54/1"})
                writer.writerow({"class_id": "c030", "source_url":
                    "https://data.mendeley.com/datasets/7ghdbftp54/1"})
                writer.writerow({"class_id": "c043", "source_url":
                    "https://zenodo.org/records/13137587"})
            self.assertEqual(mod.source_datasets(csv_path),
                             {("7ghdbftp54", 1): ["c029", "c030"]})

    def test_sha_verified_download_then_reuse(self):
        data = b"trusted-image-archive-placeholder"
        digest = hashlib.sha256(data).hexdigest()
        entry = {"filename": "data.zip", "file_id": "fa0b1842-a45b-46ad-98de-e949b4c47ce3",
                 "size": len(data), "provider_sha256": digest,
                 "eligible_file_type": True}
        dataset = {"dataset_id": "7ghdbftp54", "version": 1,
                   "cc_by_4_metadata_verified": True}
        with tempfile.TemporaryDirectory() as d:
            with patch.object(mod, "get_public", return_value=Response([data])):
                result = mod.download_verified(None, entry, dataset, Path(d),
                                               remaining_bytes=1000)
            self.assertEqual(result["status"], "VERIFIED_QUARANTINE")
            self.assertFalse(result["cached"])
            cached = mod.download_verified(None, entry, dataset, Path(d),
                                           remaining_bytes=1000)
            self.assertTrue(cached["cached"])

    def test_reject_bad_content_checksum_and_leave_no_file(self):
        data = b"one"
        entry = {"filename": "data.zip", "file_id": "fa0b1842-a45b-46ad-98de-e949b4c47ce3",
                 "size": len(data), "provider_sha256": "0" * 64,
                 "eligible_file_type": True}
        dataset = {"dataset_id": "7ghdbftp54", "version": 1,
                   "cc_by_4_metadata_verified": True}
        with tempfile.TemporaryDirectory() as d:
            with patch.object(mod, "get_public", return_value=Response([data])):
                with self.assertRaises(ValueError):
                    mod.download_verified(None, entry, dataset, Path(d),
                                          remaining_bytes=1000)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_not_licensed_and_not_hashed_stay_blocked(self):
        entry = {"filename": "data.zip", "file_id": "fa0b1842-a45b-46ad-98de-e949b4c47ce3",
                 "size": 4, "provider_sha256": "", "eligible_file_type": True}
        dataset = {"dataset_id": "7ghdbftp54", "version": 1,
                   "cc_by_4_metadata_verified": False}
        with tempfile.TemporaryDirectory() as d:
            result = mod.download_verified(None, entry, dataset, Path(d), remaining_bytes=100)
            self.assertEqual(result["status"], "SKIPPED")
            self.assertEqual(list(Path(d).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
