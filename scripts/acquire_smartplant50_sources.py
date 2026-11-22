"""Acquire public, versioned Mendeley Data archives into Drive quarantine.

Automatically discovers published files referenced by the 21-class source
register. Explicit --download requires a CC-BY-4.0 licence evidenced by API
metadata AND a published SHA-256 for each byte stream. Never puts photos in
train/val/test or marks a disease annotation as verified.

Do not use this for data redistribution, diagnosis, or training before label,
licence, original-image and group-level provenance review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import os
import re
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

API = "https://api.data.mendeley.com"
SOURCE_RE = re.compile(r"^https://data\.mendeley\.com/datasets/([a-z0-9]{6,24})/([1-9][0-9]*)/?$")
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
UUID_RE = re.compile(r"^[0-9a-fA-F-]{32,36}$")
HOST_ALLOW = (
    "mendeley.com", "elsevier.com", "amazonaws.com",
    "digitalcommonsdata.com", "digitalcommons.elsevier.com",
)
FILE_TYPES = {".zip", ".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_BYTES = 2 * 1024**3
DEFAULT_MAX_BYTES = 4 * 1024**3


def approved_https_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().strip(".")
        return (parsed.scheme == "https" and not parsed.username
                and not parsed.password and parsed.port in (None, 443)
                and host and any(host == h or host.endswith("." + h) for h in HOST_ALLOW)
                and not host.replace(".", "").isdigit())
    except (ValueError, TypeError):
        return False


def assert_public_host(url: str) -> None:
    if not approved_https_url(url):
        raise ValueError("Download URL is not an approved HTTPS data provider host")
    host = urlparse(url).hostname
    for result in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP):
        address = ipaddress.ip_address(result[4][0])
        if not address.is_global:
            raise ValueError("Provider address resolves to a non-public IP")


def get_public(session, url: str, *, params=None, stream=False):
    """Follow only approved public HTTPS redirects; fail closed on private hosts."""
    for _ in range(6):
        assert_public_host(url)
        response = session.get(url, params=params, stream=stream,
                               allow_redirects=False, timeout=(15, 70))
        if response.status_code in (301, 302, 303, 307, 308):
            from urllib.parse import urljoin
            location = response.headers.get("Location", "")
            response.close()
            if not location:
                raise ValueError("Provider redirect missing location")
            url = urljoin(response.url, location)
            params = None
            continue
        response.raise_for_status()
        return response
    raise ValueError("Too many redirects")


def get_json(session, url, params=None):
    with get_public(session, url, params=params) as response:
        return response.json()


def source_datasets(register: Path) -> dict[tuple[str, int], list[str]]:
    sources: dict[tuple[str, int], list[str]] = {}
    with register.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            match = SOURCE_RE.fullmatch(row["source_url"].strip())
            if match:
                key = (match.group(1), int(match.group(2)))
                sources.setdefault(key, []).append(row["class_id"])
    return sources


def find_cc_by_4(metadata: dict) -> bool:
    """Only a narrow positive licence proof; human licensing review still required."""
    license_data = metadata.get("data_licence") or metadata.get("data_license") or {}
    if isinstance(license_data, str):
        text = license_data
    elif isinstance(license_data, dict):
        text = " ".join(str(license_data.get(k, "")) for k in
                        ("name", "short_name", "description", "url", "license_url",
                         "licence_url", "spdx_id"))
    else:
        return False
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    return "creativecommons.org/licenses/by/4.0" in text.lower() or (
        "ccby40" in compact and "ccby-nc" not in text.lower()
    )


def list_public_files(session, dataset_id: str, version: int) -> list[dict]:
    all_files: list[dict] = []
    start, page_size = 0, 100
    for _ in range(60):
        records = get_json(session, f"{API}/datasets/publics/{dataset_id}/files",
                           params={"version": version, "$start": start,
                                   "$limit": page_size})
        if isinstance(records, list):
            page = records
        elif isinstance(records, dict):
            page = records.get("results") or records.get("files") or records.get("items")
            if page is None:
                raise ValueError("Unsupported Mendeley files listing schema")
        else:
            raise ValueError("Unsupported Mendeley files listing")
        if not isinstance(page, list):
            raise ValueError("Expected file list")
        all_files.extend(page)
        if len(page) < page_size:
            return all_files
        start += len(page)
    raise ValueError("Provider exceeded allowed file pages")


def normalise_file(item: dict) -> dict:
    detail = item.get("content_details") or {}
    name = item.get("filename") or item.get("name") or ""
    if (not isinstance(name, str) or Path(name).name != name or
            name in (".", "..") or "\x00" in name):
        raise ValueError("Unsafe provider filename")
    file_id = str(item.get("id", ""))
    if not UUID_RE.fullmatch(file_id):
        raise ValueError("Provider file is missing a valid UUID")
    size = detail.get("size", item.get("size"))
    if not isinstance(size, int) or not 0 <= size <= MAX_FILE_BYTES:
        raise ValueError("Provider file size missing or exceeds limit")
    digest = detail.get("sha256_hash") or item.get("sha256_hash") or ""
    if not isinstance(digest, str):
        digest = ""
    digest = digest.lower()
    if digest and not HEX64.fullmatch(digest):
        raise ValueError("Invalid provider SHA-256")
    return {"filename": name, "file_id": file_id, "size": size,
            "provider_sha256": digest,
            "eligible_file_type": Path(name).suffix.lower() in FILE_TYPES}


def discover(session, register: Path) -> list[dict]:
    datasets = []
    for (dataset_id, version), classes in sorted(source_datasets(register).items()):
        metadata = get_json(session, f"{API}/datasets/{dataset_id}",
                            params={"version": version})
        if not isinstance(metadata, dict):
            raise ValueError("Unexpected dataset metadata")
        actual_id = metadata.get("id")
        actual_version = metadata.get("version")
        if actual_id != dataset_id or int(actual_version) != version:
            raise ValueError("Dataset ID/version mismatch in public API")
        permitted = find_cc_by_4(metadata)
        source_files = list_public_files(session, dataset_id, version)
        files = [normalise_file(item) for item in source_files]
        datasets.append({
            "dataset_id": dataset_id, "version": version, "name": metadata.get("name"),
            "license": metadata.get("data_licence") or metadata.get("data_license"),
            "cc_by_4_metadata_verified": permitted,
            "candidate_class_ids": sorted(classes),
            "files": files,
            "state": "DISCOVERED ONLY: requires label, licence, provenance audit",
        })
    return datasets


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".tmp-", delete=False) as f:
        tmp = Path(f.name)
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def download_verified(session, entry: dict, dataset: dict, folder: Path,
                      *, remaining_bytes: int) -> dict:
    if not dataset["cc_by_4_metadata_verified"]:
        return {"status": "SKIPPED", "reason": "License not positively verified"}
    if not entry["eligible_file_type"]:
        return {"status": "SKIPPED", "reason": "Unsupported file extension"}
    if not entry["provider_sha256"]:
        return {"status": "SKIPPED", "reason": "Missing upstream SHA-256"}
    if entry["size"] > remaining_bytes:
        return {"status": "SKIPPED", "reason": "Size budget exhausted"}
    file_name = entry["file_id"] + Path(entry["filename"]).suffix.lower()
    destination = folder / file_name
    if destination.exists():
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == entry["provider_sha256"] and destination.stat().st_size == entry["size"]:
            return {"status": "VERIFIED_QUARANTINE", "path": str(destination),
                    "sha256": digest, "size": entry["size"], "cached": True}
        raise ValueError("Existing quarantine bytes fail provider SHA-256")

    folder.mkdir(parents=True, exist_ok=True)
    download_url = (f"{API}/datasets/{dataset['dataset_id']}/files/"
                    f"{entry['file_id']}/file_downloaded?version={dataset['version']}")
    total = 0
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile("wb", dir=folder, prefix=".downloading-",
                                     delete=False) as f:
        temp = Path(f.name)
    try:
        with get_public(session, download_url, stream=True) as response:
            with temp.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > entry["size"] or total > remaining_bytes:
                        raise ValueError("Downloaded data exceeds expected size or budget")
                    digest.update(chunk)
                    output.write(chunk)
        if total != entry["size"] or digest.hexdigest() != entry["provider_sha256"]:
            raise ValueError("Provider SHA-256 or content size mismatch")
        temp.replace(destination)
        return {"status": "VERIFIED_QUARANTINE", "path": str(destination),
                "sha256": digest.hexdigest(), "size": total, "cached": False}
    finally:
        temp.unlink(missing_ok=True)


def run_acquisition(register: Path, workspace: Path, *, do_download: bool = False,
                    max_total_bytes: int = DEFAULT_MAX_BYTES, session=None):
    if max_total_bytes <= 0:
        raise ValueError("Download byte budget must be positive")
    sess = session or requests.Session()
    datasets = discover(sess, register)
    report = {"schema": "smartplant-source-acquisition-v1",
              "download_requested": do_download, "source_datasets": datasets,
              "verified_download_count": 0, "downloaded_bytes": 0,
              "warnings": [
                  "Quarantined source files are not approved training data",
                  "File SHA verifies bytes; licence and disease taxonomy require separate review",
                  "No independently grouped train/validation/test splits have been created",
              ]}
    for dataset in datasets:
        for entry in dataset["files"]:
            if not do_download:
                entry["acquisition"] = {"status": "DISCOVERED_ONLY"}
                continue
            remaining = max_total_bytes - report["downloaded_bytes"]
            entry["acquisition"] = download_verified(
                sess, entry, dataset,
                workspace / "source_quarantine" /
                (dataset["dataset_id"] + "_v" + str(dataset["version"])),
                remaining_bytes=remaining)
            if entry["acquisition"]["status"] == "VERIFIED_QUARANTINE":
                report["verified_download_count"] += 1
                if not entry["acquisition"].get("cached"):
                    report["downloaded_bytes"] += entry["size"]
    atomic_json(workspace / "audit_reports" / "external_source_receipts.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", default="dataset_sources/50class_source_register.csv")
    p.add_argument("--workspace", default="/content/drive/MyDrive/SmartPlant50")
    p.add_argument("--download", action="store_true",
                   help="Actually download SHA-pinned CC BY 4.0 files to quarantine")
    p.add_argument("--max-total-gib", type=float, default=4.0)
    args = p.parse_args()
    report = run_acquisition(Path(args.registry), Path(args.workspace),
                             do_download=args.download,
                             max_total_bytes=int(args.max_total_gib * 1024**3))
    print(f"Discovered {len(report['source_datasets'])} public dataset sources")
    print(f"Verified quarantine files: {report['verified_download_count']}")
    print(f"Downloaded bytes: {report['downloaded_bytes']:,}")
    print("Review the receipts in SmartPlant50/audit_reports before importing.")
    print("No images have been assigned to training classes or model weights.")


if __name__ == "__main__":
    main()
