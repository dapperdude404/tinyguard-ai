"""Inspect a single public-file download handshake (headers only; no file bytes saved).

This network smoke test uses public, checksum-pinned Mendeley metadata.
It closes the HTTP response without reading its body and never creates images.
"""
from pathlib import Path
from urllib.parse import urlparse
import requests

from scripts.acquire_smartplant50_sources import (
    discover, get_public, approved_https_url, source_datasets,
)

register=Path("dataset_sources/50class_source_register.csv")
session=requests.Session()
datasets=discover(session,register)
verified=0
for dataset in datasets:
    if not dataset.get("cc_by_4_metadata_verified"):
        continue
    for file in dataset["files"]:
        url=file.get("download_url") or ""
        if not (file.get("eligible_file_type") and file.get("within_size_limit")
                and file.get("provider_sha256") and approved_https_url(url)):
            continue
        parsed=urlparse(url)
        if (parsed.hostname!="data.mendeley.com" or
                not parsed.path.startswith("/public-files/datasets/"
                                           +dataset["dataset_id"]+"/")):
            continue
        with get_public(session,url,stream=True) as response:
            print("Public download header handshake: HTTP",response.status_code,
                  "final host:",urlparse(response.url).hostname,
                  "content_type:",response.headers.get("Content-Type","unknown"),
                  "declared_bytes:",file["size"])
        verified=1
        break
    if verified:
        break
if not verified:
    raise SystemExit("No licensed SHA-pinned public file to test")
print("PASS: Public-file HTTPS stream could open without credentials; "
      "no file content was downloaded or stored")
