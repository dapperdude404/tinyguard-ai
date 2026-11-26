"""Diagnose public metadata endpoints without downloading dataset images.

Prints only status, content type, top-level JSON keys and file count,
never signed download links or user credentials. Research diagnostic.
"""
from __future__ import annotations
import json
import requests

DATASET="7ghdbftp54"
TARGETS=[
    f"https://api.data.mendeley.com/datasets/{DATASET}?version=1",
    f"https://api.data.mendeley.com/datasets/publics/{DATASET}?version=1",
    f"https://data.mendeley.com/public-api/datasets/{DATASET}",
    f"https://data.mendeley.com/public-api/datasets/{DATASET}?version=1",
    "https://data.mendeley.com/public-api/datasets/5nfjzntwd8?version=3",
    "https://data.mendeley.com/public-api/datasets/5nfjzntwd8?version=2",
    f"https://data.mendeley.com/public-api/datasets/{DATASET}/1",
    f"https://data.mendeley.com/datasets/{DATASET}/1",
]

for url in TARGETS:
    try:
        response=requests.get(url,timeout=12,allow_redirects=False,
                              headers={"Accept":"application/json",
                                       "User-Agent":"SmartPlantAI-ResearchDatasetMetadata/1.0"})
        result={"url":url,"status":response.status_code,
                "content_type":response.headers.get("Content-Type","")[:80]}
        if response.status_code==200 and "json" in response.headers.get("Content-Type",""):
            try:
                info=response.json()
                result["type"]=type(info).__name__
                if isinstance(info,dict):
                    result["top_level_keys"]=list(info)[:25]
                    result["dataset_version"]=info.get("version")
                    result["dataset_id_type"]="uuid" if len(str(info.get("id","")))>=32 else "short"
                    licence=info.get("data_licence")
                    result["licence"]=licence if isinstance(licence,(str,dict)) else str(type(licence))
                    result["files_count"]=len(info.get("files",[])) if isinstance(info.get("files"),list) else None
                    if isinstance(info.get("files"),list) and info["files"] and isinstance(info["files"][0],dict):
                        file=info["files"][0]
                        result["first_file_keys"]=list(file.keys())
                        detail=file.get("content_details") or {}
                        result["content_details_keys"]=list(detail.keys()) if isinstance(detail,dict) else []
                        result["has_provider_sha256"]=bool(detail.get("sha256_hash")) if isinstance(detail,dict) else False
                    if isinstance(info.get("data"),dict):
                        result["nested_data_keys"]=list(info["data"])[:15]
            except ValueError:
                result["json_decode"]="failed"
        print(json.dumps(result))
    except requests.RequestException as exc:
        print(json.dumps({"url":url,"error":type(exc).__name__}))
