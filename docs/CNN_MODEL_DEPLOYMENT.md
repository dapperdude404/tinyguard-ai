# SmartPlant AI — model deployment review

Status: **Draft PR #2; not merged or deployed.**

The user-supplied, trusted checkpoint is stored on Google Drive under file ID `1oF2W3-CelHeIxYWJLOL8m1yFRviIpWVZ`. Drive permissions were checked and include `anyone:reader`. The app's optional downloader uses this ID by default; no Streamlit secret is required.

Checkpoint structure and inference must be verified in a clean runtime before release. Prior reported validation results are not independently reproduced in this review.

The downloader validates checkpoint structure and class count, but does not yet verify file provenance with a confirmed checksum. Obtain a SHA-256 from the actual downloaded binary before production release. If Drive serves an HTML error page, a changed file, or a quota message, the app must fall back to its existing descriptive color analysis.

## Before merge

1. Reconcile the divergence between `main` and `feature/smartplant-model-deployment`. Do not overwrite newer changes on `main`.
2. Run `python -m pytest -q` and inspect GitHub Actions checks.
3. Test the public Drive download in a fresh environment using `python -c "from smartplant.model_download import ensure_cnn_checkpoint; print(ensure_cnn_checkpoint('models/smartplant_mobilenetv3.pt'))"`.
4. Run `python scripts/check_cnn_checkpoint.py`.
5. Run `streamlit run app.py` and test a real leaf photo.
6. Review possible cross-split duplicate matches in the dataset audit. The previously reported 99.63% test accuracy is **not** independently verified field accuracy.

## Release

Only after the checks pass, approve and merge PR #2. Confirm Streamlit redeploys from `main` and displays CNN predictions. Keep a rollback path.

**The current main branch contains a tiny invalid placeholder at models/smartplant_mobilenetv3.pt, not the trained binary.** The downloader attempts to replace invalid placeholders after validation. The Drive link is public to anyone with the link; do not use this method for confidential model artifacts.
