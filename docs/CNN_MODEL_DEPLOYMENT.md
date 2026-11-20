# SmartPlant AI — model deployment review

Status: **Draft PR #2; not merged or deployed.**

The user-supplied, trusted checkpoint is stored on Google Drive under file ID `1oF2W3-CelHeIxYWJLOL8m1yFRviIpWVZ`. Drive permissions were checked and include `anyone:reader`. The app's optional downloader uses this ID by default; no Streamlit secret is required.

The checkpoint was verified locally using `torch.load(weights_only=True)`: MobileNetV3-Small, 29 labels, input size 160, all weights matched and synthetic tensor inference returned 29 normalized probabilities. This **does not** measure disease classification accuracy.

Expected SHA-256:
`77a85ecacfa812b3d1ee0d9bdb709b226eef8f910fb8713543dd7e4a892b3ac3`

The downloader rejects unexpected file bytes, architecture, label count or input size. If Drive serves an HTML error page, a changed file, or a quota message, the app must fall back to its existing descriptive color analysis.

## Before merge

1. Reconcile the divergence between `main` and `feature/smartplant-model-deployment`. Do not overwrite newer changes on `main`.
2. Run `python -m pytest -q` and inspect GitHub Actions checks.
3. Test the public Drive download in a fresh environment using `python -c "from smartplant.model_download import ensure_cnn_checkpoint; print(ensure_cnn_checkpoint('models/smartplant_mobilenetv3.pt'))"`.
4. Run `python scripts/check_cnn_checkpoint.py`.
5. Run `streamlit run app.py` and test a real leaf photo.
6. Review possible cross-split duplicate matches in the dataset audit. The previously reported 99.63% test accuracy is **not** independently verified field accuracy.

## Release

Only after the checks pass, approve and merge PR #2. Confirm Streamlit redeploys from `main` and displays CNN predictions. Keep a rollback path.

**No trained model binary is committed to GitHub.** The Drive link is public to anyone with the link; do not use this method for confidential model artifacts.
