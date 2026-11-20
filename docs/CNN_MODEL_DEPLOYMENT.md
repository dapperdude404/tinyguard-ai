# SmartPlant AI trained checkpoint deployment

This branch prepares the application to include the user's **trusted, self-trained** MobileNetV3-Small checkpoint. The checkpoint binary is **not yet committed**; the pull request must stay in draft until it is uploaded and tested.

## Add the checkpoint using GitHub's web interface

1. Download `best_model.pt` from Google Drive's `SmartPlant_Results` folder.
2. In GitHub, switch to branch `feature/smartplant-model-deployment`.
3. Open **Add file → Upload files** at the repository root.
4. To preserve the required path, create the folder `models` first (e.g. create `models/.gitkeep`), then open `models` and choose **Add file → Upload files**.
5. Rename the local file to `smartplant_mobilenetv3.pt` before uploading it.
6. Commit **to this feature branch**, never directly to `main`.
7. Verify the file appears at `models/smartplant_mobilenetv3.pt` on the branch.

GitHub's web uploader may have file size and browser limitations; the checkpoint is about 6.1 MB. Do not upload access tokens or private Drive links.

## Validation before merge

- Run `python -m pytest -q`.
- Run `python scripts/check_cnn_checkpoint.py` (requires PyTorch and torchvision).
- Run `streamlit run app.py`, upload a leaf photograph, and confirm the top-five CNN prediction chart appears.
- Check label order and preprocessing against the training script.
- Investigate dataset audit warnings for possible cross-split image matches. Reported 99.63% test accuracy is **not** independently verified field accuracy.
- Review the GitHub Actions results and approve the PR only after model verification.

## After merge

Streamlit Community Cloud should redeploy from `main` if automatic deployment is configured. Reboot the app if necessary, then smoke-test an uploaded leaf. The model is CPU-loadable and may take longer to start. If deployment fails, revert the PR or remove the model file from the deployment branch.

**No new model training or new accuracy measurement is claimed by this deployment process.**

## Alternative: Google Drive model download (prepared, not yet enabled)

The owner supplied Drive file ID `1oF2W3-CelHeIxYWJLOL8m1yFRviIpWVZ`, titled `best_model.pt`. Metadata checked on 2026-11-20 showed **private** access (owner only). Do not enable the downloader until the owner intentionally changes file sharing to **Anyone with the link — Viewer** and confirms access from a signed-out browser.

In Streamlit Community Cloud app settings → Secrets, add:

```toml
SMARTPLANT_MODEL_DRIVE_ID = "1oF2W3-CelHeIxYWJLOL8m1yFRviIpWVZ"
```

The app will download the file at startup when a leaf is uploaded, validate basic checkpoint structure, and store it at `models/smartplant_mobilenetv3.pt`. This does not publish the model binary in GitHub. The download may fail due to Drive quotas or permissions. Validate with `scripts/check_cnn_checkpoint.py` and independent leaf photos before merging. A stable artifact host with a pinned checksum is preferable for production.
