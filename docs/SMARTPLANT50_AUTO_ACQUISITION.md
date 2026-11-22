# SmartPlant AI — automatic acquisition & verified import

**Development status:** Private Drive workspace prepared, source-ingestion code added to an isolated branch. No new photos, model weights or training metrics exist just because the code was committed. **Do not merge or deploy before review.**

## Works with your existing Google Drive

- [SmartPlant50 workspace](https://drive.google.com/drive/folders/1bCO4AoqpZlMjJ41hyagkBGHBSHTgo-2M): private project folder
- `MyDrive/SmartPlant_Datasets`: existing nine crop ZIP archives; **do not upload again**
- `SmartPlant50/source_quarantine`: downloaded files, SHA-verified but not label-approved
- `SmartPlant50/audit_reports`: provider API receipts and class folder review CSV
- `SmartPlant50/approved_staging`: isolated candidate images with provenance; no production training data replaced
- `SmartPlant50/data/expanded`: existing 50-class development layout, **empty until scientifically reviewed**
- `SmartPlant50/incoming_new_crops`: own crop camera photos, unreviewed

The new Colab cells use an installed Google Drive mount. They discover versioned public Mendeley datasets and download them directly into **your private Drive**. You do not manually upload datasets or ZIPs. Downloads are *not performed by merely visiting a GitHub PR*.

### 1. Google Dataset Search vs. data sources

You can find sources using [Google Dataset Search](https://datasetsearch.research.google.com/), e.g. search for `mango leaf anthracnose dataset`, `banana yellow sigatoka leaf dataset`, `wheat powdery mildew dataset`, `guava leaf spot dataset`, `maize southern rust dataset`.

**Google Dataset Search is a discovery index, not a data store or training-label authority.** The code downloads from trusted publishers' public dataset APIs when a pinned version, explicit permissions, file SHA-256 and size can be checked. It deliberately does *not* scrape random Google Images results: those lack verified reuse rights and disease diagnosis.

Selected public research leads:
- [Mango Leaf Disease Dataset v1](https://data.mendeley.com/datasets/7ghdbftp54/1) (healthy/anthracnose/powdery mildew)
- [Guava fruits and leaves v1](https://data.mendeley.com/datasets/fspx44mwfp/1) (healthy/anthracnose; **Dot is not automatically Leaf Spot**)
- [Banana dataset v3](https://data.mendeley.com/datasets/5nfjzntwd8/3) (newer publisher description includes Yellow Sigatoka, but actual file categories and source plants still need review)
- [Paddy Doctor](https://paddydoc.github.io/dataset/) (source not handled by this Mendeley-only downloader)
- [Wheat Powdery Mildew dataset](https://zenodo.org/records/13137587) (separate Zenodo object-detection dataset; licence and bounding-box→class mapping need independent review; not handled by Mendeley downloader)
- [Maize Southern Corn Leaf Blight](https://data.mendeley.com/datasets/sb4wwjs8bz/1) (proposed maize class)
- [CornLeafDiseaseCollection](https://data.mendeley.com/datasets/w56xxnykcc/1) (publisher calls class “common rust (Southern rust)” — these are not automatically interchangeable)
- [Papaya leaf dataset](https://data.mendeley.com/datasets/3kwgxg4stb/1) (healthy and curl, exact disease taxonomy needs review)

Sources remain unapproved in the class-source CSV. **Do not mark “approved” without looking at original leaf images and documenting disease classification, usage rights and source provenance.** An indexed dataset's licence belongs to the original publisher, not to Google search.

## 2. Run Google Colab (user execution required)

Open [the development notebook](../notebooks/SmartPlant_50_Dataset_and_Training.ipynb) in Colab and click **Runtime → Run all** only after reviewing which cells download data or train. Instead of running all at once, the recommended order is to run the acquisition/review cells first: the training gates intentionally stop while class approvals are incomplete.

Discovery only (metadata reports; no downloads):

```bash
python scripts/acquire_smartplant50_sources.py \
  --registry dataset_sources/50class_source_register.csv \
  --workspace /content/drive/MyDrive/SmartPlant50
```

Automatic downloaded-source quarantine (4 GiB per run cap):

```bash
python scripts/acquire_smartplant50_sources.py \
  --registry dataset_sources/50class_source_register.csv \
  --workspace /content/drive/MyDrive/SmartPlant50 \
  --download --max-total-gib 4
```

*This tool only processes public Mendeley sources from the source register.* A missing SHA-256, unrecognized licence, oversized file or unexpected URL/redirect blocks the download. It never writes into train, val or test. The report `SmartPlant50/audit_reports/external_source_receipts.json` records verified files, skipped files and the source version. Caching verifies original checksums again.

## 3. Generate label review CSV automatically

```bash
python scripts/prepare_smartplant50_candidates.py \
  --legacy-dir /content/drive/MyDrive/SmartPlant_Datasets \
  --workspace /content/drive/MyDrive/SmartPlant50 \
  --plan /content/drive/MyDrive/SmartPlant50/data/expanded/class_plan.json
```

The output `SmartPlant50/audit_reports/source_folder_review_candidates.csv` lists each source ZIP, its SHA-256, internal label directory, candidate class name, original folder sample counts and **unapproved** fields.

For each folder you genuinely verify, fill in:
- `reviewed_class_id` — one of c000–c049, verified against the actual photos.
- `license_checked=yes` — explicit publisher licence and attribution verified.
- `taxonomy_checked=yes` — leaf disease label verified, not derived solely from filename.
- `source_group_csv` — absolute path to a CSV in SmartPlant50 with exactly two headers: `member_path,source_group_id`. Each source image must have its **actual original plant/farm ID**, not a made-up per-image identifier. Review CSVs are **never overwritten** automatically.

## 4. Preflight, then stage (only genuinely approved photos)

```bash
python scripts/verified_import_smartplant50.py \
  --review-csv /content/drive/MyDrive/SmartPlant50/audit_reports/source_folder_review_candidates.csv \
  --plan /content/drive/MyDrive/SmartPlant50/data/expanded/class_plan.json
```

This does a dry-run: images are decoded; SHA-256 and perceptual hashes are checked; source plant/farm groups are deterministically assigned to partitions; cross-split duplicates and cross-class conflicts are rejected. No images are copied by default.

If reviewers have approved some folders, pass `--commit` to write a **new** immutable candidate under `SmartPlant50/approved_staging`. Partial groups are allowed as *research staging*, but **cannot be used as a 50-class training dataset** until all 50 classes have enough independently sourced images and the strict data audit passes. Existing source ZIPs and working Android/Streamlit are untouched.

The importer does **not** automatically claim that the new images are diagnostically correct, nor that per-farm split independence is proven. Splitting by a user-reviewed source group is necessary but still must be audited.

## 5. Training and device testing remain separate

1. Review the label catalogue and collect all 50 classes without duplicate crop synonyms (Maize vs Corn).
2. Require original leaf photographs, data rights and legitimate group provenance.
3. Create a complete dataset with independently sourced test images; do not simply turn unverified staging into the official test set.
4. Use the existing isolated PyTorch training pipeline, then verify macro-F1, per-class recall, confusion matrix and out-of-distribution testing.
5. Export the truly trained 50-class checkpoint to ONNX with its own SHA-256. Build/test a separate Android APK and preserve the deployed 29-class model.
6. Only merge/deploy after explicit approval.

**No manual photo ZIP upload required for the sources in your Drive or supported Mendeley sources. Running Colab is still necessary because a GitHub code commit cannot execute a Colab runtime or authorize its Drive mount.**
