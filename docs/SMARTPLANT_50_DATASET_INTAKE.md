# SmartPlant AI — 50-class training, next milestone

**Status:** Dataset sources are candidates, not approved training examples. No new model is trained or deployed.

The 50-class code introduced in merged PR #10 was subsequently rolled back from `main`. It remains available on `feature/smartplant-expanded-cnn-50`. This branch is **intentionally metadata-only**; it does not undo the rollback or change Android production.

## Data sourcing and verification

The registry `dataset_sources/50class_source_register.csv` has **all 21 proposed additions** (c029–c049). Every entry starts as unverified: zero downloaded original images; no source grouping; labels and licenses requiring review. Never use augmented images across different splits, and never silently equate two different disease names.

Source leads:
- Mango: https://data.mendeley.com/datasets/7ghdbftp54/1 (the publication describes Healthy, Anthracnose, Powdery Mildew).
- Guava: https://data.mendeley.com/datasets/fspx44mwfp/1 (leaf Healthy/Anthracnose listed; exact Leaf Spot mapping needs review).
- Rice: https://paddydoc.github.io/dataset/ (Normal, Brown Spot, Blast; verify image rights and whether Blast is leaf-specific).
- Banana: https://data.mendeley.com/datasets/5nfjzntwd8/1 (verify Sigatoka subclasses and rights).
- Wheat: https://data.mendeley.com/datasets/th422bg4yd/1 (leaf rust lead; additional healthy/powdery mildew needed).
- Citrus: https://data.mendeley.com/datasets/fhbvmpcyy2/3 (candidate Healthy and Canker leaf classes).
- Maize: https://data.mendeley.com/datasets/sb4wwjs8bz/1 (candidate Healthy and Northern Corn Leaf Blight).
- Papaya: https://data.mendeley.com/datasets/3kwgxg4stb/1 (verify exact Leaf Curl class before assigning).
- Legacy classes: https://github.com/spMohanty/PlantVillage-Dataset may cover many, but recover **the actual 29-class training source/provenance** independently.

Source sites and metadata need final licence/attribution checks before download/use. These URLs are **leads**, not a vetted, merged 50-class dataset.

## Inspect progress

```bash
python scripts/check_smartplant_sources.py
python scripts/check_smartplant_sources.py --require-ready
python -m unittest discover -s tests -p "test_smartplant_sources.py" -v
```

The first command checks structure and reports missing evidence. The second **must fail** until sources are verified. Do not interpret source-check success as model validation.

## Next: Colab training once images exist

Open `notebooks/SmartPlant_50_Dataset_and_Training.ipynb`. It uses a private Drive folder `SmartPlant50/data/expanded/` with the verified `class_plan.json`, all 50 folders c000–c049 in train/val/test, and optionally `groups.csv` with plant/farm groups. Training should be postponed if `groups.csv` cannot be established because split independence is uncertain.

The isolated 50-class scripts train a verified MobileNetV3-Small warm-start model and create metrics, confusion matrix and SHA-256, but **no valid performance result exists yet**.

## Publisher checks — class-mapping evidence (22 November 2026)

The following facts are taken from the publishers' public dataset descriptions, **not** from downloaded or audited training files. All corresponding register rows remain unapproved.

| Source | Publisher-confirmed labels/data | Important missing evidence |
| --- | --- | --- |
| [Mango Leaf Disease Dataset](https://data.mendeley.com/datasets/7ghdbftp54/1) | 250 processed Healthy, 326 Anthracnose and 206 Powdery Mildew images; publisher also reports augmented derivatives and CC BY 4.0 | Image provenance, originals-only split, attribution records and scientific label review |
| [Guava Fruits and Leaves](https://data.mendeley.com/datasets/fspx44mwfp/1) | Leaf subset includes Healthy (1,498 originals) and Anthracnose (237 originals). The dataset calls a separate label **Dot**, not **Leaf Spot**; license shown as CC BY 4.0 | `Guava | Leaf Spot` needs an independently verified label source; do not silently rename Dot |
| [Paddy Doctor](https://paddydoc.github.io/dataset/) | Field dataset lists Normal leaf, Brown spot and Blast | Exact `Rice | Leaf Blast` mapping, usage terms, split provenance, and image-file audit |
| [Wheat nitrogen deficiency and leaf rust](https://data.mendeley.com/datasets/th422bg4yd/1) | Includes control/healthy leaves and Leaf rust; publisher describes 368 rust and 491 corresponding control photographs; CC BY 4.0 | It does **not** establish a Powdery Mildew class; source needed for `Wheat | Powdery Mildew` |

Additional publisher descriptions reviewed:

| Source | Confirmed source categories | Remaining discrepancy |
| --- | --- | --- |
| [Banana and Banana Leaf Dataset](https://data.mendeley.com/datasets/5nfjzntwd8/1) | Healthy Banana Leaf (1,256 raw), Black Sigatoka (170 raw), plus other disease categories; CC BY 4.0 | **Yellow Sigatoka is not listed** and needs a separate source |
| [ABCGMP six-crop dataset](https://data.mendeley.com/datasets/fhbvmpcyy2/3) | Citrus Healthy Leaf and a category spelled **Cancker Leaf**; CC BY 4.0 | Require qualified review before mapping the misspelled category to Citrus Canker |
| [Papaya Leaf Disease Image Dataset](https://data.mendeley.com/datasets/3kwgxg4stb/1) | Healthy (594 raw) and Curl (778 raw) leaf images; CC BY 4.0 | Confirm whether `Curl` is appropriately labeled as `Papaya | Leaf Curl` |
| [Open-Source Multi-Format Maize Leaf Dataset](https://data.mendeley.com/datasets/sb4wwjs8bz/1) | Healthy Leaves and Northern Corn Leaf Blight; CC BY 4.0 | Confirm common disease-name alias and original source-level split |

**Dataset gaps:** `Guava | Leaf Spot`, `Banana | Yellow Sigatoka`, and `Wheat | Powdery Mildew` currently have no verified direct class-level source in this registry. Research different datasets rather than renaming unrelated diseases.

### Improved local diagnostics

```bash
python scripts/check_smartplant_sources.py
python scripts/inventory_smartplant50.py --dataset /path/to/expanded
python scripts/inventory_smartplant50.py --dataset /path/to/expanded --require-complete
```

The inventory never creates images, reassigns diagnoses, or starts training. It counts only recognized original image filenames and checks class folder structure plus the `groups.csv` provenance map. It **does not decode images**. The separate strict audit must still check image content and hashes before training.

The Colab notebook now uses subprocesses with `check=True` so failed source, structural and image audits stop the cell; its final training cell repeats all gates if run out of sequence. The notebook clones the research-intake and preserved model-development branches separately, because the current `main` branch no longer contains the expanded training scripts.

**Next non-code action:** recover or collect real licensed training photographs, especially `Guava | Leaf Spot`, `Wheat | Powdery Mildew`, and the old 29-class data. Do not claim data acquisition, reproducibility or a 50-class accuracy until source files and grouped splits are actually available.
