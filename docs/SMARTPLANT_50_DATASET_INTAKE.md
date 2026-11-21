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
