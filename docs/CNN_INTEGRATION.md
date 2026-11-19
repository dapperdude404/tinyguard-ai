# SmartPlant CNN integration

This branch adds optional MobileNetV3-Small inference to the existing Streamlit app. It **does not** include trained CNN weights and **does not** claim a new accuracy.

## Training

Use the previously supplied `SmartPlant_CNN_Upgrade.zip` training package. Place all nine plant ZIP datasets in one directory, then run:

```bash
python train_cnn.py --data-dir /path/to/plant-zips --output-dir outputs --epochs 12 --pretrained
```

After inspecting `outputs/test_metrics.json` and confirming independent evaluation, copy `outputs/best_model.pt` to `models/smartplant_mobilenetv3.pt` in the deployed app. Avoid committing large checkpoints to Git; use a trusted artifact distribution method. The checkpoint must contain `state_dict` and `labels` produced by the training script.

## Behavior

- With a compatible trained CNN checkpoint, the app shows top-five CNN predictions.
- Without a CNN checkpoint, it falls back to the existing Random Forest model when available.
- Without either model, it shows only descriptive leaf-color indicators.
- No result should be interpreted as a definitive plant-disease diagnosis.

## Before deployment

Verify compatible PyTorch/Torchvision wheels and Streamlit memory limits, run smoke tests with actual leaf images, and audit possible train/test leakage. Keep unverified personal field photographs outside supervised disease training until labeled.
