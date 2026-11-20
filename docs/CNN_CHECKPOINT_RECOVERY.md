# CNN checkpoint compatibility recovery (PyTorch weights-only)

The production error `WeightsUnpickler error: Unsupported operand 13` means PyTorch rejected the current checkpoint under safe `weights_only=True` loading. **Do not change the Streamlit server to `weights_only=False`.** GitHub Actions passing does not verify this external checkpoint.

## Preferred repair: re-export from the original training notebook

In the original trusted Colab session, where the trained model and label mapping are already in memory, save a fresh checkpoint using only tensors and primitive metadata:

```python
import torch
torch.save({
    "architecture": "mobilenet_v3_small",
    "input_size": 160,
    "labels": {str(name): int(idx) for name, idx in class_to_idx.items()},
    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
}, "best_model_safe.pt", pickle_protocol=2)
test = torch.load("best_model_safe.pt", map_location="cpu", weights_only=True)
assert len(test["labels"]) == 29
```

Replace `model` and `class_to_idx` with the actual training notebook variable names. If they are no longer available, restore the training session from **your own trusted source** in an isolated environment; do not use unsafe loading in the public Streamlit application.

Upload the newly exported `best_model_safe.pt` to Drive, then update the downloader's file ID and SHA-256 only after verifying the exact new file. Run `python scripts/check_cnn_checkpoint.py` and test real leaf inference in a separate Streamlit test app.

Do not merge or deploy this fix until a compatible, hash-verified model passes these checks.
