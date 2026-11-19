import io
import numpy as np
import pytest
from PIL import Image
from smartplant.core import decode_image, extract_features, leaf_observations, fit_model, predict

def make_image(color):
    return Image.new("RGB", (160,160), color)

def test_decode_valid_image():
    data = io.BytesIO()
    make_image((20,130,30)).save(data,format="PNG")
    assert decode_image(data.getvalue()).size == (128,128)

def test_decode_invalid_image():
    with pytest.raises(ValueError):
        decode_image(b"not an image")

def test_features_finite_and_repeatable():
    im = make_image((10,120,20))
    a = extract_features(im)
    b = extract_features(im)
    assert len(a) > 100
    assert np.isfinite(a).all()
    assert np.array_equal(a,b)

def test_green_leaf_observation():
    result = leaf_observations(make_image((10,150,20)))
    assert result["green_pixels_pct"] == 100.0

def test_insufficient_training_data_rejected():
    with pytest.raises(ValueError):
        fit_model([make_image((1,2,3))]*4,["Healthy"]*4)

def test_fit_and_predict_on_toy_images():
    images = []
    labels = []
    rng = np.random.default_rng(42)
    for name, base in [("Healthy", (20,150,30)),("Brown_spot",(150,65,25))]:
        for _ in range(10):
            c = tuple(int(np.clip(v+rng.integers(-15,16),0,255)) for v in base)
            images.append(make_image(c))
            labels.append(name)
    model, metrics = fit_model(images, labels)
    results = predict(model, make_image((25,140,30)))
    assert len(results) == 2
    assert abs(sum(p for _,p in results)-1) < 1e-6
    assert 0 <= model.validation_accuracy <= 1
    assert metrics["validation_count"] == 5
