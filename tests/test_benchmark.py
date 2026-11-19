import pytest
from benchmark import run_benchmark


def test_benchmark_models_and_metrics():
    results = run_benchmark(seeds=1, samples=120)
    assert set(results["model"]) == {"Isolation Forest", "Tiny Autoencoder"}
    assert len(results) == 2
    for metric in ("precision", "recall", "f1", "roc_auc"):
        assert results[metric].between(0, 1).all()
    assert (results[["tn", "fp", "fn", "tp"]].sum(axis=1) == results["test_count"]).all()


def test_benchmark_invalid_parameters():
    with pytest.raises(ValueError):
        run_benchmark(seeds=0)
    with pytest.raises(ValueError):
        run_benchmark(samples=30)
