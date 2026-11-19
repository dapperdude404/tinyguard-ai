import numpy as np
import pandas as pd
import pytest
from tinyguard.core import simulate, prepare, split_normal_train_test, fit_detector, evaluate

def test_simulation_reproducible():
    assert simulate(seed=8).equals(simulate(seed=8))

def test_simulation_label_count():
    assert simulate(500, .1)['anomaly'].sum() == 50

def test_bad_columns_rejected():
    with pytest.raises(ValueError, match='Missing'):
        prepare(pd.DataFrame({'temperature':[1]}))

def test_nonfinite_rejected():
    x = simulate(100)
    x.loc[0,'temperature'] = np.inf
    with pytest.raises(ValueError, match='finite'):
        prepare(x)

@pytest.mark.parametrize('kind',['Isolation Forest','Tiny Autoencoder'])
def test_train_predict_evaluate(kind):
    df = simulate(300)
    train, test, y = split_normal_train_test(df)
    model = fit_detector(train, kind=kind)
    pred, scores = model.predict(test)
    assert pred.shape == scores.shape == (105,)
    assert np.isfinite(scores).all()
    report = evaluate(model, test, y)
    assert 0 <= report['f1'] <= 1
    assert sum(map(sum, report['confusion'])) == len(y)

def test_invalid_model_rejected():
    with pytest.raises(ValueError):
        fit_detector(np.ones((40,4)), kind='unknown')
