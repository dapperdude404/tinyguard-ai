"""Deterministic simulation, leakage-safe training, evaluation and inference."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

FEATURES = ['temperature', 'vibration', 'humidity', 'current']


def simulate(n=1200, anomaly_fraction=0.12, seed=42):
    if n < 50 or not 0 <= anomaly_fraction < 0.5:
        raise ValueError('n >= 50 and 0 <= anomaly_fraction < 0.5 required')
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    data = np.column_stack([
        42 + 2*np.sin(t/30) + rng.normal(0, 1.5, n),
        0.35 + 0.06*np.sin(t/20) + rng.normal(0, .045, n),
        48 + 4*np.sin(t/45) + rng.normal(0, 2, n),
        5 + .3*np.sin(t/25) + rng.normal(0, .18, n),
    ])
    y = np.zeros(n, dtype=int)
    k = int(n * anomaly_fraction)
    idx = rng.choice(n, k, replace=False)
    y[idx] = 1
    data[idx, 0] += rng.uniform(13, 30, k)
    data[idx, 1] += rng.uniform(.3, .85, k)
    data[idx, 3] += rng.uniform(1.1, 2.5, k)
    df = pd.DataFrame(data, columns=FEATURES)
    df['anomaly'] = y
    df['sample'] = t
    return df


def prepare(df):
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f'Missing sensor columns: {sorted(missing)}')
    x = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    if x.isna().any().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Sensor columns must contain finite numeric values')
    return x.to_numpy(dtype=float)


@dataclass
class Detector:
    scaler: StandardScaler
    model: object
    threshold: float
    kind: str

    def scores(self, x):
        z = self.scaler.transform(x)
        if self.kind == 'Isolation Forest':
            return -self.model.score_samples(z)
        return np.mean((z - self.model.predict(z)) ** 2, axis=1)

    def predict(self, x):
        s = self.scores(x)
        return (s > self.threshold).astype(int), s


def fit_detector(train_normal, kind='Isolation Forest', seed=42, threshold_quantile=.97):
    if kind not in ('Isolation Forest', 'Tiny Autoencoder'):
        raise ValueError('Unknown model kind')
    if len(train_normal) < 20:
        raise ValueError('At least 20 normal training examples required')
    scaler = StandardScaler().fit(train_normal)
    z = scaler.transform(train_normal)
    if kind == 'Isolation Forest':
        model = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1).fit(z)
        scores = -model.score_samples(z)
    else:
        model = MLPRegressor(hidden_layer_sizes=(8, 3, 8), activation='tanh',
                             solver='adam', max_iter=450, random_state=seed,
                             early_stopping=False).fit(z, z)
        scores = np.mean((z - model.predict(z)) ** 2, axis=1)
    return Detector(scaler, model, float(np.quantile(scores, threshold_quantile)), kind)


def evaluate(detector, x, y):
    pred, scores = detector.predict(x)
    result = dict(precision=float(precision_score(y, pred, zero_division=0)),
                  recall=float(recall_score(y, pred, zero_division=0)),
                  f1=float(f1_score(y, pred, zero_division=0)),
                  confusion=confusion_matrix(y, pred, labels=[0, 1]).tolist())
    result['roc_auc'] = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None
    return result


def split_normal_train_test(df, train_fraction=.65):
    """Chronological holdout: normal-only train; later observations for testing."""
    cut = int(len(df)*train_fraction)
    train = df.iloc[:cut]
    normal = train.loc[train['anomaly'] == 0]
    if len(normal) < 20:
        raise ValueError('Insufficient normal training rows')
    return prepare(normal), prepare(df.iloc[cut:]), df.iloc[cut:]['anomaly'].to_numpy(dtype=int)
