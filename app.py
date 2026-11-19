import io
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tinyguard.core import FEATURES, simulate, prepare, fit_detector, evaluate, split_normal_train_test

st.set_page_config(page_title='TinyGuard AI | Research Studio', page_icon='🛡️', layout='wide')
st.markdown('''<style>.block-container{padding-top:1.4rem;max-width:1300px}h1{letter-spacing:-.035em}div[data-testid="stMetric"]{border:1px solid #334155;border-radius:14px;padding:14px}</style>''', unsafe_allow_html=True)
st.title('🛡️ TinyGuard AI')
st.caption('Explainable anomaly detection • Tiny-model research • M.Tech AI classroom edition')

with st.sidebar:
    st.header('Experiment settings')
    source = st.radio('Data source', ['Reproducible simulation', 'Upload CSV'])
    kind = st.selectbox('Detector', ['Isolation Forest', 'Tiny Autoencoder'])
    seed = st.number_input('Random seed', min_value=0, max_value=100000, value=42)
    quantile = st.slider('Training normal-score threshold percentile', 90, 99, 97) / 100
    n = st.slider('Simulation samples', 300, 3000, 1200, 100) if source == 'Reproducible simulation' else None
    uploaded = st.file_uploader('CSV: temperature, vibration, humidity, current; optional anomaly (0/1)', type='csv') if source == 'Upload CSV' else None

if source == 'Upload CSV':
    if uploaded is None:
        st.info('Upload a CSV to start. Sensor column names must match the documented schema.')
        st.stop()
    try:
        df = pd.read_csv(uploaded)
        prepare(df)
        if len(df) < 60: raise ValueError('At least 60 rows are needed')
        if 'anomaly' not in df.columns:
            st.warning('No ground-truth anomaly column. Evaluation scores are unavailable; a baseline assumes first 65% is normal.')
            df['anomaly'] = 0
            labels_available = False
        else:
            if not df['anomaly'].isin([0,1]).all(): raise ValueError('anomaly must be 0 or 1')
            labels_available = True
        df['sample'] = np.arange(len(df))
    except Exception as exc:
        st.error(f'Invalid CSV: {exc}')
        st.stop()
else:
    df = simulate(n=n, seed=int(seed))
    labels_available = True

try:
    x_train, x_test, y_test = split_normal_train_test(df)
    detector = fit_detector(x_train, kind=kind, seed=int(seed), threshold_quantile=quantile)
    predictions, scores = detector.predict(x_test)
except Exception as exc:
    st.error(f'Experiment cannot run: {exc}')
    st.stop()

st.subheader('Detection overview')
metrics = evaluate(detector, x_test, y_test) if labels_available else None
cols = st.columns(4)
cols[0].metric('Test observations', len(x_test))
cols[1].metric('Flagged anomalies', int(predictions.sum()))
cols[2].metric('F1 score', f"{metrics['f1']:.3f}" if metrics else 'N/A')
cols[3].metric('ROC AUC', f"{metrics['roc_auc']:.3f}" if metrics and metrics['roc_auc'] is not None else 'N/A')
st.caption('Metrics use a chronological holdout and synthetic labels when simulation is selected. Scores are not evidence of field performance.')

tab1, tab2, tab3, tab4 = st.tabs(['📈 Sensor monitoring', '🔬 Evaluation', '🧪 Live demo', '📄 Research & export'])
with tab1:
    feature = st.selectbox('Plot sensor', FEATURES)
    test = df.iloc[int(len(df)*.65):].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test['sample'], y=test[feature], mode='lines', name=feature, line=dict(color='#38bdf8')))
    flagged = test.iloc[np.where(predictions == 1)[0]]
    fig.add_trace(go.Scatter(x=flagged['sample'], y=flagged[feature], mode='markers', name='Predicted anomaly', marker=dict(color='#fb7185', size=9, symbol='x')))
    fig.update_layout(template='plotly_dark', height=420, xaxis_title='Sample index', yaxis_title=feature)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(test.assign(predicted_anomaly=predictions, anomaly_score=scores).tail(30), use_container_width=True)
with tab2:
    if metrics:
        a,b,c = st.columns(3)
        a.metric('Precision', f"{metrics['precision']:.3f}")
        b.metric('Recall', f"{metrics['recall']:.3f}")
        c.metric('F1', f"{metrics['f1']:.3f}")
        st.write('Confusion matrix (rows: actual 0/1; columns: predicted 0/1)')
        st.dataframe(pd.DataFrame(metrics['confusion'], index=['Actual normal','Actual anomaly'], columns=['Predicted normal','Predicted anomaly']))
    else:
        st.info('Provide a ground-truth anomaly column for evaluation metrics.')
    st.markdown('**Method:** Fit the scaler and detector only on normal samples from the first 65% of the chronological dataset. Select the threshold from training-normal scores. Evaluate on the remaining 35%.')
with tab3:
    st.write('Change sensor values and click Detect to demonstrate a single prediction.')
    values = {}
    defaults = [42., .35, 48., 5.]
    ranges = [(0.,150.),(0.,3.),(0.,100.),(0.,20.)]
    inputs = st.columns(4)
    for i, feature in enumerate(FEATURES):
        values[feature] = inputs[i].number_input(feature.title(), min_value=ranges[i][0], max_value=ranges[i][1], value=defaults[i], step=.1)
    if st.button('🔎 Detect anomaly', type='primary'):
        row = pd.DataFrame([values])
        pred, score = detector.predict(prepare(row))
        if pred[0]: st.error(f'ANOMALY DETECTED · Score {score[0]:.4f} · Threshold {detector.threshold:.4f}')
        else: st.success(f'Normal pattern · Score {score[0]:.4f} · Threshold {detector.threshold:.4f}')
        st.caption('Anomaly scores are model-dependent; they are not calibrated probabilities.')
with tab4:
    st.markdown('''**Research question:** How does lightweight reconstruction-based detection compare with an Isolation Forest baseline on simulated IoT sensor anomalies?

**Limitations:** The dataset is synthetic. The sklearn autoencoder is a compact research baseline, not a TensorFlow Lite model. Use the optional TensorFlow export script to produce an int8 TFLite artifact and benchmark on actual target hardware before claiming embedded deployment. No measured device latency or energy consumption is asserted.''')
    output = df.iloc[int(len(df)*.65):].copy()
    output['predicted_anomaly'] = predictions
    output['anomaly_score'] = scores
    st.download_button('Download predictions CSV', output.to_csv(index=False), 'tinyguard_predictions.csv', 'text/csv')
    st.download_button('Download sample dataset CSV', simulate().to_csv(index=False), 'tinyguard_sample.csv', 'text/csv')
st.divider()
st.caption('TinyGuard AI · Research demonstration only. Not a certified industrial safety monitoring system.')
