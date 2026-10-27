"""Predictive maintenance research dashboard; compatible with Step 17 config/model."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
BLUE, RED, LIGHT_BLUE = '#17365D', '#C62828', '#83B4D8'

st.set_page_config(page_title='Predictive Maintenance | AI Research', page_icon='⚙️', layout='wide')
st.markdown('''<style>
.stApp {background:#F5F8FC}
.block-container {max-width:1370px; padding-top:1.5rem; padding-bottom:3rem}
h1,h2,h3,h4,label {font-weight:750!important;color:#17365D!important}
[data-testid="stSidebar"] {background:#EAF2FA;border-right:1px solid #DCE7F2}
[data-testid="stExpander"] {border:1px solid #DCE7F2;border-radius:12px;background:white;margin-bottom:.7rem}
[data-testid="stMetric"] {background:white;border:1px solid #DCE7F2;padding:15px;border-radius:12px}
[data-testid="stMetricValue"] {font-weight:800;color:#17365D}
[data-testid="stFormSubmitButton"] button[kind="primary"],div.stButton>button[kind="primary"] {background:#C62828;color:white;font-weight:800;border:0;min-height:45px;border-radius:10px}
.stTabs [data-baseweb="tab"] {font-weight:750}
.hero {border-radius:14px;padding:24px 28px;background:linear-gradient(110deg,#17365D,#28618F);color:white;margin-bottom:16px}
.hero h1 {font-size:1.76rem;color:white!important;margin:0 0 6px}
.hero p {color:#E4EFF9;margin:0}
.kpi {background:white;border:1px solid #DBE5EF;border-radius:12px;padding:15px;min-height:106px}
.kpi-title {font-size:.92rem;font-weight:800;color:#17365D}
.kpi-number {font-size:1.6rem;font-weight:850;color:#C62828;margin:4px 0}
.kpi-note {font-size:.8rem;color:#64748B}
.note-panel {background:white;border:1px solid #DBE5EF;border-radius:12px;padding:18px;margin-bottom:12px;color:#40516A;line-height:1.65}
.note-title {color:#17365D;font-weight:850;margin-bottom:5px}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    # Never load joblib files from user uploads; only trusted packaged artifacts.
    return joblib.load(BASE / 'model.joblib'), joblib.load(BASE / 'encoder.joblib')

@st.cache_data
def load_config():
    return json.loads((BASE / 'config.json').read_text(encoding='utf-8'))

model, encoder = load_artifacts()
config = load_config()
features = config['features']
columns = config['feature_order']
classes = [str(c) for c in encoder.classes_]
positive_class = str(config['positive_class'])
threshold = float(config['threshold'])
if len(columns) != 6 or len(classes) != 2 or positive_class != '1' or abs(threshold - .25)>1e-10:
    raise RuntimeError('Model configuration does not match the frozen maintenance research model.')
feature_map = {f['name']:f for f in features}
pos_index = classes.index(positive_class)

LABELS = {
    'Type':'Machine Type',
    'Air temperature':'Air Temperature (K)',
    'Process temperature':'Process Temperature (K)',
    'Rotational speed':'Rotational Speed (rpm)',
    'Torque':'Torque (Nm)',
    'Tool wear':'Tool Wear (min)',
}
HELP = {
    'Type':'Machine product-quality group from the AI4I dataset: L, M, or H.',
    'Air temperature':'Ambient air temperature measured in kelvin.',
    'Process temperature':'Operating process temperature measured in kelvin.',
    'Rotational speed':'Machine shaft rotational speed in revolutions per minute.',
    'Torque':'Machine torque measured in newton-metres.',
    'Tool wear':'Cumulative tool-wear time (minutes).',
}


def cast_value(value, field):
    if pd.isna(value):
        return np.nan
    kind = field['data_type']
    if kind == 'integer':
        num = float(value)
        if not np.isfinite(num) or not num.is_integer():
            raise ValueError(f"{field['name']}: integer value required")
        return int(num)
    if kind == 'float':
        num = float(value)
        if not np.isfinite(num):
            raise ValueError(f"{field['name']}: finite numeric value required")
        return num
    return str(value)


def prepare_frame(data):
    missing = [name for name in columns if name not in data.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    X = data[columns].copy()
    for name in columns:
        feature = feature_map[name]
        X[name] = X[name].map(lambda value, f=feature: cast_value(value,f))
        if feature['input_type'] == 'select':
            accepted = {str(v) for v in feature['choices']}
            received = {str(v) for v in X[name].dropna().unique()}
            invalid = received - accepted
            if invalid:
                raise ValueError(f'{name}: invalid category {sorted(invalid)}')
    return X


def predict_rows(df):
    X = prepare_frame(df)
    probabilities = model.predict_proba(X)
    labels_idx = np.where(probabilities[:,pos_index]>=threshold,pos_index,1-pos_index)
    labels = encoder.inverse_transform(labels_idx)
    output = pd.DataFrame({'Prediction_Code':labels,
                           'Prediction':np.where(np.asarray(labels).astype(str)=='1','Failure risk alert','No failure alert')})
    for i,c in enumerate(classes):
        output[f'Probability_{c}'] = probabilities[:,i]
    return output


def probability_figure(probabilities):
    failure = float(probabilities[pos_index])*100
    normal = float(probabilities[1-pos_index])*100
    fig,ax = plt.subplots(figsize=(6.1,2.1))
    y = np.arange(2)
    vals = [normal, failure]
    ax.barh(y,vals,color=[BLUE,RED],height=.47)
    ax.set_yticks(y,['No failure','Machine failure'],fontsize=9,fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0,112)
    ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel('Model-estimated probability (%)',fontweight='bold')
    ax.grid(axis='x',alpha=.16)
    ax.set_axisbelow(True)
    for i,v in enumerate(vals):
        ax.text(min(v+2,105),i,f'{v:.1f}%',va='center',fontweight='bold',color=BLUE)
    for tick in ax.get_xticklabels():tick.set_fontweight('bold')
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    fig.patch.set_facecolor('white')
    fig.tight_layout(pad=1.2)
    return fig


with st.sidebar:
    st.markdown('## ⚙️ Maintenance AI')
    st.caption('Research dashboard | XGBoost')
    st.divider()
    st.markdown('**Model specification**')
    st.write('**Inputs:** 6 measurements/attributes')
    st.write('**Output:** machine failure risk')
    st.write('**Alert threshold:** 0.25')
    st.divider()
    st.warning('Synthetic benchmark. Not a substitute for industrial safety monitoring.')

st.markdown('''<div class="hero"><h1>⚙️ Predictive Maintenance Intelligence Dashboard</h1>
<p>Explainable machine-learning framework for machine condition monitoring and failure-risk assessment</p></div>''',unsafe_allow_html=True)

head_a,head_b,head_c = st.columns(3,gap='medium')
with head_a:
    st.markdown('<div class="kpi"><div class="kpi-title">🔵 Model Inputs</div><div class="kpi-number">6</div><div class="kpi-note">Five numeric measurements + machine type</div></div>',unsafe_allow_html=True)
with head_b:
    st.markdown('<div class="kpi"><div class="kpi-title">🔴 Failure Alert Threshold</div><div class="kpi-number">25%</div><div class="kpi-note">Frozen, training-selected classification cutoff</div></div>',unsafe_allow_html=True)
with head_c:
    st.markdown('<div class="kpi"><div class="kpi-title">⚙️ Research Model</div><div class="kpi-number" style="font-size:1.35rem;color:#17365D">XGBoost</div><div class="kpi-note">Saved preprocessing + trained classifier</div></div>',unsafe_allow_html=True)

st.caption('Demonstration trained on the synthetic AI4I predictive-maintenance dataset; alerts are not validated industrial safety decisions.')

individual,batch,explain = st.tabs(['🎯 Individual Prediction','📁 Batch CSV Prediction','📊 Model Explainability'])

with individual:
    left,right = st.columns([3.1,1.15],gap='large')
    with left:
        st.markdown('### Machine Operating Conditions')
        st.caption('Enter the measured conditions, then generate the risk assessment.')
        form_values = {}
        with st.form('machine_prediction_form'):
            with st.expander('01  Machine Identification',expanded=True):
                feat = feature_map['Type']
                options = feat['choices']
                default_index = options.index(feat['default']) if feat['default'] in options else 0
                form_values['Type'] = st.selectbox('Machine Type',options,index=default_index,
                    format_func=lambda v: {'L':'L — Low quality','M':'M — Medium quality','H':'H — High quality'}.get(str(v),str(v)),
                    help=HELP['Type'])
            with st.expander('02  Temperature and Motion Sensors',expanded=True):
                cols=st.columns(3,gap='medium')
                for i,name in enumerate(['Air temperature','Process temperature','Rotational speed']):
                    feat=feature_map[name]
                    with cols[i]:
                        if feat['data_type']=='integer':
                            form_values[name]=st.number_input(LABELS[name],value=int(feat['default']),step=1,help=HELP[name])
                        else:
                            form_values[name]=st.number_input(LABELS[name],value=float(feat['default']),step=.1,format='%.2f',help=HELP[name])
            with st.expander('03  Load and Tool Condition',expanded=True):
                cols=st.columns(2,gap='medium')
                for i,name in enumerate(['Torque','Tool wear']):
                    feat=feature_map[name]
                    with cols[i]:
                        if feat['data_type']=='integer':
                            form_values[name]=st.number_input(LABELS[name],value=int(feat['default']),step=1,help=HELP[name])
                        else:
                            form_values[name]=st.number_input(LABELS[name],value=float(feat['default']),step=.1,format='%.2f',help=HELP[name])
            submitted=st.form_submit_button('⚡ Analyze Machine Failure Risk',type='primary',use_container_width=True)
        if submitted:
            try:
                output=predict_rows(pd.DataFrame([form_values]))
                st.session_state['maintenance_result']={
                    'label':str(output.loc[0,'Prediction']),
                    'probabilities':[float(output.loc[0,f'Probability_{x}']) for x in classes]
                }
            except Exception as exc:
                st.error(f'Unable to evaluate machine state: {exc}')
        if 'maintenance_result' in st.session_state:
            r=st.session_state['maintenance_result']
            probs=np.asarray(r['probabilities'],dtype=float)
            risk=100*probs[pos_index]
            st.markdown('### Model Prediction Results')
            x,y,z=st.columns(3)
            x.metric('Classification',r['label'])
            y.metric('Predicted Failure Probability',f'{risk:.2f}%')
            z.metric('Alert Threshold','25.00%')
            if risk>=100*threshold:
                st.error('Research model alert: predicted failure probability reaches the 25% threshold.')
            else:
                st.success('No model alert at the frozen 25% threshold. This does not guarantee safe operation.')
            chart=probability_figure(probs)
            st.pyplot(chart,use_container_width=False)
            plt.close(chart)
            st.caption('This is a probability estimate from a synthetic-data model; the output is not a diagnosis of a physical fault.')
    with right:
        st.markdown('''<div class="note-panel"><div class="note-title">ℹ️ About the application</div>
        This model compares machine operating measurements to patterns in the AI4I benchmark to estimate failure risk.</div>''',unsafe_allow_html=True)
        st.markdown('''<div class="note-panel"><div class="note-title">📡 Inputs</div>
        Air and process temperatures, rotational speed, torque, tool wear and machine type.</div>''',unsafe_allow_html=True)
        st.markdown('''<div class="note-panel"><div class="note-title">🔴 Alert policy</div>
        Failure probability ≥ 0.25 generates an alert. This threshold was frozen before final testing.</div>''',unsafe_allow_html=True)
        st.markdown('''<div class="note-panel"><div class="note-title">🧪 Research limitation</div>
        The trained dataset is synthetic. Real industrial deployment requires external testing and sensor-quality controls.</div>''',unsafe_allow_html=True)

with batch:
    st.markdown('### Batch Machine-Failure Predictions')
    st.write('Upload CSV records with these six exact columns: '+', '.join(f'`{c}`' for c in columns)+'.')
    template=pd.DataFrame([{f['name']:f['default'] for f in features}])
    st.download_button('⬇ Download Sample CSV',template.to_csv(index=False),file_name='maintenance_input_template.csv',mime='text/csv')
    uploaded=st.file_uploader('Upload machine measurements (.csv)',type='csv')
    if uploaded is not None:
        try:
            data=pd.read_csv(uploaded)
            st.caption(f'{len(data)} records detected')
            if len(data)>5000:
                st.error('Maximum batch size: 5,000 rows.')
            elif st.button('Run Batch Predictions',type='primary'):
                results=predict_rows(data)
                st.dataframe(results,use_container_width=True,hide_index=True)
                st.download_button('⬇ Download Prediction Results',results.to_csv(index=False),
                                   'maintenance_batch_results.csv',mime='text/csv')
        except Exception as exc:
            st.error(f'Batch prediction error: {exc}')

with explain:
    st.markdown('### Global XGBoost Feature Importance')
    st.info('This is global feature importance, not per-machine SHAP and not a causal explanation.')
    try:
        transformer=model.named_steps['preprocessing']
        classifier=model.named_steps['classifier']
        names=transformer.get_feature_names_out()
        importance=pd.DataFrame({'Feature':names,'Importance':classifier.feature_importances_}).sort_values('Importance',ascending=False)
        display=importance.head(10).iloc[::-1]
        fig,ax=plt.subplots(figsize=(6.3,3.5))
        ax.barh(display['Feature'].str.replace(r'^(numeric|categorical|num|cat)__','',regex=True),
                display['Importance'],color=[BLUE if i%2==0 else RED for i in range(len(display))])
        ax.set_xlabel('XGBoost feature importance',fontweight='bold')
        for tick in ax.get_xticklabels()+ax.get_yticklabels(): tick.set_fontweight('bold')
        ax.tick_params(axis='y',labelsize=8)
        ax.grid(axis='x',alpha=.15)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig,use_container_width=False)
        plt.close(fig)
        st.download_button('⬇ Download Importance Table',importance.to_csv(index=False),
                           'maintenance_feature_importance.csv',mime='text/csv')
    except Exception as exc:
        st.error(f'Explainability view could not be generated: {exc}')

st.divider()
st.caption('IEEE research prototype · AI4I synthetic benchmark · Frozen model threshold 0.25 · Not a live industrial monitoring system')
