
import streamlit as st, numpy as np, pandas as pd, joblib
model, scaler = joblib.load("thyroid_rf_model.pkl"), joblib.load("scaler.pkl")
st.set_page_config(page_title="ThyroSense", layout="wide")
st.markdown("""
<style>
.big-title{text-align:center;font-size:50px;font-weight:bold;
background:-webkit-linear-gradient(#6C2BD9,#00C9A7);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub-title{text-align:center;font-size:22px;color:#555;}
.result-box{text-align:center;font-size:30px;font-weight:bold;padding:20px;border-radius:15px;}
.low{background:#d4edda;color:#155724;}
.medium{background:#fff3cd;color:#856404;}
.high{background:#f8d7da;color:#721c24;}
label,.stRadio label{font-size:20px!important;font-weight:600!important;}

.metric-box{
    text-align:center;
    padding:20px;
    border-radius:15px;
    background:#f5f5f5;
    font-size:22px;
    font-weight:600;
}
.metric-value{
    font-size:40px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">ThyroSense AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Smart Thyroid Risk Prediction System</div>', unsafe_allow_html=True)
with st.form("f"):
    c1,c2,c3 = st.columns(3)
    age = c1.slider("Age",10,90,30)
    gender = c1.radio("Gender",["Male","Female"])
    smoking = c1.radio("Smoking",["Yes","No"])
    obesity = c1.radio("Obesity",["Yes","No"])
    family = c2.radio("Family History",["Yes","No"])
    radiation = c2.radio("Radiation Exposure",["Yes","No"])
    iodine = c2.radio("Iodine Deficiency",["Yes","No"])
    diabetes = c2.radio("Diabetes",["Yes","No"])
    tsh = c3.number_input("TSH",0.0,10.0,2.5)
    t3 = c3.number_input("T3",0.0,5.0,1.5)
    t4 = c3.number_input("T4",0.0,15.0,8.0)
    nodule = c3.number_input("Nodule Size",0.0,5.0,1.0)
    diagnosis = c3.radio("Diagnosis",["Benign","Malignant"])
    submit = st.form_submit_button("Analyze Risk")

if submit:
    enc = lambda x: 1 if x in ["Male","Yes","Malignant"] else 0
    data = [[
        age, enc(gender), 0, 0,
        enc(family), enc(radiation), enc(iodine),
        enc(smoking), enc(obesity), enc(diabetes),
        tsh, t3, t4, nodule, enc(diagnosis)
    ]]
    cols = ['Age','Gender','Country','Ethnicity','Family_History',
            'Radiation_Exposure','Iodine_Deficiency','Smoking','Obesity',
            'Diabetes','TSH_Level','T3_Level','T4_Level','Nodule_Size','Diagnosis']
    input_df = pd.DataFrame(data, columns=cols)
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""
    <div class="metric-box">
    Low Risk<br>
    <span class="metric-value">{prob[0]*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="metric-box">
    Medium Risk<br>
    <span class="metric-value">{prob[1]*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)
    c3.markdown(f"""
    <div class="metric-box">
    High Risk<br>
    <span class="metric-value">{prob[2]*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-box {"low" if pred==0 else "medium" if pred==1 else "high"}">'
        f'{"LOW RISK" if pred==0 else "MEDIUM RISK" if pred==1 else "HIGH RISK"}</div>',
        unsafe_allow_html=True
    )
    st.progress(int(max(prob)*100))