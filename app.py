import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

.main{
    background-color:#f4f8fb;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:#d90429;
    text-align:center;
}

.subtitle{
    font-size:20px;
    color:gray;
    text-align:center;
    margin-bottom:30px;
}

.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#ff416c,#ff4b2b);
    color:white;
    font-size:20px;
    border-radius:12px;
    height:55px;
    border:none;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#ff4b2b,#ff416c);
}

.card{
    padding:20px;
    border-radius:15px;
    background:white;
    box-shadow:0px 5px 15px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------- HEADER ---------------- #
st.markdown("<div class='title'>❤️ Heart Disease Prediction System</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>AI Powered Heart Disease Risk Predictor</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=120)
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age",18,100,40)

sex = st.sidebar.selectbox(
    "Gender",
    ["M","F"]
)

chest_pain = st.sidebar.selectbox(
    "Chest Pain Type",
    ["ATA","NAP","TA","ASY"]
)

resting_bp = st.sidebar.number_input(
    "Resting BP",
    80,200,120
)

cholesterol = st.sidebar.number_input(
    "Cholesterol",
    100,600,200
)

fasting_bs = st.sidebar.selectbox(
    "Fasting Blood Sugar",
    [0,1]
)

resting_ecg = st.sidebar.selectbox(
    "Resting ECG",
    ["Normal","ST","LVH"]
)

max_hr = st.sidebar.slider(
    "Maximum Heart Rate",
    60,220,150
)

exercise_angina = st.sidebar.selectbox(
    "Exercise Angina",
    ["Y","N"]
)

oldpeak = st.sidebar.slider(
    "OldPeak",
    0.0,6.0,1.0
)

st_slope = st.sidebar.selectbox(
    "ST Slope",
    ["Up","Flat","Down"]
)

# ---------------- DISPLAY INPUTS ---------------- #
col1,col2,col3 = st.columns(3)

with col1:
    st.metric("Age",age)
    st.metric("Blood Pressure",resting_bp)

with col2:
    st.metric("Cholesterol",cholesterol)
    st.metric("Max HR",max_hr)

with col3:
    st.metric("OldPeak",oldpeak)
    st.metric("Fasting BS",fasting_bs)

st.divider()

# ---------------- PREDICTION ---------------- #
if st.button("❤️ Predict Heart Disease"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0

    input_df=input_df[expected_columns]

    scaled_input=scaler.transform(input_df)

    prediction=model.predict(scaled_input)[0]

    st.divider()

    if prediction==1:

        st.error("⚠️ High Risk of Heart Disease")

        st.progress(90)

        st.markdown("""
        ### Recommendation

        - Visit a Cardiologist
        - Maintain Healthy Diet
        - Exercise Daily
        - Stop Smoking
        - Monitor Blood Pressure
        - Reduce Cholesterol
        """)

    else:

        st.success("✅ Low Risk of Heart Disease")

        st.progress(20)

        st.balloons()

        st.markdown("""
        ### Recommendation

        ✔ Continue Healthy Lifestyle

        ✔ Regular Exercise

        ✔ Healthy Food

        ✔ Routine Health Checkup
        """)
