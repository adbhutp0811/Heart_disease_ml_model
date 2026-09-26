from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart health screening",
    page_icon=":material/monitor_heart:",
    layout="centered",
)


@st.cache_resource
def load_model_assets():
    assets_dir = Path(__file__).resolve().parent
    model = joblib.load(assets_dir / "knn_heart_model.pkl")
    scaler = joblib.load(assets_dir / "heart_scaler.pkl")
    expected_columns = joblib.load(assets_dir / "heart_columns.pkl")
    return model, scaler, expected_columns


model, scaler, expected_columns = load_model_assets()

st.title("Heart health screening", icon=":material/monitor_heart:")
st.caption(
    "Enter the measurements below for an educational model estimate. "
    "This tool does not provide a diagnosis."
)

sex_options = {"Female": "F", "Male": "M"}
chest_pain_options = {
    "Atypical angina": "ATA",
    "Non-anginal pain": "NAP",
    "Typical angina": "TA",
    "Asymptomatic": "ASY",
}
resting_ecg_options = {
    "Normal": "Normal",
    "ST-T wave abnormality": "ST",
    "Left ventricular hypertrophy": "LVH",
}
st_slope_options = {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}

with st.form("screening_form", border=False):
    left_column, right_column = st.columns(2)

    with left_column:
        with st.container(border=True):
            st.subheader("Patient profile", icon=":material/person:")
            age = st.slider("Age (years)", 18, 100, 40)
            sex_label = st.segmented_control(
                "Sex", options=list(sex_options), default="Female", key="sex"
            )

        with st.container(border=True):
            st.subheader("Blood measurements", icon=":material/monitoring:")
            resting_bp = st.number_input(
                "Resting blood pressure (mm Hg)", 80, 200, 120
            )
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
            fasting_bs_label = st.segmented_control(
                "Fasting blood sugar above 120 mg/dL",
                options=["No", "Yes"],
                default="No",
                key="fasting_bs",
            )

    with right_column:
        with st.container(border=True):
            st.subheader("Symptoms", icon=":material/healing:")
            chest_pain_label = st.selectbox(
                "Chest pain type", options=list(chest_pain_options)
            )
            exercise_angina_label = st.segmented_control(
                "Angina during exercise",
                options=["No", "Yes"],
                default="No",
                key="exercise_angina",
            )

        with st.container(border=True):
            st.subheader("Heart test results", icon=":material/ecg_heart:")
            resting_ecg_label = st.selectbox(
                "Resting ECG", options=list(resting_ecg_options)
            )
            max_hr = st.slider("Maximum heart rate (bpm)", 60, 220, 150)
            oldpeak = st.slider(
                "ST depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1
            )
            st_slope_label = st.selectbox("ST slope", options=list(st_slope_options))

    submitted = st.form_submit_button(
        "Generate screening estimate",
        type="primary",
        icon=":material/analytics:",
        width="stretch",
    )

if submitted:
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": int(fasting_bs_label == "Yes"),
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        f"Sex_{sex_options[sex_label]}": 1,
        f"ChestPainType_{chest_pain_options[chest_pain_label]}": 1,
        f"RestingECG_{resting_ecg_options[resting_ecg_label]}": 1,
        f"ExerciseAngina_{'Y' if exercise_angina_label == 'Yes' else 'N'}": 1,
        f"ST_Slope_{st_slope_options[st_slope_label]}": 1,
    }

    input_df = pd.DataFrame([raw_input])
    for column in expected_columns:
        if column not in input_df.columns:
            input_df[column] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.subheader("Screening estimate", icon=":material/analytics:")
    with st.container(border=True):
        if prediction == 1:
            st.error(
                "The model placed this input in its higher-risk group.",
                icon=":material/warning:",
            )
        else:
            st.success(
                "The model placed this input in its lower-risk group.",
                icon=":material/check_circle:",
            )
        st.write(
            "This classification is not a probability or a diagnosis. "
            "Discuss health concerns with a qualified healthcare professional."
        )

st.caption(
    "For educational use only. Do not use this result to make medical decisions."
)