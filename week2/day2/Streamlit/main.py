import streamlit as st
import pandas as pd
import joblib


def float_to_int(x):
    return x.astype(int)


st.title("Heart Disease Data Form")

age = st.number_input("Age (years):", min_value=20, max_value=120, value=50)
sex = st.selectbox("Sex:", ["M", "F"])
# chest_pain_type = st.selectbox(
#     "Chest Pain Type:",
#     [
#         "TA: Typical Angina",
#         "ATA: Atypical Angina",
#         "NAP: Non-Anginal Pain",
#         "ASY: Asymptomatic",
#     ],
# )
chest_pain_type = st.selectbox(
    "Chest Pain Type:",
    [
        "TA",
        "ATA",
        "NAP",
        "ASY",
    ],
)
resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg):", min_value=50, max_value=250, value=120
)
cholesterol = st.number_input(
    "Serum Cholesterol (mg/dl):", min_value=100, max_value=600, value=200
)
fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl:", [0, 1])
resting_ecg = st.selectbox("Resting ECG:", ["Normal", "ST", "LVH"])
max_hr = st.number_input(
    "Maximum Heart Rate Achieved:", min_value=60, max_value=202, value=150
)
exercise_angina = st.radio("Exercise-Induced Angina:", ["Y", "N"])
oldpeak = st.number_input(
    "Oldpeak (ST depression):", min_value=0.0, max_value=10.0, step=0.1, value=1.0
)
st_slope = st.selectbox("ST Slope:", ["Up", "Flat", "Down"])

if st.button("Submit"):
    data = {
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain_type],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope],
    }

    df = pd.DataFrame(data)

    st.write("Collected Data")
    st.dataframe(df)
    st.write(df.columns)

    model = joblib.load(
        "/home/dmitry/elbrus/Phase1/ds-phase1-git/week2/day2/Streamlit/rf_pipe.pkl"
    )

    prediction = model.predict(df)
    if st.button("Predict"):
        st.write("Predicted Heart Disease:", "Yes" if prediction[0] == 1 else "No")
