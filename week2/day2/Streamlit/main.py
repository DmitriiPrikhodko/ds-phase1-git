import streamlit as st
import pandas as pd
import joblib
import sklearn


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OrdinalEncoder,
    TargetEncoder,
    FunctionTransformer,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

# for model learning
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
)

# models
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    BaggingClassifier,
)

# Metrics
from sklearn.metrics import accuracy_score

sklearn.set_config(transform_output="pandas")


def float_to_int(x):
    return x.astype(int)


st.title("Heart Disease Data Form")

# Инициализируем состояние сессии для хранения результата
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'show_form' not in st.session_state:
    st.session_state.show_form = True

def reset_form():
    st.session_state.prediction_made = False
    st.session_state.prediction_result = None
    st.session_state.show_form = True

# Показываем форму только если не было предсказания или нажата "New Predict"
if st.session_state.show_form:
    age = st.number_input("Age (years):", min_value=20, max_value=120, value=50)
    sex = st.selectbox("Sex:", ["M", "F"])
    
    chest_pain_type = st.selectbox(
        "Chest Pain Type:",
        ["TA", "ATA", "NAP", "ASY"]
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

    if st.button("Submit and Predict"):
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

        # Загружаем модель и делаем предсказание
        try:
            model = joblib.load("model/rf_pipe.pkl")
            prediction = model.predict(df)
            st.session_state.prediction_result = "Yes" if prediction[0] == 1 else "No"
            st.session_state.prediction_made = True
            st.session_state.show_form = False  # Скрываем форму после предсказания
            
        except Exception as e:
            st.error(f"Error loading model or making prediction: {e}")

# Показываем результат предсказания если он есть
if st.session_state.prediction_made:
    st.success(f"### Predicted Heart Disease: {st.session_state.prediction_result}")
    
    # Кнопка для нового предсказания
    if st.button("New Predict"):
        reset_form()
        st.rerun()  # Перезагружаем страницу для очистки формы
