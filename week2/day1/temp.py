import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Важная настройка для корректной настройки pipeline!
import sklearn

sklearn.set_config(transform_output="pandas")

# Pipeline
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
)
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
)


df = pd.read_csv("/home/dmitry/elbrus/Phase1/ds-phase1-git/week2/data/heart.csv")

num_features = df.select_dtypes(exclude="object")
cat_features = df.select_dtypes(include="object")

X, y = df.drop("HeartDisease", axis=1), df["HeartDisease"]


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

my_imputer = ColumnTransformer(
    transformers=[("num_imputer", SimpleImputer(strategy="mean"), ["Age"])],
    verbose_feature_names_out=False,
    remainder="passthrough",
)


ordinal_encoding_columns = list(cat_features.columns)

standart_scaler_columns = ["RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]


encode_and_scale = ColumnTransformer(
    [
        ("ordinal_enc", OrdinalEncoder(), ordinal_encoding_columns),
        ("scaling_num_columns", StandardScaler(), standart_scaler_columns),
    ],
    verbose_feature_names_out=False,
    remainder="passthrough",
)

preprocessor = Pipeline(
    [("imputer", my_imputer), ("scaler_and_encoder", encode_and_scale)]
)

dsas = preprocessor.fit_transform(X_train)
print(dsas.head(5))
