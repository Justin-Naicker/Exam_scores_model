# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

# Load Data
train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s6e1/sample_submission.csv')

# Feature Engineering
def feature_engineering(df):
    df = df.copy()
    df['sleep_dev'] = abs(df['sleep_hours'] - 7)
    df['study_effort'] = df['study_hours'] * df['class_attendance']
    df['study_hours_sq'] = df['study_hours'] ** 2
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Features
X = train_df.drop(['exam_score'], axis=1)
y = train_df['exam_score']
X_test_final = test_df.copy()

# Columns
object_cols = X.select_dtypes(include='object').columns.tolist()
low_cardinality_columns = [col for col in object_cols if X[col].nunique() < 10]
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Transformers / Pipeline
numerical_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, low_cardinality_columns)])

# Split
train_x, valid_x, train_y, valid_y = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=0)

X_train = preprocessor.fit_transform(train_x)
X_valid = preprocessor.transform(valid_x)

# Model
model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    random_state=0,
    n_jobs=-1,
    early_stopping_rounds=50
)

# Validation Score
model.fit(X_train, train_y, eval_set=[(X_valid, valid_y)],verbose=False)

valid_predictions = model.predict(X_valid)
mae = mean_absolute_error(valid_y, valid_predictions)
print(f"Validation MAE: {mae:.3f}")

# Final Model

final_model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    random_state=0,
    n_jobs=-1
)

# Full Score
X_full = preprocessor.fit_transform(X)
X_test = preprocessor.transform(X_test_final)

final_model.fit(X_full, y)
test_preds = final_model.predict(X_test)

# Submission
submission = sample_submission.copy()
submission['exam_score'] = test_preds
submission.to_csv('submission.csv', index=False)

print("submission.csv created successfully!")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
