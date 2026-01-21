# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import mutual_info_regression

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# Imports
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# Feature Engineering
for df in [test_df, train_df]:
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    df['fare_per_person'] = df['Fare']/df['family_size']
    df['has_cabin'] = df['Cabin'].notnull().astype(int)
    ticket_counts = df['Ticket'].value_counts() 
    df['ticket_group_size'] = df['Ticket'].map(ticket_counts)

# Features
X = train_df.drop(['Survived', 'PassengerId'], axis=1)
y = train_df['Survived']
x_test_final = test_df.copy()

# Columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = [col for col in X.columns if X[col].dtype in ['float64', 'int64']]
low_cardinality_cols = [col for col in categorical_cols if X[col].nunique() < 10]

# Pipeline
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, low_cardinality_cols)])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42
    ))
])

score = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

pipeline.fit(X, y)
test_preds = pipeline.predict(x_test_final)
print(score)

submission = gender_submission.copy()
submission['Survived'] = test_preds
submission.to_csv('submission.csv', index=False)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
