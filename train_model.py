import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load data
train_df = pd.read_csv('kaggle_data/train.csv')
X = train_df.drop('exam_score', axis=1)
y = train_df['exam_score']

# Feature engineering
X['sleep_dev'] = abs(X['sleep_hours'] - 7)
X['study_effort'] = X['study_hours'] * X['class_attendance']
X['study_hours_sq'] = X['study_hours'] ** 2

# Preprocessor
numerical_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()
categorical_cols = [col for col in categorical_cols if X[col].nunique() < 10]

numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Process
X_processed = preprocessor.fit_transform(X)

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=0, n_jobs=-1)
model.fit(X_processed, y)

# Save preprocessor & model
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(model, 'model.pkl')
print("Training complete! Preprocessor and model saved.")
