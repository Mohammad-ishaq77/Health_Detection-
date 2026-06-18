import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from xgboost import XGBClassifier

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(BASE_DIR, "data", "heart_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_model.pkl")

# ==============================
# LOAD DATA
# ==============================
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# ==============================
# FEATURE ENGINEERING 🔥
# ==============================
df["Age_HR"] = df["Age"] * df["MaxHR"]
df["BP_Chol"] = df["RestingBP"] * df["Cholesterol"]
df["Oldpeak_ST"] = df["Oldpeak"] * (df["ST_Slope"] == "Flat").astype(int)

# ==============================
# FEATURES & TARGET
# ==============================
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# ==============================
# COLUMN TYPES
# ==============================
categorical_cols = [
    "Sex", "ChestPainType", "RestingECG",
    "ExerciseAngina", "ST_Slope"
]

numerical_cols = [col for col in X.columns if col not in categorical_cols]

# ==============================
# PREPROCESSING PIPELINE
# ==============================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# ==============================
# MODEL (TUNED XGBOOST 🔥)
# ==============================
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

# ==============================
# FULL PIPELINE
# ==============================
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])

# ==============================
# TRAIN TEST SPLIT (STRATIFIED)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# TRAIN MODEL
# ==============================
pipeline.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n===== TEST RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# CROSS VALIDATION (REAL METRIC)
# ==============================
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')

print("\n===== CROSS VALIDATION =====")
print("CV ROC-AUC Scores:", cv_scores)
print("Mean CV ROC-AUC:", cv_scores.mean())

# ==============================
# SAVE MODEL (IMPORTANT)
# ==============================
joblib.dump(pipeline, MODEL_PATH)

print("\n✅ Heart model saved successfully!")

# ==============================
# SAMPLE TEST (OPTIONAL)
# ==============================
sample = pd.DataFrame([{
    "Age": 60,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 160,
    "Cholesterol": 300,
    "FastingBS": 1,
    "RestingECG": "ST",
    "MaxHR": 110,
    "ExerciseAngina": "Y",
    "Oldpeak": 3.5,
    "ST_Slope": "Flat"
}])

# Apply feature engineering
sample["Age_HR"] = sample["Age"] * sample["MaxHR"]
sample["BP_Chol"] = sample["RestingBP"] * sample["Cholesterol"]
sample["Oldpeak_ST"] = sample["Oldpeak"] * (sample["ST_Slope"] == "Flat").astype(int)

pred = pipeline.predict(sample)[0]
prob = pipeline.predict_proba(sample)[0][1]

print("\n===== SAMPLE PREDICTION =====")
print("Prediction:", pred)
print("Probability:", prob)