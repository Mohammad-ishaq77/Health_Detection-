import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.pkl")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

# ==============================
# FEATURES & TARGET
# ==============================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ==============================
# PIPELINE
# ==============================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# ==============================
# SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# TRAIN
# ==============================
pipeline.fit(X_train, y_train)

# ==============================
# EVALUATE
# ==============================
y_pred = pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# SAVE
# ==============================
joblib.dump(pipeline, MODEL_PATH)

print("\n✅ Diabetes model saved!")