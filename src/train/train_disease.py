import pandas as pd
import numpy as np
import pickle
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ==============================
# 0. PATH SETUP (VERY IMPORTANT)
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(BASE_DIR, "data", "skin-disease.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.json")

# ==============================
# 1. LOAD DATA
# ==============================
print("Loading data from:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
print("Dataset Shape:", df.shape)

# ==============================
# 2. CLEAN DATA
# ==============================
df = df.drop(columns=["Unnamed: 133"], errors='ignore')

print("Missing values:", df.isnull().sum().sum())

# ==============================
# 3. SPLIT FEATURES & TARGET
# ==============================
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Save feature names
columns = list(X.columns)
with open(COLUMNS_PATH, "w") as f:
    json.dump(columns, f)

# ==============================
# 4. LABEL ENCODING
# ==============================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)

# ==============================
# 5. ADD NOISE
# ==============================
X_noisy = X.copy()

noise = np.random.binomial(1, 0.05, X_noisy.shape)
X_noisy = np.logical_xor(X_noisy, noise).astype(int)

# ==============================
# 6. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y_encoded, test_size=0.2, random_state=42
)

# ==============================
# 7. MODEL TRAINING
# ==============================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 8. EVALUATION
# ==============================
y_pred = model.predict(X_test)

# Decode labels
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

print("\n🔍 Classification Report (Real Labels):\n")
print(classification_report(y_test_labels, y_pred_labels))

accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Test Accuracy:", accuracy)

train_acc = model.score(X_train, y_train)
print("🔥 Train Accuracy:", train_acc)

# ==============================
# 9. SAVE MODEL
# ==============================
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model, encoder, and columns saved successfully!")