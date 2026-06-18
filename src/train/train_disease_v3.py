import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(BASE_DIR, "data", "dizziness_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model_v3.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "disease_columns_v3.pkl")

# ==============================
# LOAD DATA
# ==============================
print("Loading data...")

df = pd.read_csv(DATA_PATH)

print("Original Dataset Shape:", df.shape)

# ==============================
# CLEAN DATA
# ==============================
df = df.dropna()

# ==============================
# FEATURES & TARGET
# ==============================
X = df.drop("diseases", axis=1)
y = df["diseases"]

# ==============================
# REMOVE RARE CLASSES 🔥
# ==============================
print("\nFiltering rare classes...")

class_counts = y.value_counts()

# Keep diseases with at least 10 samples
valid_classes = class_counts[class_counts >= 10].index

df = df[df["diseases"].isin(valid_classes)]

print("Filtered Dataset Shape:", df.shape)
print("Number of classes:", len(valid_classes))

# Recreate X and y
X = df.drop("diseases", axis=1)
y = df["diseases"]

# ==============================
# OPTIONAL: REDUCE DATA SIZE (FAST TRAINING)
# ==============================
print("\nReducing dataset size for performance...")

df = df.sample(n=100000, random_state=42)

X = df.drop("diseases", axis=1)
y = df["diseases"]

print("Final Training Shape:", X.shape)

# ==============================
# SAVE COLUMNS (IMPORTANT)
# ==============================
joblib.dump(list(X.columns), COLUMNS_PATH)

# ==============================
# TRAIN TEST SPLIT
# ==============================
print("\nSplitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# MODEL (MEMORY OPTIMIZED)
# ==============================
print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=100,       # reduced
    max_depth=15,           # reduced
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1                # VERY IMPORTANT (no parallel memory issue)
)

model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================
print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Accuracy:", accuracy)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, MODEL_PATH)

print("\n🔥 Disease V3 model saved successfully!")

# ==============================
# SAMPLE TEST
# ==============================
print("\nRunning sample test...")

sample = X_test.iloc[0:1]

pred = model.predict(sample)[0]
probs = model.predict_proba(sample)[0]

top3_idx = np.argsort(probs)[-3:][::-1]
top3 = model.classes_[top3_idx]

print("\nSample Prediction:", pred)
print("Top 3 Predictions:", top3)