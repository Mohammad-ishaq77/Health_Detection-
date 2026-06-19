import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model_v3.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "disease_columns_v3.pkl")