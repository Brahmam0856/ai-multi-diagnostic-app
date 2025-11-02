# models.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from joblib import dump, load

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_models(force_retrain=False):
    """
    Train heart and breast models. Saves:
      - saved_models/heart_model.joblib
      - saved_models/breast_model.joblib
      - saved_models/breast_scaler.joblib
      - saved_models/breast_features.npy
    If models exist and force_retrain=False, returns immediately.
    """
    heart_path = os.path.join(MODEL_DIR, "heart_model.joblib")
    breast_path = os.path.join(MODEL_DIR, "breast_model.joblib")
    breast_scaler_path = os.path.join(MODEL_DIR, "breast_scaler.joblib")
    breast_mean_path = os.path.join(MODEL_DIR, "breast_features.npy")

    if not force_retrain and os.path.exists(heart_path) and os.path.exists(breast_path):
        print("Models already exist. Set force_retrain=True to retrain.")
        return

    # -------------------------
    # Heart disease model (synthetic improved dataset)
    # -------------------------
    print("Training Heart Disease Model...")
    rng = np.random.RandomState(42)
    n = 400
    ages = rng.randint(30, 85, size=n)
    chest_pain = rng.choice([0, 1, 2, 3], size=n)
    cholesterol = rng.randint(150, 350, size=n)
    # Create a logistic-like scoring for synthetic target
    logits = (ages - 45) * 0.035 + (chest_pain) * 0.45 + (cholesterol - 210) * 0.012
    probs = 1 / (1 + np.exp(-logits))
    target = (probs > 0.5).astype(int)

    heart_df = pd.DataFrame({
        "age": ages,
        "chest_pain": chest_pain,
        "cholesterol": cholesterol,
        "target": target
    })

    Xh = heart_df[["age", "chest_pain", "cholesterol"]]
    yh = heart_df["target"]
    heart_model = RandomForestClassifier(n_estimators=250, random_state=42)
    heart_model.fit(Xh, yh)
    dump(heart_model, heart_path)
    print(f"Saved Heart Disease Model -> {heart_path}")

    # -------------------------
    # Breast cancer model (sklearn dataset with scaler)
    # -------------------------
    print("Training Breast Cancer Model...")
    breast = load_breast_cancer()
    X = breast.data
    y = breast.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    breast_model = RandomForestClassifier(n_estimators=250, random_state=42)
    breast_model.fit(X_train_scaled, y_train)

    dump(breast_model, breast_path)
    dump(scaler, breast_scaler_path)
    np.save(breast_mean_path, np.mean(X, axis=0))
    print(f"Saved Breast Cancer Model -> {breast_path}")
    print(f"Saved Breast Scaler -> {breast_scaler_path}")
    print(f"Saved Breast Mean Features -> {breast_mean_path}")

def load_models():
    """Load models and scaler. Returns (heart_model, breast_model, breast_scaler, breast_mean)"""
    heart_model = load(os.path.join(MODEL_DIR, "heart_model.joblib"))
    breast_model = load(os.path.join(MODEL_DIR, "breast_model.joblib"))
    scaler = load(os.path.join(MODEL_DIR, "breast_scaler.joblib"))
    breast_mean = np.load(os.path.join(MODEL_DIR, "breast_features.npy"))
    return heart_model, breast_model, scaler, breast_mean
