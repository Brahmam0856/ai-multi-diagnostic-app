# app.py
from flask import Flask, render_template, request, redirect, url_for, send_file, abort
import numpy as np
from models import train_and_save_models, load_models
import os
import io

app = Flask(__name__)

# Ensure models exist (train if necessary)
train_and_save_models()

# Load models & scaler & mean vector
heart_model, breast_model, breast_scaler, breast_mean = load_models()

def clinical_advice(probability):
    """
    Return a professional clinical-tone advice string based on probability percentage.
    Option A: Professional Clinical Tone
    Thresholds:
      0 - 30%   : low
      30 - 60%  : moderate
      60 - 100% : high / urgent
    """
    p = float(probability)
    if p < 30:
        return ("The model indicates a low probability of the condition. "
                "Continue routine monitoring and maintain a healthy lifestyle. "
                "If you have concerns or symptoms, consult a healthcare provider.")
    elif p < 60:
        return ("The model indicates a moderate probability of the condition. "
                "Clinical assessment and further diagnostic evaluation are recommended. "
                "Please consider scheduling an appointment with your physician.")
    else:
        return ("The model indicates a high probability of the condition. "
                "This may require prompt medical attention and diagnostic follow-up. "
                "Please consult a qualified healthcare professional as soon as possible.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/services")
def services():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    disease = request.form.get("disease")
    name = request.form.get("name", "Patient")
    result = "Unknown"
    probability = 0.0

    if disease == "heart":
        # parse heart inputs with safe defaults
        try:
            age = int(request.form.get("age", 55))
            chest_pain = int(request.form.get("chest_pain", 1))
            cholesterol = int(request.form.get("cholesterol", 220))
        except ValueError:
            return "Invalid heart inputs. Please enter valid numbers."

        features = np.array([[age, chest_pain, cholesterol]])
        pred = heart_model.predict(features)[0]
        prob = heart_model.predict_proba(features)[0][1]
        result = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"
        probability = round(float(prob) * 100, 2)

    elif disease == "breast":
        features_input = request.form.get("breast_features", "").strip()
        if features_input:
            try:
                arr = list(map(float, features_input.split(",")))
                features = np.array(arr).reshape(1, -1)
            except Exception:
                # Bad dimensionality or parse error -> fallback to mean
                features = breast_mean.reshape(1, -1)
        else:
            features = breast_mean.reshape(1, -1)

        # scale
        try:
            features_scaled = breast_scaler.transform(features)
        except Exception:
            # fallback to mean if user-provided dimensionality wrong
            features_scaled = breast_scaler.transform(breast_mean.reshape(1, -1))

        pred = breast_model.predict(features_scaled)[0]
        prob = breast_model.predict_proba(features_scaled)[0][1]
        result = "Breast Cancer Detected" if pred == 1 else "No Breast Cancer Detected"
        probability = round(float(prob) * 100, 2)

    else:
        return "Unknown disease selection."

    advice = clinical_advice(probability)

    # Render result page
    return render_template("result.html",
                           name=name,
                           result=result,
                           probability=probability,
                           advice=advice,
                           disease=disease)

@app.route("/download-report", methods=["POST"])
def download_report():
    """
    Placeholder endpoint for generating a downloadable report (PDF/CSV).
    Currently returns a small text-based file as a placeholder.
    You can replace this with a PDF generator (WeasyPrint/reportlab) in future.
    """
    name = request.form.get("name", "Patient")
    disease = request.form.get("disease", "unknown")
    result = request.form.get("result", "N/A")
    probability = request.form.get("probability", "0")
    advice = request.form.get("advice", "")

    # Create a simple text report as a placeholder
    content = f"Diagnostic Report\n\nPatient: {name}\nService: {disease}\nResult: {result}\nProbability: {probability}%\n\nAdvice:\n{advice}\n\nNote: This is a demo report."
    mem = io.BytesIO()
    mem.write(content.encode("utf-8"))
    mem.seek(0)

    return send_file(mem,
                     as_attachment=True,
                     download_name=f"diagnostic_report_{name.replace(' ','_')}.txt",
                     mimetype="text/plain")

if __name__ == "__main__":
    # Host 0.0.0.0 helps Windows accept localhost requests reliably
    app.run(debug=True, host="0.0.0.0", port=5000)
