"""Test SHAP explanations display"""
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

print("=" * 70)
print("SHAP EXPLANATIONS TEST")
print("=" * 70)

# Load a model
model_file = Path("trained_models/kidney_failure_advanced_v1.0.0.pkl")
print(f"\nLoading model: {model_file.name}")

bundle = joblib.load(model_file)
model = bundle['model']
scaler = bundle['scaler']
feature_names = scaler.feature_names_in_

print(f"[OK] Model loaded successfully")
print(f"Feature names: {feature_names[:5]}... (showing first 5)")

# Create a sample patient
print("\n" + "-" * 70)
print("SAMPLE PATIENT DATA:")
print("-" * 70)

patient_data = {
    'age': 68.0,
    'creatinine': 1.8,
    'systolic_bp': 140.0,
    'glucose': 120.0,
    'hemoglobin': 11.5,
    'gender': 0,
    'heart_rate': 85.0,
    'diastolic_bp': 80.0,
    'temperature': 98.6,
    'wbc_count': 9.0,
    'platelet_count': 200.0,
    'respiratory_rate': 16.0,
    'bun': 20.0,
    'lactate': 1.5
}

# Fill any missing features with defaults
for fname in feature_names:
    if fname not in patient_data:
        patient_data[fname] = 0

print(f"Age: {patient_data['age']} years")
print(f"Creatinine: {patient_data['creatinine']} mg/dL")
print(f"Systolic BP: {patient_data['systolic_bp']} mmHg")
print(f"Glucose: {patient_data['glucose']} mg/dL")

# Prepare and predict
X = pd.DataFrame([patient_data])[feature_names]
X_scaled = scaler.transform(X)
risk_prob = model.predict_proba(X_scaled)[0][1]

print(f"\n" + "=" * 70)
print(f"PREDICTION: Kidney Failure Risk = {risk_prob:.1%}")
print("=" * 70)

# Compute SHAP values
print("\nComputing SHAP explanations...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

# Handle different output formats
if isinstance(shap_values, list):
    shap_vals = shap_values[1][0]  # Positive class
else:
    shap_vals = shap_values[0]

# Get top 5 features
top_indices = np.argsort(np.abs(shap_vals))[-5:][::-1]

print("\n" + "=" * 70)
print("SHAP FEATURE IMPORTANCE (Top 5 Contributors)")
print("=" * 70)

for i, idx in enumerate(top_indices, 1):
    feature = feature_names[idx]
    value = patient_data[feature]
    impact = shap_vals[idx]
    direction = "INCREASES" if impact > 0 else "DECREASES"
    
    print(f"\n{i}. {feature}")
    print(f"   Value: {value:.2f}")
    print(f"   SHAP Impact: {impact:+.3f}")
    print(f"   Direction: {direction} risk")

print("\n" + "=" * 70)
print("CLINICAL INTERPRETATION:")
print("=" * 70)

top_feature = feature_names[top_indices[0]]
top_impact = shap_vals[top_indices[0]]

if abs(top_impact) > 0.1:
    print(f"\nThe most important factor is '{top_feature}'")
    print(f"which {'STRONGLY INCREASES' if top_impact > 0 else 'STRONGLY DECREASES'} the risk")
    print(f"by a SHAP value of {abs(top_impact):.3f}")
else:
    print(f"\nNo single factor strongly dominates the prediction.")

print("\n" + "=" * 70)
print("To see this interactive in the dashboard:")
print("  python enhanced_dashboard_with_whatif.py")
print("  Then open: http://127.0.0.1:8051")
print("=" * 70)
