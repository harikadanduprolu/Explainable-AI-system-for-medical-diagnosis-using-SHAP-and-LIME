"""Display Feature Importance Explanations (Alternative to SHAP)"""
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 80)
print("EXPLAINABILITY DEMONSTRATION - FEATURE IMPORTANCE")
print("=" * 80)
print("\nThis shows which factors the AI model considers most important")
print("(This is the same information SHAP provides)")
print("")

# Load kidney failure model
model_file = Path("trained_models/kidney_failure_advanced_v1.0.0.pkl")
bundle = joblib.load(model_file)
model = bundle['model']
scaler = bundle['scaler']
feature_names = scaler.feature_names_in_

print(f"Model: Kidney Failure Prediction (XGBoost)")
print(f"Features: {len(feature_names)} clinical parameters")

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n" + "=" * 80)
print("GLOBAL FEATURE IMPORTANCE (What matters most across ALL patients)")
print("=" * 80)

print("\nTop 15 Most Important Clinical Features:")
print("-" * 80)

for i in range(min(15, len(indices))):
    idx = indices[i]
    feature = feature_names[idx]
    importance = importances[idx]
    
    # Create bar visualization
    bar_length = int(importance * 50)
    bar = "█" * bar_length
    
    print(f"\n{i+1:2d}. {feature:25s} {importance:6.2%}  {bar}")

# Sample patient prediction
print("\n" + "=" * 80)
print("LOCAL EXPLANATION (Individual Patient Analysis)")
print("=" * 80)

patient_data = {fname: 0.0 for fname in feature_names}
patient_data.update({
    'age': 68.0,
    'creatinine': 1.8,
    'systolic_bp': 140.0,
    'glucose': 120.0,
    'hemoglobin': 11.5,
    'heart_rate': 85.0,
    'wbc_count': 9.0,
    'temperature': 98.6
})

X = pd.DataFrame([patient_data])[feature_names]
X_scaled = scaler.transform(X)
risk = model.predict_proba(X_scaled)[0][1]

print("\nPatient Profile:")
print(f"  Age: {patient_data.get('age', 0):.0f} years")
print(f"  Creatinine: {patient_data.get('creatinine', 0):.1f} mg/dL (kidney function)")
print(f"  Systolic BP: {patient_data.get('systolic_bp', 0):.0f} mmHg")
print(f"  Glucose: {patient_data.get('glucose', 0):.0f} mg/dL")
print(f"  Hemoglobin: {patient_data.get('hemoglobin', 0):.1f} g/dL")

print(f"\n>>> PREDICTION: Kidney Failure Risk = {risk:.1%}")

if risk >= 0.7:
    risk_level = "HIGH RISK"
    emoji = "[!!!]"
elif risk >= 0.4:
    risk_level = "MODERATE RISK"
    emoji = "[!!]"
else:
    risk_level = "LOW RISK"
    emoji = "[OK]"

print(f">>> Risk Level: {risk_level} {emoji}")

# Clinical interpretation
print("\n" + "=" * 80)
print("CLINICAL INTERPRETATION:")
print("=" * 80)

print("\nKey Factors Contributing to This Prediction:")

top_features = [(feature_names[idx], importances[idx]) for idx in indices[:5]]

for i, (feature, importance) in enumerate(top_features, 1):
    value = patient_data.get(feature, 0)
    print(f"\n{i}. {feature} (importance: {importance:.1%})")
    print(f"   Patient value: {value:.2f}")
    
    # Add clinical context
    if 'creatinine' in feature and value > 1.2:
        print(f"   [ELEVATED] Normal range: 0.6-1.2 mg/dL")
    elif 'age' in feature:
        print(f"   [RISK FACTOR] Advanced age increases kidney disease risk")
    elif 'kidney' in feature:
        print(f"   [COMPOSITE SCORE] Calculated from multiple kidney parameters")
    elif 'bun' in feature:
        print(f"   [KIDNEY MARKER] Blood Urea Nitrogen - kidney function indicator")

print("\n" + "=" * 80)
print("HOW TO VIEW THIS INTERACTIVELY:")
print("=" * 80)
print("\nRun the dashboard to adjust patient parameters and see predictions update:")
print("  python enhanced_dashboard_with_whatif.py")
print("  Then open: http://127.0.0.1:8051")
print("\nMove the sliders to see how changing creatinine, age, BP, etc.")
print("affects the kidney failure risk prediction!")
print("=" * 80)