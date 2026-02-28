"""Check model type and try different SHAP approach"""
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

model_file = Path("trained_models/kidney_failure_advanced_v1.0.0.pkl")
bundle = joblib.load(model_file)

print("Model type:", bundle['model_type'])
print("Model class:", type(bundle['model']).__name__)

#Try getting SHAP values differently
model = bundle['model']
scaler = bundle['scaler']
feature_names = scaler.feature_names_in_

# Sample patient
patient_data = {fname: 0.0 for fname in feature_names}
patient_data['age'] = 68.0
patient_data['creatinine'] = 1.8
patient_data['systolic_bp'] = 140.0
patient_data['glucose'] = 120.0

X = pd.DataFrame([patient_data])[feature_names]
X_scaled = scaler.transform(X)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (from model)")
print("=" * 70)

# Get feature importance directly from the model
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 most important features:")
    for i in range(min(10, len(indices))):
        idx = indices[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
else:
    print("Model does not have feature_importances_")

# Try prediction
risk = model.predict_proba(X_scaled)[0][1]
print(f"\nKidney Failure Risk: {risk:.1%}")
