"""Check model bundle contents"""
import joblib
from pathlib import Path

model_file = Path("trained_models/kidney_failure_advanced_v1.0.0.pkl")
bundle = joblib.load(model_file)

print("Model bundle keys:")
for key in bundle.keys():
    print(f"  - {key}")
    if key == 'scaler' and hasattr(bundle[key], 'feature_names_in_'):
        print(f"    Scaler feature names: {bundle[key].feature_names_in_[:5]}...")
