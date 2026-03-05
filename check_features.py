import joblib
from pathlib import Path

models_dir = Path('trained_models')
for f in sorted(models_dir.glob('*_xgboost_v*.pkl')):
    bundle = joblib.load(f)
    features = bundle['feature_names']
    print(f'{f.stem:30s} Features: {len(features)}')
    if len(features) > 14:
        print(f'  Extra features: {[f for f in features if f not in ["age", "gender", "heart_rate", "systolic_bp", "diastolic_bp", "temperature", "respiratory_rate", "wbc_count", "hemoglobin", "platelet_count", "creatinine", "bun", "glucose", "lactate"]]}')
