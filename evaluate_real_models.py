import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation_pipeline import EvaluationPipeline
from train_advanced_models import AdvancedFeatureEngineer

BASE_FEATURES = [
    "age",
    "gender",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "respiratory_rate",
    "wbc_count",
    "hemoglobin",
    "platelet_count",
    "creatinine",
    "bun",
    "glucose",
    "lactate",
]

DEFAULT_DISEASES = [
    "sepsis",
    "kidney_failure",
    "diabetes",
    "anemia",
    "thrombocytopenia",
    "hypertension",
    "mortality",
]


def resolve_eval_mask(df: pd.DataFrame) -> pd.Series:
    if "split" in df.columns:
        split = df["split"].astype(str).str.lower()
        if (split == "test").any():
            return split == "test"
        if (split == "val").any():
            return split == "val"
    return pd.Series([False] * len(df), index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained real disease models and generate plots")
    parser.add_argument("--data-path", default="mimic4_mini_training_data.csv", help="Path to labeled dataset CSV")
    parser.add_argument("--models-dir", default="trained_models", help="Directory containing trained model bundles")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory for graphs and tables")
    parser.add_argument("--bootstrap", type=int, default=500, help="Number of bootstrap samples for CIs")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    models_dir = Path(args.models_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    df = pd.read_csv(data_path)
    missing_base = [c for c in BASE_FEATURES if c not in df.columns]
    if missing_base:
        raise ValueError(f"Dataset missing required base features: {missing_base}")

    X_engineered = AdvancedFeatureEngineer.engineer_features(df[BASE_FEATURES])

    eval_mask = resolve_eval_mask(df)
    use_holdout = bool(eval_mask.any())

    pipeline = EvaluationPipeline(random_seed=42, n_bootstrap=args.bootstrap)

    evaluated = []
    for disease in DEFAULT_DISEASES:
        model_path = models_dir / f"{disease}_advanced_v1.0.0.pkl"
        if disease not in df.columns:
            print(f"[SKIP] Missing label column: {disease}")
            continue
        if not model_path.exists():
            print(f"[SKIP] Missing model file: {model_path}")
            continue

        y = df[disease].astype(int).values
        if np.unique(y).size < 2:
            print(f"[SKIP] Label has one class only: {disease}")
            continue

        bundle = joblib.load(model_path)
        model = bundle.get("model")
        scaler = bundle.get("scaler")
        model_type = str(bundle.get("model_type", "advanced"))

        if use_holdout:
            X_eval = X_engineered.loc[eval_mask]
            y_eval = y[eval_mask.values]
        else:
            _, X_eval, _, y_eval = train_test_split(
                X_engineered,
                y,
                test_size=0.2,
                stratify=y,
                random_state=42,
            )

        if scaler is not None:
            X_eval = scaler.transform(X_eval)
        else:
            X_eval = np.asarray(X_eval)

        pipeline.evaluate_model(
            model=model,
            X_test=X_eval,
            y_test=np.asarray(y_eval),
            disease_name=disease,
            model_type=model_type,
        )
        evaluated.append(disease)

    if not evaluated:
        raise RuntimeError("No diseases were evaluated. Check labels and model files.")

    pipeline.save_results(output_dir=args.output_dir)
    print(f"Evaluated diseases: {evaluated}")


if __name__ == "__main__":
    main()
