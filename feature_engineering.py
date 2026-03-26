"""
Shared clinical feature engineering utilities.
Transforms the 14 raw features (age, gender, vitals, labs) into a richer set
of derived ratios, interaction terms, severity scores, and clinical flags so
that training, inference (FastAPI), and data pipelines stay in sync.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

BASE_FEATURES: List[str] = [
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

_Z_SCORE_PARAMS = {
    "heart_rate_z": ("heart_rate", 80, 15),
    "systolic_bp_z": ("systolic_bp", 120, 20),
    "diastolic_bp_z": ("diastolic_bp", 75, 12),
    "temperature_z": ("temperature", 98.6, 1.0),
    "respiratory_rate_z": ("respiratory_rate", 18, 4),
}

_FLAG_RULES = {
    "anemia_flag": lambda df: (df["hemoglobin"] < 11).astype(int),
    "thrombocytopenia_flag": lambda df: (df["platelet_count"] < 150).astype(int),
    "hyperlactatemia_flag": lambda df: (df["lactate"] > 2.0).astype(int),
    "hyperglycemia_flag": lambda df: (df["glucose"] > 180).astype(int),
    "hypotension_flag": lambda df: (df["systolic_bp"] < 100).astype(int),
    "tachycardia_flag": lambda df: (df["heart_rate"] > 100).astype(int),
    "tachypnea_flag": lambda df: (df["respiratory_rate"] > 22).astype(int),
    "fever_flag": lambda df: (df["temperature"] > 100.4).astype(int),
    "hypothermia_flag": lambda df: (df["temperature"] < 96.8).astype(int),
    "infection_marker_flag": lambda df: (
        (df["wbc_count"] > 12) | (df["wbc_count"] < 4)
    ).astype(int),
    "kidney_failure_flag": lambda df: (
        (df["creatinine"] > 2.0) | (df["bun"] > 40)
    ).astype(int),
}


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with ~30 engineered features appended."""
    data = df.copy()

    derived = {}

    # Hemodynamic ratios
    derived["shock_index"] = _safe_div(data["heart_rate"], data["systolic_bp"])
    derived["pulse_pressure"] = data["systolic_bp"] - data["diastolic_bp"]
    derived["mean_arterial_pressure"] = (
        data["systolic_bp"] + 2 * data["diastolic_bp"]
    ) / 3.0
    derived["map_deviation"] = 70 - derived["mean_arterial_pressure"]

    # Kidney / hematologic markers
    derived["creatinine_bun_ratio"] = _safe_div(data["creatinine"], data["bun"])
    egfr = (140 - data["age"].clip(1, None)) / (
        data["creatinine"].replace(0, np.nan) * 72
    )
    egfr = egfr * np.where(data["gender"] == 0, 0.85, 1.0)
    derived["egfr_proxy"] = egfr
    derived["hemoglobin_wbc_ratio"] = _safe_div(data["hemoglobin"], data["wbc_count"])
    derived["platelet_wbc_ratio"] = _safe_div(
        data["platelet_count"], data["wbc_count"]
    )

    derived["renal_burden_score"] = data["creatinine"] + (data["bun"] / 25.0)

    # Metabolic / perfusion markers
    derived["age_glucose_product"] = data["age"] * data["glucose"]
    derived["lactate_heart_rate_ratio"] = _safe_div(
        data["lactate"], data["heart_rate"]
    )
    derived["lactate_glucose_ratio"] = _safe_div(data["lactate"], data["glucose"])
    derived["oxygen_delivery_proxy"] = data["hemoglobin"] * data["systolic_bp"]

    # Severity scores
    temp_flag = ((data["temperature"] > 100.4) | (data["temperature"] < 96.8)).astype(
        int
    )
    hr_flag = (data["heart_rate"] > 90).astype(int)
    resp_flag = (data["respiratory_rate"] > 20).astype(int)
    wbc_flag = ((data["wbc_count"] > 12) | (data["wbc_count"] < 4)).astype(int)
    derived["sirs_score"] = temp_flag + hr_flag + resp_flag + wbc_flag

    sofa_flags = (
        (data["systolic_bp"] < 90).astype(int)
        + (data["creatinine"] > 2.0).astype(int)
        + (data["platelet_count"] < 100).astype(int)
        + (data["lactate"] > 2.0).astype(int)
    )
    derived["sofa_like_score"] = sofa_flags

    sepsis_score = (
        0.3 * temp_flag
        + 0.25 * hr_flag
        + 0.2 * resp_flag
        + 0.25 * wbc_flag
        + 0.2 * (data["lactate"] / 4).clip(0, 1)
    )
    derived["sepsis_risk_score"] = sepsis_score.clip(0, 1)

    hemodynamic_instability = (
        derived["shock_index"].fillna(0)
        + (data["systolic_bp"] < 100).astype(int)
        + (derived["pulse_pressure"] < 30).astype(int)
    )
    derived["hemodynamic_instability_score"] = hemodynamic_instability

    metabolic_stress = (
        (data["lactate"] / 2).clip(0, 3)
        + (data["glucose"] / 200).clip(0, 3)
        + ((data["temperature"] - 98.6) / 2).clip(-2, 2)
    )
    derived["metabolic_stress_index"] = metabolic_stress

    derived["shock_lactate_interaction"] = (
        derived["shock_index"].fillna(0) * data["lactate"]
    )

    # Z-scores
    for name, (feature, mean_val, std_val) in _Z_SCORE_PARAMS.items():
        derived[name] = (data[feature] - mean_val) / std_val

    # Logs / transforms
    derived["wbc_log"] = np.log1p(data["wbc_count"])
    derived["glucose_log"] = np.log1p(data["glucose"])
    derived["bun_log"] = np.log1p(data["bun"])

    # Flags
    for name, rule in _FLAG_RULES.items():
        derived[name] = rule(data)

    # Cleanup
    derived_df = pd.DataFrame(derived, index=data.index)
    derived_df = derived_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    combined = pd.concat([data, derived_df], axis=1)
    return combined


def get_all_feature_columns(df: pd.DataFrame | None = None) -> List[str]:
    """Return ordered list of usable feature columns."""
    engineered = add_derived_features(pd.DataFrame({c: [0] for c in BASE_FEATURES}))
    cols = [c for c in engineered.columns if c not in {"HADM_ID", "ICUSTAY_ID"}]
    if df is not None:
        cols = [c for c in cols if c in df.columns]
    return cols
