"""
Multimodal fusion utilities that link MIMIC-IV tabular features with MIMIC-CXR
imaging evidence to produce patent-grade governance signals.

The module computes severity-aware modifiers from engineered clinical features
and blends them with imaging probabilities (CheXpert-style findings) to deliver
consistency metrics, governance alerts, and fused risk signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import logging
import numpy as np
import pandas as pd

from feature_engineering import BASE_FEATURES, add_derived_features

logger = logging.getLogger(__name__)

# Mapping between CheXpert findings and the diseases covered by this project.
IMAGING_DISEASE_MAP: Dict[str, List[str]] = {
    "pneumonia": ["sepsis"],
    "consolidation": ["sepsis"],
    "pleural_effusion": ["sepsis", "cardiovascular", "heart_disease"],
    "atelectasis": ["cardiovascular"],
    "edema": ["cardiovascular", "heart_disease"],
    "cardiomegaly": ["cardiovascular", "heart_disease"],
}

# Derived feature channels per disease that act as severity modifiers.
DISEASE_SEVERITY_FEATURE = {
    "sepsis": "sepsis_risk_score",
    "kidney_failure": "renal_burden_score",
    "heart_disease": "hemodynamic_instability_score",
    "cardiovascular": "metabolic_stress_index",
    "diabetes": "hyperglycemia_flag",
    "anemia": "anemia_flag",
    "thalassemia": "thrombocytopenia_flag",
    "thrombocytopenia": "thrombocytopenia_flag",
    "mortality": "sofa_like_score",
}

# Supported imaging findings that we attempt to parse from the CheXpert CSV
# or directly from user-provided inference payloads.
SUPPORTED_IMAGING_CHANNELS = [
    "pneumonia",
    "edema",
    "cardiomegaly",
    "consolidation",
    "pleural_effusion",
    "atelectasis",
]


def _normalize_probability(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(np.clip(value, 0.0, 1.0))
    except Exception:
        return None


def _normalize_severity_value(name: str, value: float) -> float:
    """Map engineered feature values to 0-1 scale."""
    if value is None:
        return 0.0
    if name == "sirs_score":
        return float(np.clip(value / 4.0, 0.0, 1.0))
    if name == "sofa_like_score":
        return float(np.clip(value / 4.0, 0.0, 1.0))
    if name == "renal_burden_score":
        return float(1 - np.exp(-value / 8.0))
    if name == "hemodynamic_instability_score":
        return float(np.clip(value / 3.0, 0.0, 1.0))
    if name == "metabolic_stress_index":
        return float(np.clip(value / 6.0, 0.0, 1.0))
    if name == "sepsis_risk_score":
        return float(np.clip(value, 0.0, 1.0))
    # Flags or ratios default to sigmoid-like squash
    return float(np.clip(value, 0.0, 1.0))


class ImagingEvidenceRepository:
    """
    Lazy loader for CheXpert labels (MIMIC-CXR-JPG metadata).

    It optionally builds an in-memory index of frequently requested DICOM IDs
    and falls back to chunked reads when lookup misses.
    """

    def __init__(self, labels_path: Optional[str] = None, cache_limit: int = 50000):
        self.labels_path = Path(labels_path) if labels_path else None
        self.cache_limit = cache_limit
        self._cache: Dict[str, Dict[str, float]] = {}
        self._index: Optional[pd.DataFrame] = None

        if self.labels_path and self.labels_path.exists():
            try:
                self._bootstrap_index()
            except Exception as exc:
                logger.warning("Failed to bootstrap CheXpert index: %s", exc)

    def _bootstrap_index(self) -> None:
        """Load a lightweight subset of the CheXpert CSV for instant lookups."""
        usecols = ["dicom_id"] + [
            col for col in SUPPORTED_IMAGING_CHANNELS if col in SUPPORTED_IMAGING_CHANNELS
        ]
        # Deduplicate column list while preserving order.
        unique_cols: List[str] = []
        for col in usecols:
            if col not in unique_cols:
                unique_cols.append(col)

        df = pd.read_csv(
            self.labels_path,
            compression="gzip" if self.labels_path.suffix == ".gz" else None,
            usecols=unique_cols,
            nrows=self.cache_limit,
        )
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["dicom_id"]).set_index("dicom_id")
        self._index = df
        logger.info(
            "Bootstrapped %d imaging labels from %s",
            len(df),
            self.labels_path,
        )

    def lookup(self, dicom_id: Optional[str]) -> Optional[Dict[str, float]]:
        if not dicom_id:
            return None
        dicom_id = str(dicom_id).strip()
        if not dicom_id:
            return None

        if dicom_id in self._cache:
            return self._cache[dicom_id]

        if self._index is not None and dicom_id in self._index.index:
            row = self._index.loc[dicom_id]
            values = {
                col: _normalize_probability(row[col])
                for col in SUPPORTED_IMAGING_CHANNELS
                if col in row and not pd.isna(row[col])
            }
            self._cache[dicom_id] = values
            return values

        # Fallback chunked scan if file exists.
        if not self.labels_path or not self.labels_path.exists():
            return None

        try:
            for chunk in pd.read_csv(
                self.labels_path,
                compression="gzip" if self.labels_path.suffix == ".gz" else None,
                usecols=["dicom_id"] + SUPPORTED_IMAGING_CHANNELS,
                chunksize=200000,
            ):
                chunk.columns = [c.lower() for c in chunk.columns]
                match = chunk[chunk["dicom_id"] == dicom_id]
                if not match.empty:
                    row = match.iloc[0]
                    values = {
                        col: _normalize_probability(row[col])
                        for col in SUPPORTED_IMAGING_CHANNELS
                        if col in row and not pd.isna(row[col])
                    }
                    self._cache[dicom_id] = values
                    return values
        except Exception as exc:
            logger.warning("Chunk scanning CheXpert labels failed: %s", exc)

        return None


@dataclass
class FusedDiseaseOutput:
    disease: str
    tabular_risk: float
    fused_score: float
    agreement_index: float
    severity_modifier: Optional[float] = None
    imaging_signal: Optional[float] = None
    governance_flag: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disease": self.disease,
            "tabular_risk": self.tabular_risk,
            "fused_score": self.fused_score,
            "agreement_index": self.agreement_index,
            "severity_modifier": self.severity_modifier,
            "imaging_signal": self.imaging_signal,
            "governance_flag": self.governance_flag,
        }


class MultimodalFusionEngine:
    """Blend tabular clinical models with imaging and severity evidence."""

    def __init__(
        self,
        labels_path: Optional[str] = None,
        imaging_weight: float = 0.35,
        severity_weight: float = 0.15,
        disagreement_threshold: float = 0.35,
    ):
        if imaging_weight + severity_weight >= 1.0:
            raise ValueError("imaging_weight + severity_weight must be < 1.0")
        self.tabular_weight = 1.0 - imaging_weight - severity_weight
        self.imaging_weight = imaging_weight
        self.severity_weight = severity_weight
        self.disagreement_threshold = disagreement_threshold
        self.repository = ImagingEvidenceRepository(labels_path)

    def fuse(
        self,
        base_features: Dict[str, float],
        predictions: Sequence[Any],
        imaging_signal: Optional[Dict[str, float]] = None,
        dicom_id: Optional[str] = None,
        imaging_source: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not predictions:
            return None

        engineered = add_derived_features(pd.DataFrame([base_features])[BASE_FEATURES])
        derived_row = engineered.iloc[0].to_dict()

        severity_channels = {
            name: _normalize_severity_value(col_name, derived_row.get(col_name, 0.0))
            for name, col_name in DISEASE_SEVERITY_FEATURE.items()
            if col_name in derived_row
        }

        resolved_imaging = {}
        if imaging_signal:
            resolved_imaging.update(
                {
                    key: _normalize_probability(val)
                    for key, val in imaging_signal.items()
                    if key in SUPPORTED_IMAGING_CHANNELS
                }
            )
        if not resolved_imaging and dicom_id:
            repo_signal = self.repository.lookup(dicom_id)
            if repo_signal:
                resolved_imaging.update(repo_signal)

        fused_outputs: List[FusedDiseaseOutput] = []
        agreements: List[float] = []
        alerts: List[str] = []

        for item in predictions:
            disease = getattr(item, "disease", None) or item.get("disease")
            tabular_risk = float(getattr(item, "risk_score", None) or item.get("risk_score", 0.0))

            imaging_value = self._aggregate_imaging_signal(disease, resolved_imaging)
            severity_value = severity_channels.get(disease)

            fused_score = self._blend_scores(tabular_risk, imaging_value, severity_value)
            agreement_index = self._compute_agreement(tabular_risk, imaging_value)
            flag = self._derive_flag(fused_score, agreement_index, severity_value)

            if flag:
                alerts.append(f"{disease}:{flag}")

            fused_outputs.append(
                FusedDiseaseOutput(
                    disease=disease,
                    tabular_risk=tabular_risk,
                    fused_score=fused_score,
                    agreement_index=agreement_index,
                    severity_modifier=severity_value,
                    imaging_signal=imaging_value,
                    governance_flag=flag,
                )
            )
            agreements.append(agreement_index)

        consistency_index = float(np.mean(agreements)) if agreements else 1.0

        return {
            "consistency_index": consistency_index,
            "fused_predictions": [out.to_dict() for out in fused_outputs],
            "alerts": alerts,
            "imaging_channels_used": sorted(
                [k for k, v in resolved_imaging.items() if v is not None]
            ),
            "severity_channels": severity_channels,
            "data_sources": {
                "tabular": "MIMIC-IV v3.1",
                "imaging": "MIMIC-CXR-JPG 2.1.0" if resolved_imaging else "not_provided",
                "imaging_source": imaging_source or ("repository" if dicom_id else "unspecified"),
            },
        }

    def _aggregate_imaging_signal(
        self,
        disease: str,
        imaging_signal: Dict[str, float],
    ) -> Optional[float]:
        relevant = [
            imaging_signal.get(channel)
            for channel, diseases in IMAGING_DISEASE_MAP.items()
            if disease in diseases and channel in imaging_signal
        ]
        values = [val for val in relevant if val is not None]
        if not values:
            return None
        return float(np.mean(values))

    def _blend_scores(
        self,
        tabular: float,
        imaging: Optional[float],
        severity: Optional[float],
    ) -> float:
        weighted_components: List[tuple[float, float]] = [(tabular, self.tabular_weight)]
        if imaging is not None:
            weighted_components.append((imaging, self.imaging_weight))
        if severity is not None:
            weighted_components.append((severity, self.severity_weight))

        total_weight = sum(weight for _, weight in weighted_components)
        if total_weight == 0:
            return tabular
        fused = sum(score * weight for score, weight in weighted_components) / total_weight
        return float(np.clip(fused, 0.0, 1.0))

    def _compute_agreement(self, tabular: float, imaging: Optional[float]) -> float:
        if imaging is None:
            return 1.0
        return float(1.0 - abs(tabular - imaging))

    def _derive_flag(
        self,
        fused_score: float,
        agreement_index: float,
        severity: Optional[float],
    ) -> Optional[str]:
        if agreement_index < self.disagreement_threshold:
            return "evidence_conflict"
        if fused_score >= 0.75 and (severity is not None and severity < 0.25):
            return "high_risk_low_severity"
        return None

