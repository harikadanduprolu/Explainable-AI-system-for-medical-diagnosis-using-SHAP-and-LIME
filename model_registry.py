"""
Model & Explanation Versioning System
=====================================

Purpose
-------
- Register each trained model with rich metadata (provenance, data lineage,
  hyperparameters, metrics, governance references).
- Prevent explanation–model mismatches by enforcing compatibility checks
  between model_version and explainer_version.
- Track training data lineage (data version, data hash, feature schema hash).
- Support rollback to prior versions and audit queries over historical models.

Clinical/Regulatory Rationale
-----------------------------
- FDA SaMD/MLDS: Requires traceability of algorithm versions, training data
  provenance, and performance claims tied to specific model builds.
- EU AI Act (high-risk): Demands technical documentation, version control,
  and the ability to reconstruct model changes over lifecycle.
- Patent (IN/US/EU): Provides evidence of technical effect and controlled
  evolution of the invention via documented versions and compatibility rules.

Version Compatibility Rules
---------------------------
- An explanation must reference the exact model_version it was built for.
- The explainer_feature_hash must match the model_feature_hash; otherwise
  raise VersionCompatibilityError.
- Training data lineage (data_version, data_hash) must be present to register.
- Rollback is allowed only to a registered version; registry keeps immutable
  records (append-only) and marks current pointer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List


class VersionCompatibilityError(RuntimeError):
    """Raised when model and explanation versions are incompatible."""


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    disease: str
    model_name: str
    model_version: str
    created_at: str
    training_data_version: str
    training_data_hash: str
    feature_schema_hash: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    code_hash: str  # hash of training code/commit
    artifact_path: str
    governance_refs: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExplanationMetadata:
    explainer_id: str
    model_id: str
    model_version: str
    explainer_version: str
    created_at: str
    method: str  # e.g., shap_tree, shap_kernel, lime_tabular, gradcam
    feature_schema_hash: str
    parameters: Dict[str, Any]
    artifact_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """In-memory registry for models and explainers with compatibility checks."""

    def __init__(self):
        self.models: Dict[str, ModelMetadata] = {}
        self.explainers: Dict[str, ExplanationMetadata] = {}
        self.current_model_by_disease: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Model registration and lookup
    # ------------------------------------------------------------------

    def register_model(
        self,
        disease: str,
        model_name: str,
        model_version: str,
        training_data_version: str,
        training_data_hash: str,
        feature_schema_hash: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        code_hash: str,
        artifact_path: str,
        governance_refs: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> ModelMetadata:
        model_id = f"model-{uuid.uuid4()}"
        metadata = ModelMetadata(
            model_id=model_id,
            disease=disease,
            model_name=model_name,
            model_version=model_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            training_data_version=training_data_version,
            training_data_hash=training_data_hash,
            feature_schema_hash=feature_schema_hash,
            hyperparameters=hyperparameters,
            metrics=metrics,
            code_hash=code_hash,
            artifact_path=artifact_path,
            governance_refs=governance_refs or {},
            notes=notes,
        )
        self.models[model_id] = metadata
        self.current_model_by_disease[disease] = model_id
        return metadata

    def get_model(self, model_id: str) -> ModelMetadata:
        return self.models[model_id]

    def get_current_model_for_disease(self, disease: str) -> Optional[ModelMetadata]:
        model_id = self.current_model_by_disease.get(disease)
        return self.models.get(model_id) if model_id else None

    # ------------------------------------------------------------------
    # Explanation registration and compatibility
    # ------------------------------------------------------------------

    def register_explainer(
        self,
        model_id: str,
        model_version: str,
        explainer_version: str,
        method: str,
        feature_schema_hash: str,
        parameters: Dict[str, Any],
        artifact_path: Optional[str] = None,
    ) -> ExplanationMetadata:
        model_meta = self.get_model(model_id)
        self._assert_feature_schema_match(model_meta, feature_schema_hash)
        self._assert_model_version_match(model_meta, model_version)

        explainer_id = f"expl-{uuid.uuid4()}"
        metadata = ExplanationMetadata(
            explainer_id=explainer_id,
            model_id=model_id,
            model_version=model_version,
            explainer_version=explainer_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            method=method,
            feature_schema_hash=feature_schema_hash,
            parameters=parameters,
            artifact_path=artifact_path,
        )
        self.explainers[explainer_id] = metadata
        return metadata

    def get_explainer(self, explainer_id: str) -> ExplanationMetadata:
        return self.explainers[explainer_id]

    def assert_compatible(self, model_id: str, explainer_id: str) -> None:
        model_meta = self.get_model(model_id)
        expl_meta = self.get_explainer(explainer_id)
        self._assert_model_version_match(model_meta, expl_meta.model_version)
        self._assert_feature_schema_match(model_meta, expl_meta.feature_schema_hash)

    # ------------------------------------------------------------------
    # Rollback and audit queries
    # ------------------------------------------------------------------

    def rollback_current(self, disease: str, target_model_id: str) -> None:
        if target_model_id not in self.models:
            raise KeyError(f"Unknown model_id {target_model_id} for rollback")
        target = self.models[target_model_id]
        if target.disease != disease:
            raise VersionCompatibilityError(
                f"Rollback violation: target model belongs to {target.disease}, not {disease}"
            )
        self.current_model_by_disease[disease] = target_model_id

    def list_models(self, disease: Optional[str] = None) -> List[ModelMetadata]:
        if disease is None:
            return list(self.models.values())
        return [m for m in self.models.values() if m.disease == disease]

    def list_explainers(self, model_id: Optional[str] = None) -> List[ExplanationMetadata]:
        if model_id is None:
            return list(self.explainers.values())
        return [e for e in self.explainers.values() if e.model_id == model_id]

    # ------------------------------------------------------------------
    # Internal assertions
    # ------------------------------------------------------------------

    def _assert_feature_schema_match(self, model_meta: ModelMetadata, feature_schema_hash: str) -> None:
        if model_meta.feature_schema_hash != feature_schema_hash:
            raise VersionCompatibilityError(
                "Explainer/model mismatch: feature schema hash differs; regenerate explainer for this model_version (FDA/EU traceability)."
            )

    def _assert_model_version_match(self, model_meta: ModelMetadata, model_version: str) -> None:
        if model_meta.model_version != model_version:
            raise VersionCompatibilityError(
                "Explainer/model mismatch: model_version differs; explanations must be bound to the exact model build (FDA/EU)."
            )


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    registry = ModelRegistry()

    model_meta = registry.register_model(
        disease="Sepsis",
        model_name="xgb_sepsis",
        model_version="1.3.2",
        training_data_version="mimic-v5",
        training_data_hash="sha256:abc123",
        feature_schema_hash="sha256:feat123",
        hyperparameters={"max_depth": 6, "n_estimators": 500},
        metrics={"auroc": 0.91, "auprc": 0.55},
        code_hash="git:deadbeef",
        artifact_path="models/sepsis/1.3.2/model.pkl",
        governance_refs={"validation_report": "reports/sepsis_1.3.2.pdf"},
    )

    expl_meta = registry.register_explainer(
        model_id=model_meta.model_id,
        model_version=model_meta.model_version,
        explainer_version="1.0.0",
        method="shap_tree",
        feature_schema_hash="sha256:feat123",
        parameters={"background_size": 200},
        artifact_path="explainers/sepsis/1.3.2/shap.pkl",
    )

    registry.assert_compatible(model_meta.model_id, expl_meta.explainer_id)
    print("Model and explainer registered with enforced compatibility.")
