"""
Adaptive multimodal risk fusion engine.

Combines structured-model probabilities, imaging-model scores, and optional
clinician feedback into a single fused risk with transparent weighting that
reacts to contextual severity scores (SIRS, SOFA-like indicators, shock index,
image latency/quality, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math


@dataclass
class SourceRisk:
    risk: float
    reliability: float = 0.8


class RiskFusionEngine:
    def __init__(self, diseases: List[str]):
        self.diseases = diseases

    @staticmethod
    def _normalize_risk(risk: float) -> float:
        return min(max(risk, 0.0), 1.0)

    @staticmethod
    def _normalize_reliability(rel: float) -> float:
        return min(max(rel, 0.05), 1.5)

    def _context_weight(self, disease: str, context: Dict[str, float]) -> float:
        sirs = context.get("sirs_score", 0)
        sofa = context.get("sofa_like_score", 0)
        shock_index = context.get("shock_index", 0)
        cxr_latency = context.get("cxr_latency_hours", 0)

        weight = 1.0 + 0.05 * sirs + 0.05 * sofa
        if disease in {"sepsis", "kidney_failure"}:
            weight += 0.05 * max(shock_index - 1.0, 0)
        if cxr_latency and disease in {"pneumonia", "sepsis"}:
            weight -= min(cxr_latency / 24.0, 0.2)
        return max(weight, 0.2)

    def _fuse_single(
        self,
        disease: str,
        structured: Optional[SourceRisk],
        imaging: Optional[SourceRisk],
        feedback: Optional[SourceRisk],
        context: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        components = {}
        weights = {}

        if structured:
            c = self._context_weight(disease, context)
            w = self._normalize_reliability(structured.reliability * c)
            weights["structured"] = w
            components["structured"] = structured.risk

        if imaging:
            view_bonus = 0.0
            if context.get("cxr_quality_score"):
                view_bonus = 0.1 * context["cxr_quality_score"]
            w = self._normalize_reliability(imaging.reliability + view_bonus)
            weights["imaging"] = w
            components["imaging"] = imaging.risk

        if feedback:
            w = self._normalize_reliability(0.5 + feedback.reliability)
            weights["feedback"] = w
            components["feedback"] = feedback.risk

        if not weights:
            return 0.0, {}

        total_weight = sum(weights.values())
        fused = sum(
            components[name] * weight for name, weight in weights.items()
        ) / total_weight
        fused = self._normalize_risk(fused)

        confidence = min(0.99, max(0.01, total_weight / (len(weights) * 1.5)))
        entropy = -(
            fused * math.log(fused + 1e-6)
            + (1 - fused) * math.log(1 - fused + 1e-6)
        )
        explanation = {
            "weights": weights,
            "confidence": confidence,
            "entropy": entropy,
            "sources": components,
        }
        return fused, explanation

    def fuse(
        self,
        structured: Dict[str, SourceRisk],
        imaging: Optional[Dict[str, SourceRisk]] = None,
        feedback: Optional[Dict[str, SourceRisk]] = None,
        context: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        context = context or {}
        fused_output: Dict[str, Dict[str, float]] = {}
        for disease in self.diseases:
            fused_risk, details = self._fuse_single(
                disease=disease,
                structured=structured.get(disease),
                imaging=(imaging or {}).get(disease),
                feedback=(feedback or {}).get(disease),
                context=context,
            )
            fused_output[disease] = {
                "risk": fused_risk,
                "details": details,
            }
        return fused_output
