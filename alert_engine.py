"""
Clinical Alert Engine for High-Risk AI Predictions
==================================================

Purpose
-------
Trigger timely, explainable alerts to clinicians when AI predictions exceed
disease-specific risk thresholds. Designed for FDA/EU-regulated deployment
with mandatory explanation attachment, audit logging, and alert fatigue
prevention.

Key Features
------------
- Disease-specific threshold policies (MODERATE / HIGH / CRITICAL)
- Mandatory explanation summary (fail-closed: no explanation = no alert)
- Alert deduplication and cooldown (prevent alert fatigue)
- Audit logging of every alert with linked prediction/explanation IDs
- Clinician-facing formatting with actionable recommendations
- Escalation rules based on severity and response time

Regulatory Compliance
---------------------
- FDA SaMD: Alerts must be traceable, explainable, and logged for post-market surveillance
- EU AI Act: High-risk alerts require human oversight and transparency (explanation attachment)
- Clinical Safety: Alerts include severity, rationale, and recommended actions

Usage
-----
    ctx = GovernanceContext.from_logger(...)
    engine = AlertEngine(ctx, alert_policies)
    
    # After prediction and explanation
    alert = engine.trigger_alert(
        trace=decision_trace,
        prediction_probability=0.82,
        explanation_summary="High lactate (4.2 mmol/L) and elevated WBC (15.2)",
        threshold_used=0.35
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone

from .governance import GovernanceContext, RegulatoryViolationError
from .decision_trace import DecisionTrace
from .audit_logging import AuditEvent


# =============================================================================
# Alert Severity and Policy
# =============================================================================

class AlertSeverity(str, Enum):
    """Three-tier severity classification for clinical alerts."""
    MODERATE = "MODERATE"  # Elevated risk, monitor closely
    HIGH = "HIGH"          # Significant risk, prompt evaluation
    CRITICAL = "CRITICAL"  # Immediate intervention required


@dataclass
class AlertPolicy:
    """Disease-specific threshold policy for alert triggering."""
    disease: str
    moderate_threshold: float  # e.g., 0.35
    high_threshold: float      # e.g., 0.55
    critical_threshold: float  # e.g., 0.75
    
    cooldown_seconds: int = 300  # Prevent duplicate alerts (5 min default)
    escalation_timeout_seconds: int = 900  # Escalate if no response (15 min)
    
    moderate_actions: List[str] = field(default_factory=list)
    high_actions: List[str] = field(default_factory=list)
    critical_actions: List[str] = field(default_factory=list)
    
    def get_severity(self, probability: float) -> Optional[AlertSeverity]:
        """Classify probability into severity tier."""
        if probability >= self.critical_threshold:
            return AlertSeverity.CRITICAL
        elif probability >= self.high_threshold:
            return AlertSeverity.HIGH
        elif probability >= self.moderate_threshold:
            return AlertSeverity.MODERATE
        return None  # Below alert threshold
    
    def get_recommended_actions(self, severity: AlertSeverity) -> List[str]:
        """Retrieve recommended actions for severity level."""
        if severity == AlertSeverity.CRITICAL:
            return self.critical_actions
        elif severity == AlertSeverity.HIGH:
            return self.high_actions
        elif severity == AlertSeverity.MODERATE:
            return self.moderate_actions
        return []


# =============================================================================
# Default Alert Policies (Disease-Specific)
# =============================================================================

DEFAULT_ALERT_POLICIES = {
    "Sepsis": AlertPolicy(
        disease="Sepsis",
        moderate_threshold=0.35,
        high_threshold=0.55,
        critical_threshold=0.75,
        cooldown_seconds=300,
        escalation_timeout_seconds=900,
        moderate_actions=[
            "Monitor vitals every 2 hours",
            "Review trend in lactate and WBC",
            "Consider repeat clinical assessment"
        ],
        high_actions=[
            "Obtain blood cultures immediately",
            "Initiate broad-spectrum antibiotics within 1 hour",
            "Administer IV fluids (30 mL/kg crystalloid)",
            "Notify attending physician"
        ],
        critical_actions=[
            "Activate sepsis protocol (Code Sepsis)",
            "Transfer to ICU immediately",
            "Obtain blood cultures and start antibiotics STAT",
            "Consider vasopressor support",
            "Notify critical care team"
        ]
    ),
    "Acute Kidney Injury": AlertPolicy(
        disease="Acute Kidney Injury",
        moderate_threshold=0.30,
        high_threshold=0.50,
        critical_threshold=0.70,
        cooldown_seconds=600,  # 10 min (AKI evolves slower)
        escalation_timeout_seconds=1800,  # 30 min
        moderate_actions=[
            "Monitor creatinine and urine output",
            "Review nephrotoxic medications",
            "Ensure adequate hydration"
        ],
        high_actions=[
            "Discontinue nephrotoxic drugs (NSAIDs, ACE-I)",
            "Fluid resuscitation if hypovolemic",
            "Obtain renal ultrasound to rule out obstruction",
            "Notify nephrology for consultation"
        ],
        critical_actions=[
            "Urgent nephrology consultation",
            "Consider emergent dialysis if indicated",
            "Rule out urinary obstruction (Foley catheter, imaging)",
            "Notify ICU for potential admission"
        ]
    ),
    "Cardiovascular Event": AlertPolicy(
        disease="Cardiovascular Event",
        moderate_threshold=0.25,
        high_threshold=0.45,
        critical_threshold=0.65,
        cooldown_seconds=180,  # 3 min (CV events are time-sensitive)
        escalation_timeout_seconds=600,  # 10 min
        moderate_actions=[
            "Obtain 12-lead ECG",
            "Monitor for chest pain or dyspnea",
            "Check troponin levels"
        ],
        high_actions=[
            "Activate chest pain protocol",
            "Administer aspirin 325 mg",
            "Obtain serial troponins and ECG",
            "Notify cardiology on call"
        ],
        critical_actions=[
            "Activate STEMI protocol (Code STEMI)",
            "Transfer to cath lab immediately",
            "Administer dual antiplatelet therapy",
            "Alert interventional cardiology STAT"
        ]
    ),
    "Mortality Risk": AlertPolicy(
        disease="Mortality Risk",
        moderate_threshold=0.20,
        high_threshold=0.40,
        critical_threshold=0.60,
        cooldown_seconds=600,
        escalation_timeout_seconds=1200,  # 20 min
        moderate_actions=[
            "Increase frequency of vital sign monitoring",
            "Review goals of care with patient/family",
            "Ensure nursing awareness of elevated risk"
        ],
        high_actions=[
            "Notify attending physician of elevated mortality risk",
            "Consider rapid response team evaluation",
            "Review and optimize management of active conditions",
            "Discuss with family if appropriate"
        ],
        critical_actions=[
            "Urgent physician evaluation required",
            "Consider ICU transfer",
            "Activate rapid response team",
            "Discuss goals of care urgently with family"
        ]
    )
}


# =============================================================================
# Alert Data Model
# =============================================================================

@dataclass
class ClinicalAlert:
    """Clinician-facing alert with explanation and recommended actions."""
    alert_id: str
    patient_id: str
    disease: str
    severity: AlertSeverity
    probability: float
    threshold_used: float
    
    explanation_summary: str  # REQUIRED (e.g., "High lactate + elevated WBC")
    explanation_details: Optional[Dict[str, Any]] = None  # Top SHAP features
    
    recommended_actions: List[str] = field(default_factory=list)
    
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    
    prediction_id: Optional[str] = None
    explanation_id: Optional[str] = None
    
    def to_clinician_message(self) -> str:
        """Format alert for clinical display (EHR banner, pager message)."""
        severity_emoji = {
            AlertSeverity.MODERATE: "⚠️",
            AlertSeverity.HIGH: "🔴",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        msg = f"{severity_emoji[self.severity]} {self.severity.value} ALERT: {self.disease}\n"
        msg += f"Patient: {self.patient_id}\n"
        msg += f"Risk: {self.probability:.0%} (Threshold: {self.threshold_used:.0%})\n"
        msg += f"\nRationale: {self.explanation_summary}\n"
        
        if self.recommended_actions:
            msg += f"\nRecommended Actions:\n"
            for action in self.recommended_actions:
                msg += f"  • {action}\n"
        
        msg += f"\nGenerated: {self.timestamp_utc[:19]} UTC"
        return msg


# =============================================================================
# Alert Engine
# =============================================================================

class AlertEngine:
    """
    Trigger explainable clinical alerts with audit logging and fatigue prevention.
    
    Enforces:
    - No alert without explanation (fail-closed)
    - Alert deduplication via cooldown
    - Audit logging of every alert
    - Severity-based escalation
    """
    
    def __init__(
        self,
        ctx: GovernanceContext,
        alert_policies: Optional[Dict[str, AlertPolicy]] = None
    ):
        self.ctx = ctx
        self.policies = alert_policies or DEFAULT_ALERT_POLICIES
        
        # Alert fatigue prevention
        self._last_alert_time: Dict[str, float] = {}  # (patient_id, disease) -> timestamp
        
        # Escalation tracking
        self._pending_escalations: Dict[str, ClinicalAlert] = {}  # alert_id -> alert
    
    # --------------------------------------------------------------------------
    # Core Alert Triggering
    # --------------------------------------------------------------------------
    
    def trigger_alert(
        self,
        trace: DecisionTrace,
        prediction_probability: float,
        explanation_summary: str,
        threshold_used: float,
        explanation_details: Optional[Dict[str, Any]] = None
    ) -> Optional[ClinicalAlert]:
        """
        Trigger alert if probability exceeds threshold.
        
        Requirements (fail-closed):
        - trace must have prediction_event and explanation_event
        - explanation_summary must be provided (no empty explanations)
        
        Returns:
            ClinicalAlert if triggered, None if below threshold or in cooldown
        
        Raises:
            RegulatoryViolationError if explanation missing
        """
        # Guard: require explanation
        if not explanation_summary or not explanation_summary.strip():
            raise RegulatoryViolationError(
                "Regulatory violation: Alert cannot be triggered without explanation summary (FDA/EU explainability)."
            )
        
        # Guard: require prediction and explanation events in trace
        if trace.prediction_event is None:
            raise RegulatoryViolationError(
                "Regulatory violation: Alert requires logged prediction event (FDA/EU traceability)."
            )
        if trace.explanation_event is None:
            raise RegulatoryViolationError(
                "Regulatory violation: Alert requires logged explanation event (FDA/EU explainability)."
            )
        
        # Get policy
        policy = self.policies.get(trace.disease)
        if policy is None:
            raise ValueError(f"No alert policy defined for disease: {trace.disease}")
        
        # Determine severity
        severity = policy.get_severity(prediction_probability)
        if severity is None:
            return None  # Below alert threshold
        
        # Check cooldown (prevent alert fatigue)
        cooldown_key = (trace.patient_id, trace.disease)
        if self._is_in_cooldown(cooldown_key, policy.cooldown_seconds):
            return None  # Suppressed due to recent alert
        
        # Create alert
        alert = ClinicalAlert(
            alert_id=f"alert-{trace.trace_id}",
            patient_id=trace.patient_id,
            disease=trace.disease,
            severity=severity,
            probability=prediction_probability,
            threshold_used=threshold_used,
            explanation_summary=explanation_summary,
            explanation_details=explanation_details,
            recommended_actions=policy.get_recommended_actions(severity),
            prediction_id=trace.prediction_event.event_id,
            explanation_id=trace.explanation_event.event_id,
        )
        
        # Log alert to audit trail
        alert_event = self.ctx.log_alert(
            prediction_event=trace.prediction_event,
            explanation_event=trace.explanation_event,
            threshold=threshold_used,
            payload={
                "alert_id": alert.alert_id,
                "severity": severity.value,
                "probability": prediction_probability,
                "explanation_summary": explanation_summary,
                "recommended_actions": alert.recommended_actions,
            },
            human_message=f"{severity.value} alert for {trace.disease}: {explanation_summary}"
        )
        
        # Update trace
        trace.alert_event = alert_event
        
        # Update cooldown
        self._last_alert_time[cooldown_key] = time.time()
        
        # Track for escalation
        self._pending_escalations[alert.alert_id] = alert
        
        return alert
    
    # --------------------------------------------------------------------------
    # Alert Acknowledgment
    # --------------------------------------------------------------------------
    
    def acknowledge_alert(
        self,
        alert: ClinicalAlert,
        clinician_id: str,
        notes: Optional[str] = None
    ) -> AuditEvent:
        """
        Record clinician acknowledgment of alert.
        
        Logs clinician action to audit trail and removes from pending escalations.
        """
        alert.acknowledged = True
        alert.acknowledged_by = clinician_id
        alert.acknowledged_at = datetime.now(timezone.utc).isoformat()
        
        # Log to audit (requires alert_event from trace)
        # In practice, retrieve trace or pass alert_event directly
        # For now, create a stub (in real system, link to trace.alert_event)
        from audit_logging import AuditEvent, AuditEventType
        
        # This would typically come from trace.alert_event
        # Here we create a minimal event for demonstration
        action_event = self.ctx.audit_logger.log_event(
            event_type=AuditEventType.CLINICIAN_ACTION,
            patient_id=alert.patient_id,
            disease=alert.disease,
            alert_id=alert.alert_id,
            prediction_id=alert.prediction_id,
            explanation_id=alert.explanation_id,
            payload={
                "action": "acknowledged",
                "clinician_id": clinician_id,
                "notes": notes or "",
                "alert_severity": alert.severity.value,
            },
            human_message=f"Clinician {clinician_id} acknowledged {alert.severity.value} alert for {alert.disease}"
        )
        
        # Remove from escalation queue
        if alert.alert_id in self._pending_escalations:
            del self._pending_escalations[alert.alert_id]
        
        return action_event
    
    # --------------------------------------------------------------------------
    # Escalation Logic
    # --------------------------------------------------------------------------
    
    def check_escalations(self) -> List[ClinicalAlert]:
        """
        Check for unacknowledged alerts that require escalation.
        
        Returns list of alerts exceeding escalation timeout.
        """
        now = time.time()
        escalated = []
        
        for alert in list(self._pending_escalations.values()):
            if alert.acknowledged:
                continue
            
            policy = self.policies.get(alert.disease)
            if policy is None:
                continue
            
            alert_time = datetime.fromisoformat(alert.timestamp_utc).timestamp()
            elapsed = now - alert_time
            
            if elapsed > policy.escalation_timeout_seconds:
                escalated.append(alert)
        
        return escalated
    
    # --------------------------------------------------------------------------
    # Alert Fatigue Prevention
    # --------------------------------------------------------------------------
    
    def _is_in_cooldown(self, cooldown_key: tuple, cooldown_seconds: int) -> bool:
        """Check if alert is in cooldown period."""
        last_time = self._last_alert_time.get(cooldown_key)
        if last_time is None:
            return False
        
        elapsed = time.time() - last_time
        return elapsed < cooldown_seconds
    
    # --------------------------------------------------------------------------
    # Safety Rules
    # --------------------------------------------------------------------------
    
    def validate_alert_safety(self, alert: ClinicalAlert) -> List[str]:
        """
        Validate alert safety rules.
        
        Returns list of warnings (empty if all checks pass).
        """
        warnings = []
        
        # Rule 1: Explanation must not be empty
        if not alert.explanation_summary or len(alert.explanation_summary.strip()) < 10:
            warnings.append("Explanation summary too short (minimum 10 characters)")
        
        # Rule 2: Critical alerts must have at least 2 recommended actions
        if alert.severity == AlertSeverity.CRITICAL:
            if len(alert.recommended_actions) < 2:
                warnings.append("Critical alerts must have at least 2 recommended actions")
        
        # Rule 3: Probability must significantly exceed threshold
        margin = alert.probability - alert.threshold_used
        if margin < 0.05:  # Less than 5% above threshold
            warnings.append(f"Probability ({alert.probability:.2f}) barely exceeds threshold ({alert.threshold_used:.2f})")
        
        return warnings


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    from audit_logging import AuditLogger
    from governance import GovernanceContext
    from decision_trace import DecisionTraceManager
    
    # Setup
    ctx = GovernanceContext.from_logger(
        audit_logger=AuditLogger("logs/audit.log", system_id="ED-AI-01"),
        actor_id="alert_service",
        session_id="sess-alert-001"
    )
    
    engine = AlertEngine(ctx)
    trace_mgr = DecisionTraceManager(ctx)
    
    # Create trace
    trace = trace_mgr.start_trace(
        patient_id="P123",
        disease="Sepsis",
        model_name="xgb_sepsis",
        model_version="1.3.2",
        input_summary={"age": 67, "hr": 120, "lactate": 4.2, "wbc": 15.2}
    )
    
    # Log prediction
    trace_mgr.log_prediction(
        trace,
        payload={"probability": 0.82, "threshold": 0.35},
        human_message="Sepsis risk predicted at 82%",
        threshold=0.35
    )
    
    # Log explanation
    trace_mgr.log_explanation(
        trace,
        payload={
            "method": "shap",
            "top_features": [
                {"name": "lactate", "value": 4.2, "shap": 0.18},
                {"name": "wbc", "value": 15.2, "shap": 0.12},
                {"name": "heart_rate", "value": 120, "shap": 0.09}
            ]
        },
        human_message="SHAP explanation: lactate, wbc, heart_rate are top risk factors"
    )
    
    # Trigger alert
    alert = engine.trigger_alert(
        trace=trace,
        prediction_probability=0.82,
        explanation_summary="High lactate (4.2 mmol/L), elevated WBC (15.2), tachycardia (HR 120)",
        threshold_used=0.35,
        explanation_details={
            "top_features": [
                {"name": "lactate", "value": 4.2, "shap": 0.18},
                {"name": "wbc", "value": 15.2, "shap": 0.12},
                {"name": "heart_rate", "value": 120, "shap": 0.09}
            ]
        }
    )
    
    if alert:
        print("=" * 80)
        print("CLINICAL ALERT TRIGGERED")
        print("=" * 80)
        print(alert.to_clinician_message())
        print("\n" + "=" * 80)
        
        # Validate safety
        warnings = engine.validate_alert_safety(alert)
        if warnings:
            print("\nSafety warnings:")
            for w in warnings:
                print(f"  ⚠️  {w}")
        else:
            print("\n✓ Alert passed all safety checks")
        
        # Simulate acknowledgment
        print("\n[Simulating clinician acknowledgment...]")
        engine.acknowledge_alert(alert, clinician_id="DR_SMITH", notes="Started sepsis bundle")
        print(f"✓ Alert acknowledged by DR_SMITH")
