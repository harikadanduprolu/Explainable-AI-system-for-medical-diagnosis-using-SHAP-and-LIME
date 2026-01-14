# Regulatory Compliance Matrix

## Overview

This document maps regulatory requirements (FDA, EU AI Act, GDPR) to system implementations.


## FDA_SAMD

### [FDA-1] Algorithm Change Traceability

**Compliance Status:** FULL

**System Components:**
- `model_registry.py (ModelRegistry)`
- `audit_logging.py (MODEL_VERSION events)`
- `decision_trace.py (model_version binding)`

**Implementation:**
ModelRegistry stores immutable ModelMetadata for each trained model, including model_version, training_data_version, training_data_hash, feature_schema_hash, hyperparameters, metrics, and code_hash. Every prediction event logs the active model_version via audit_logging. Decision traces bind model_version to each clinical decision, enabling reconstruction of which algorithm was active at decision time.

**Evidence:** model_registry.py lines 70-130 (ModelMetadata schema), audit_logging.py lines 50-80 (MODEL_VERSION event type), decision_trace.py lines 30-50 (model_version in DecisionTrace)

---

### [FDA-3] Risk Management and Traceability

**Compliance Status:** FULL

**System Components:**
- `decision_trace.py (DecisionTrace)`
- `governance.py (GovernanceContext)`
- `alert_engine.py (severity classification)`

**Implementation:**
DecisionTrace links input_summary → prediction → explanation → alert → clinician_action in a single traceable record. GovernanceContext enforces fail-closed logging: no prediction without audit trail, no alert without explanation. Alert engine classifies severity (MODERATE/HIGH/CRITICAL) with recommended actions per tier.

**Evidence:** decision_trace.py lines 40-100 (DecisionTrace schema), governance.py lines 50-120 (fail-closed guards), alert_engine.py lines 80-150 (AlertSeverity classification)

---

### [FDA-5] Clinical Performance Validation

**Compliance Status:** FULL

**System Components:**
- `evaluation_pipeline.py (EvaluationPipeline)`

**Implementation:**
EvaluationPipeline computes AUROC, AUPRC with 95% CI (1000 bootstrap samples), accuracy, precision, recall, F1, specificity, Brier score, log loss. Generates ROC/PR curves, calibration plots, confusion matrices. Exports LaTeX tables for publication/submission.

**Evidence:** evaluation_pipeline.py lines 200-450 (ModelMetrics with CI), evaluation_pipeline.py lines 550-700 (ROC/PR plot generation), evaluation_pipeline.py lines 900-1000 (LaTeX table export)

---


## FDA_MLDS

### [FDA-2] Post-Market Surveillance and Monitoring

**Compliance Status:** FULL

**System Components:**
- `audit_logging.py (hash-chained event log)`
- `evaluation_pipeline.py (performance metrics)`
- `alert_engine.py (alert tracking)`

**Implementation:**
Audit logger captures every prediction, explanation, alert, and clinician action with SHA-256 hash chaining for tamper evidence. Evaluation pipeline computes AUROC, AUPRC, calibration metrics on test sets, enabling periodic performance drift detection. Alert engine tracks alert rates, acknowledgment rates, and escalations for safety monitoring.

**Evidence:** audit_logging.py lines 150-250 (AuditLogger with hash chain), evaluation_pipeline.py lines 200-400 (ModelMetrics computation), alert_engine.py lines 300-350 (alert tracking)

---

### [FDA-4] Data Provenance and Quality

**Compliance Status:** FULL

**System Components:**
- `model_registry.py (training_data_version, training_data_hash)`
- `evaluation_pipeline.py (DataSplit with metadata)`

**Implementation:**
ModelMetadata includes training_data_version (e.g., 'mimic-v5') and training_data_hash (SHA-256 of training set) for complete provenance. DataSplit records split_params (test_size, stratified, random_seed) and class distributions for reproducibility.

**Evidence:** model_registry.py lines 75-85 (training_data_version/hash fields), evaluation_pipeline.py lines 100-150 (DataSplit with provenance)

---


## EU_AI_ACT

### [EU-1] User Transparency and Information

**Compliance Status:** FULL

**System Components:**
- `xai_engine.py (ClinicalExplanation with clinical_summary)`
- `alert_engine.py (ClinicalAlert.to_clinician_message)`
- `governed_whatif_engine.py (SimulationResult with disclaimer)`

**Implementation:**
XAI Engine generates clinical_summary in plain English (e.g., 'High lactate and elevated WBC indicate infection risk'). Alert messages include severity, risk probability, and recommended actions. What-if simulations include mandatory disclaimer: 'HYPOTHETICAL ONLY' with plausibility assessment.

**Evidence:** xai_engine.py lines 200-250 (ClinicalExplanation schema), alert_engine.py lines 150-200 (to_clinician_message formatting), governed_whatif_engine.py lines 250-300 (disclaimer logic)

---

### [EU-2] Automatic Logging of Events

**Compliance Status:** FULL

**System Components:**
- `audit_logging.py (AuditLogger)`
- `governance.py (mandatory logging enforcement)`

**Implementation:**
AuditLogger records every prediction, explanation, alert, clinician action, model version change, threshold change, and failure event. Hash-chained with SHA-256 for tamper detection. GovernanceContext enforces fail-closed: no operation without active logger.

**Evidence:** audit_logging.py lines 50-100 (AuditEventType taxonomy), audit_logging.py lines 150-300 (AuditLogger implementation), governance.py lines 50-80 (require_logger guards)

---

### [EU-3] Human Oversight and Intervention

**Compliance Status:** FULL

**System Components:**
- `alert_engine.py (clinician acknowledgment)`
- `governed_whatif_engine.py (clinician-initiated simulations)`
- `decision_trace.py (clinician_action_event)`

**Implementation:**
Alert engine requires explicit clinician acknowledgment (logged with clinician_id, notes, timestamp). What-if simulations require clinician_id and log every simulation. Decision traces capture clinician_action_event with action details. No autonomous action without human-in-the-loop.

**Evidence:** alert_engine.py lines 400-450 (acknowledge_alert method), governed_whatif_engine.py lines 200-250 (clinician_id requirement), decision_trace.py lines 150-180 (log_clinician_action)

---

### [EU-4] Accuracy and Robustness Requirements

**Compliance Status:** FULL

**System Components:**
- `evaluation_pipeline.py (comprehensive metrics)`
- `model_registry.py (metrics stored per version)`

**Implementation:**
Evaluation pipeline computes 11+ metrics (AUROC, AUPRC, calibration, etc.) with 95% CI. Metrics stored in ModelMetadata per version, enabling performance tracking over time. Threshold validation ensures predictions meet clinical accuracy requirements.

**Evidence:** evaluation_pipeline.py lines 200-300 (ModelMetrics with CI), model_registry.py lines 80-90 (metrics field in ModelMetadata)

**⚠️ Gaps:** Adversarial robustness testing not yet implemented

**Remediation:** Add adversarial perturbation tests to evaluation_pipeline (Q2 2026)

---

### [EU-5] Technical Documentation and Version Control

**Compliance Status:** FULL

**System Components:**
- `model_registry.py (ModelMetadata)`
- `technical_effect_registry.py (TechnicalEffect documentation)`

**Implementation:**
ModelRegistry stores complete metadata: model_version, training_data_version/hash, feature_schema_hash, hyperparameters, metrics, code_hash, artifact_path. Technical effect registry documents problems, solutions, and measurable effects for each component.

**Evidence:** model_registry.py lines 70-130 (ModelMetadata schema), technical_effect_registry.py lines 100-200 (TechnicalEffect schema)

---

### [EU-6] Training Data Quality and Governance

**Compliance Status:** PARTIAL

**System Components:**
- `model_registry.py (training_data_version, training_data_hash)`
- `evaluation_pipeline.py (stratified splits with class balance)`

**Implementation:**
ModelMetadata captures training_data_version and SHA-256 hash of training set. DataSplit uses stratified sampling to preserve class balance across train/val/test. Class distribution reported for each split.

**Evidence:** model_registry.py lines 75-85 (data provenance fields), evaluation_pipeline.py lines 100-200 (stratified splitting)

**⚠️ Gaps:** Bias analysis not yet implemented

**Remediation:** Add fairness metrics (demographic parity, equalized odds) to evaluation_pipeline (Q2 2026)

---


## GDPR

### [GDPR-1] Right to Information about Processing

**Compliance Status:** FULL

**System Components:**
- `xai_engine.py (ClinicalExplanation)`
- `decision_trace.py (input_summary binding)`

**Implementation:**
DecisionTrace captures input_summary (patient features used in decision). XAI Engine generates ClinicalExplanation with top_risk_factors (feature name, value, importance, clinical interpretation). Patients can request trace via patient_id.

**Evidence:** decision_trace.py lines 40-60 (input_summary field), xai_engine.py lines 200-300 (ClinicalExplanation with feature details)

---

### [GDPR-2] Right to Explanation for Automated Decisions

**Compliance Status:** FULL

**System Components:**
- `xai_engine.py (SHAP/LIME explanations)`
- `governance.py (no prediction without explanation)`

**Implementation:**
XAI Engine generates SHAP/LIME explanations for every prediction, providing feature importances and clinical interpretations. GovernanceContext enforces: no alert without explanation_event. Clinical summaries translate technical outputs to plain language.

**Evidence:** xai_engine.py lines 400-600 (SHAP/LIME implementation), governance.py lines 80-120 (require_explanation_link guards)

---

### [GDPR-3] Data Subject Access to Processing Records

**Compliance Status:** FULL

**System Components:**
- `audit_logging.py (patient_id indexing)`
- `decision_trace.py (DecisionTraceManager)`

**Implementation:**
Audit log includes patient_id in every event. DecisionTraceManager stores traces indexed by patient_id. Export functions (export_human_readable) generate CSV reports for patient data access requests.

**Evidence:** audit_logging.py lines 60-80 (patient_id field in AuditEvent), audit_logging.py lines 350-400 (export_human_readable method), decision_trace.py lines 100-150 (trace storage by patient_id)

---

### [GDPR-4] Privacy by Design and Default

**Compliance Status:** PARTIAL

**System Components:**
- `audit_logging.py (no PII in payload by default)`
- `decision_trace.py (input_summary pseudonymization)`

**Implementation:**
Audit events use patient_id (pseudonym) rather than PHI. Payload fields contain clinical features (numeric values) without names/identifiers. Input summaries use de-identified feature names.

**Evidence:** audit_logging.py lines 60-90 (patient_id as pseudonym), decision_trace.py lines 50-70 (input_summary with de-identified features)

**⚠️ Gaps:** Encryption at rest not enforced by code

**Remediation:** Add encrypted audit log storage with key management (Q3 2026)

---

### [GDPR-5] Accountability and Demonstrable Compliance

**Compliance Status:** FULL

**System Components:**
- `audit_logging.py (immutable audit trail)`
- `governance.py (fail-closed enforcement)`
- `technical_effect_registry.py (technical documentation)`

**Implementation:**
Hash-chained audit log provides cryptographic proof of event integrity. GovernanceContext enforces mandatory logging with RegulatoryViolationError on bypass. Technical effect registry documents compliance measures and measurable effects.

**Evidence:** audit_logging.py lines 150-300 (hash chain implementation), governance.py lines 20-50 (RegulatoryViolationError guards), technical_effect_registry.py (complete compliance documentation)

---

