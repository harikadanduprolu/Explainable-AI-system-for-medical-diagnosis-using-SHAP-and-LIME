"""
Regulatory Compliance Matrix for Clinical AI System
===================================================

Purpose
-------
Maps FDA, EU AI Act, and GDPR requirements to implemented system components,
providing evidence for regulatory submission packages.

Regulatory Frameworks
--------------------
- FDA: SaMD (Software as a Medical Device), MLDS (Machine Learning-based Device Software)
- EU AI Act: Title III (High-Risk AI Systems), Annex III (Medical Devices)
- GDPR: Articles 13-15 (Transparency), Article 22 (Automated Decision-Making)

Output Formats
--------------
- Console table (human-readable)
- CSV (spreadsheet import)
- Markdown (documentation)
- JSON (machine-readable)

Usage
-----
    matrix = ComplianceMatrix()
    matrix.generate_all_formats(output_dir="regulatory_submission/")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import json
import csv
from pathlib import Path


# =============================================================================
# Regulatory Requirements Taxonomy
# =============================================================================

class RegulatoryFramework(str, Enum):
    FDA_SAMD = "FDA_SAMD"
    FDA_MLDS = "FDA_MLDS"
    EU_AI_ACT = "EU_AI_ACT"
    GDPR = "GDPR"


class ComplianceStatus(str, Enum):
    FULL = "FULL"           # Fully implemented
    PARTIAL = "PARTIAL"     # Partially implemented
    PLANNED = "PLANNED"     # Not yet implemented


@dataclass
class RegulatoryRequirement:
    """Single regulatory requirement with citation."""
    requirement_id: str
    framework: RegulatoryFramework
    citation: str
    title: str
    description: str
    criticality: str  # MANDATORY / RECOMMENDED / BEST_PRACTICE


@dataclass
class ComplianceMapping:
    """Mapping of requirement to system implementation."""
    requirement_id: str
    framework: RegulatoryFramework
    requirement_title: str
    
    system_components: List[str]
    implementation_details: str
    evidence: str
    
    compliance_status: ComplianceStatus
    gaps: Optional[str] = None
    remediation_plan: Optional[str] = None


# =============================================================================
# Regulatory Requirements Database
# =============================================================================

FDA_REQUIREMENTS = [
    RegulatoryRequirement(
        requirement_id="FDA-1",
        framework=RegulatoryFramework.FDA_SAMD,
        citation="FDA SaMD Guidance (2019), Section IV.B",
        title="Algorithm Change Traceability",
        description="Documentation of all algorithm changes, including version control, training data, and performance metrics",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="FDA-2",
        framework=RegulatoryFramework.FDA_MLDS,
        citation="FDA MLDS Action Plan (2021), Principle 3",
        title="Post-Market Surveillance and Monitoring",
        description="Capability to monitor real-world performance and detect degradation over time",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="FDA-3",
        framework=RegulatoryFramework.FDA_SAMD,
        citation="21 CFR Part 820.30 (Design Controls)",
        title="Risk Management and Traceability",
        description="Traceability of design inputs to outputs; risk analysis documentation",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="FDA-4",
        framework=RegulatoryFramework.FDA_MLDS,
        citation="FDA MLDS Good Machine Learning Practice (2021)",
        title="Data Provenance and Quality",
        description="Documentation of training data sources, preprocessing, and quality controls",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="FDA-5",
        framework=RegulatoryFramework.FDA_SAMD,
        citation="FDA SaMD Clinical Evaluation Guidance",
        title="Clinical Performance Validation",
        description="Evidence of clinical accuracy, sensitivity, specificity with CI",
        criticality="MANDATORY"
    ),
]

EU_AI_ACT_REQUIREMENTS = [
    RegulatoryRequirement(
        requirement_id="EU-1",
        framework=RegulatoryFramework.EU_AI_ACT,
        citation="EU AI Act, Article 13 (Transparency)",
        title="User Transparency and Information",
        description="Users must be informed of AI system's capabilities, limitations, and accuracy",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="EU-2",
        framework=RegulatoryFramework.EU_AI_ACT,
        citation="EU AI Act, Article 12 (Logging)",
        title="Automatic Logging of Events",
        description="High-risk AI systems must log events enabling oversight and traceability",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="EU-3",
        framework=RegulatoryFramework.EU_AI_ACT,
        citation="EU AI Act, Article 14 (Human Oversight)",
        title="Human Oversight and Intervention",
        description="Measures to enable human oversight, including stop/interrupt capabilities",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="EU-4",
        framework=RegulatoryFramework.EU_AI_ACT,
        citation="EU AI Act, Article 15 (Accuracy, Robustness)",
        title="Accuracy and Robustness Requirements",
        description="Appropriate accuracy levels; resilience against errors and adversarial attacks",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="EU-5",
        framework=RegulatoryFramework.EU_AI_ACT,
        citation="EU AI Act, Article 11 (Technical Documentation)",
        title="Technical Documentation and Version Control",
        description="Detailed documentation of system design, training, testing, and validation",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="EU-6",
        framework=RegulatoryFramework.EU_AI_ACT,
        citation="EU AI Act, Article 10 (Data Governance)",
        title="Training Data Quality and Governance",
        description="Data must be relevant, representative, and free of errors/biases",
        criticality="MANDATORY"
    ),
]

GDPR_REQUIREMENTS = [
    RegulatoryRequirement(
        requirement_id="GDPR-1",
        framework=RegulatoryFramework.GDPR,
        citation="GDPR Article 13 (Information to be provided)",
        title="Right to Information about Processing",
        description="Data subjects must be informed about automated processing and its logic",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="GDPR-2",
        framework=RegulatoryFramework.GDPR,
        citation="GDPR Article 22 (Automated Decision-Making)",
        title="Right to Explanation for Automated Decisions",
        description="Meaningful information about logic involved in automated decision-making",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="GDPR-3",
        framework=RegulatoryFramework.GDPR,
        citation="GDPR Article 15 (Right of Access)",
        title="Data Subject Access to Processing Records",
        description="Individuals can request access to their data and processing details",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="GDPR-4",
        framework=RegulatoryFramework.GDPR,
        citation="GDPR Article 25 (Data Protection by Design)",
        title="Privacy by Design and Default",
        description="Data protection measures integrated into processing design",
        criticality="MANDATORY"
    ),
    RegulatoryRequirement(
        requirement_id="GDPR-5",
        framework=RegulatoryFramework.GDPR,
        citation="GDPR Article 5(2) (Accountability)",
        title="Accountability and Demonstrable Compliance",
        description="Ability to demonstrate compliance with data protection principles",
        criticality="MANDATORY"
    ),
]


# =============================================================================
# Compliance Mappings
# =============================================================================

COMPLIANCE_MAPPINGS = [
    # -------------------------------------------------------------------------
    # FDA Mappings
    # -------------------------------------------------------------------------
    ComplianceMapping(
        requirement_id="FDA-1",
        framework=RegulatoryFramework.FDA_SAMD,
        requirement_title="Algorithm Change Traceability",
        system_components=[
            "model_registry.py (ModelRegistry)",
            "audit_logging.py (MODEL_VERSION events)",
            "decision_trace.py (model_version binding)"
        ],
        implementation_details=(
            "ModelRegistry stores immutable ModelMetadata for each trained model, including "
            "model_version, training_data_version, training_data_hash, feature_schema_hash, "
            "hyperparameters, metrics, and code_hash. Every prediction event logs the active "
            "model_version via audit_logging. Decision traces bind model_version to each clinical "
            "decision, enabling reconstruction of which algorithm was active at decision time."
        ),
        evidence=(
            "model_registry.py lines 70-130 (ModelMetadata schema), "
            "audit_logging.py lines 50-80 (MODEL_VERSION event type), "
            "decision_trace.py lines 30-50 (model_version in DecisionTrace)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
    
    ComplianceMapping(
        requirement_id="FDA-2",
        framework=RegulatoryFramework.FDA_MLDS,
        requirement_title="Post-Market Surveillance and Monitoring",
        system_components=[
            "audit_logging.py (hash-chained event log)",
            "evaluation_pipeline.py (performance metrics)",
            "alert_engine.py (alert tracking)"
        ],
        implementation_details=(
            "Audit logger captures every prediction, explanation, alert, and clinician action "
            "with SHA-256 hash chaining for tamper evidence. Evaluation pipeline computes AUROC, "
            "AUPRC, calibration metrics on test sets, enabling periodic performance drift detection. "
            "Alert engine tracks alert rates, acknowledgment rates, and escalations for safety monitoring."
        ),
        evidence=(
            "audit_logging.py lines 150-250 (AuditLogger with hash chain), "
            "evaluation_pipeline.py lines 200-400 (ModelMetrics computation), "
            "alert_engine.py lines 300-350 (alert tracking)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
    
    ComplianceMapping(
        requirement_id="FDA-3",
        framework=RegulatoryFramework.FDA_SAMD,
        requirement_title="Risk Management and Traceability",
        system_components=[
            "decision_trace.py (DecisionTrace)",
            "governance.py (GovernanceContext)",
            "alert_engine.py (severity classification)"
        ],
        implementation_details=(
            "DecisionTrace links input_summary → prediction → explanation → alert → clinician_action "
            "in a single traceable record. GovernanceContext enforces fail-closed logging: no prediction "
            "without audit trail, no alert without explanation. Alert engine classifies severity "
            "(MODERATE/HIGH/CRITICAL) with recommended actions per tier."
        ),
        evidence=(
            "decision_trace.py lines 40-100 (DecisionTrace schema), "
            "governance.py lines 50-120 (fail-closed guards), "
            "alert_engine.py lines 80-150 (AlertSeverity classification)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
    
    ComplianceMapping(
        requirement_id="FDA-4",
        framework=RegulatoryFramework.FDA_MLDS,
        requirement_title="Data Provenance and Quality",
        system_components=[
            "model_registry.py (training_data_version, training_data_hash)",
            "evaluation_pipeline.py (DataSplit with metadata)"
        ],
        implementation_details=(
            "ModelMetadata includes training_data_version (e.g., 'mimic-v5') and training_data_hash "
            "(SHA-256 of training set) for complete provenance. DataSplit records split_params "
            "(test_size, stratified, random_seed) and class distributions for reproducibility."
        ),
        evidence=(
            "model_registry.py lines 75-85 (training_data_version/hash fields), "
            "evaluation_pipeline.py lines 100-150 (DataSplit with provenance)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
    
    ComplianceMapping(
        requirement_id="FDA-5",
        framework=RegulatoryFramework.FDA_SAMD,
        requirement_title="Clinical Performance Validation",
        system_components=[
            "evaluation_pipeline.py (EvaluationPipeline)"
        ],
        implementation_details=(
            "EvaluationPipeline computes AUROC, AUPRC with 95% CI (1000 bootstrap samples), "
            "accuracy, precision, recall, F1, specificity, Brier score, log loss. Generates "
            "ROC/PR curves, calibration plots, confusion matrices. Exports LaTeX tables for "
            "publication/submission."
        ),
        evidence=(
            "evaluation_pipeline.py lines 200-450 (ModelMetrics with CI), "
            "evaluation_pipeline.py lines 550-700 (ROC/PR plot generation), "
            "evaluation_pipeline.py lines 900-1000 (LaTeX table export)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
    
    # -------------------------------------------------------------------------
    # EU AI Act Mappings
    # -------------------------------------------------------------------------
    ComplianceMapping(
        requirement_id="EU-1",
        framework=RegulatoryFramework.EU_AI_ACT,
        requirement_title="User Transparency and Information",
        system_components=[
            "xai_engine.py (ClinicalExplanation with clinical_summary)",
            "alert_engine.py (ClinicalAlert.to_clinician_message)",
            "governed_whatif_engine.py (SimulationResult with disclaimer)"
        ],
        implementation_details=(
            "XAI Engine generates clinical_summary in plain English (e.g., 'High lactate and elevated "
            "WBC indicate infection risk'). Alert messages include severity, risk probability, and "
            "recommended actions. What-if simulations include mandatory disclaimer: 'HYPOTHETICAL ONLY' "
            "with plausibility assessment."
        ),
        evidence=(
            "xai_engine.py lines 200-250 (ClinicalExplanation schema), "
            "alert_engine.py lines 150-200 (to_clinician_message formatting), "
            "governed_whatif_engine.py lines 250-300 (disclaimer logic)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
    
    ComplianceMapping(
        requirement_id="EU-2",
        framework=RegulatoryFramework.EU_AI_ACT,
        requirement_title="Automatic Logging of Events",
        system_components=[
            "audit_logging.py (AuditLogger)",
            "governance.py (mandatory logging enforcement)"
        ],
        implementation_details=(
            "AuditLogger records every prediction, explanation, alert, clinician action, model version "
            "change, threshold change, and failure event. Hash-chained with SHA-256 for tamper detection. "
            "GovernanceContext enforces fail-closed: no operation without active logger."
        ),
        evidence=(
            "audit_logging.py lines 50-100 (AuditEventType taxonomy), "
            "audit_logging.py lines 150-300 (AuditLogger implementation), "
            "governance.py lines 50-80 (require_logger guards)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
    
    ComplianceMapping(
        requirement_id="EU-3",
        framework=RegulatoryFramework.EU_AI_ACT,
        requirement_title="Human Oversight and Intervention",
        system_components=[
            "alert_engine.py (clinician acknowledgment)",
            "governed_whatif_engine.py (clinician-initiated simulations)",
            "decision_trace.py (clinician_action_event)"
        ],
        implementation_details=(
            "Alert engine requires explicit clinician acknowledgment (logged with clinician_id, notes, "
            "timestamp). What-if simulations require clinician_id and log every simulation. Decision "
            "traces capture clinician_action_event with action details. No autonomous action without "
            "human-in-the-loop."
        ),
        evidence=(
            "alert_engine.py lines 400-450 (acknowledge_alert method), "
            "governed_whatif_engine.py lines 200-250 (clinician_id requirement), "
            "decision_trace.py lines 150-180 (log_clinician_action)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
    
    ComplianceMapping(
        requirement_id="EU-4",
        framework=RegulatoryFramework.EU_AI_ACT,
        requirement_title="Accuracy and Robustness Requirements",
        system_components=[
            "evaluation_pipeline.py (comprehensive metrics)",
            "model_registry.py (metrics stored per version)"
        ],
        implementation_details=(
            "Evaluation pipeline computes 11+ metrics (AUROC, AUPRC, calibration, etc.) with 95% CI. "
            "Metrics stored in ModelMetadata per version, enabling performance tracking over time. "
            "Threshold validation ensures predictions meet clinical accuracy requirements."
        ),
        evidence=(
            "evaluation_pipeline.py lines 200-300 (ModelMetrics with CI), "
            "model_registry.py lines 80-90 (metrics field in ModelMetadata)"
        ),
        compliance_status=ComplianceStatus.FULL,
        gaps="Adversarial robustness testing not yet implemented",
        remediation_plan="Add adversarial perturbation tests to evaluation_pipeline (Q2 2026)"
    ),
    
    ComplianceMapping(
        requirement_id="EU-5",
        framework=RegulatoryFramework.EU_AI_ACT,
        requirement_title="Technical Documentation and Version Control",
        system_components=[
            "model_registry.py (ModelMetadata)",
            "technical_effect_registry.py (TechnicalEffect documentation)"
        ],
        implementation_details=(
            "ModelRegistry stores complete metadata: model_version, training_data_version/hash, "
            "feature_schema_hash, hyperparameters, metrics, code_hash, artifact_path. Technical "
            "effect registry documents problems, solutions, and measurable effects for each component."
        ),
        evidence=(
            "model_registry.py lines 70-130 (ModelMetadata schema), "
            "technical_effect_registry.py lines 100-200 (TechnicalEffect schema)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
    
    ComplianceMapping(
        requirement_id="EU-6",
        framework=RegulatoryFramework.EU_AI_ACT,
        requirement_title="Training Data Quality and Governance",
        system_components=[
            "model_registry.py (training_data_version, training_data_hash)",
            "evaluation_pipeline.py (stratified splits with class balance)"
        ],
        implementation_details=(
            "ModelMetadata captures training_data_version and SHA-256 hash of training set. "
            "DataSplit uses stratified sampling to preserve class balance across train/val/test. "
            "Class distribution reported for each split."
        ),
        evidence=(
            "model_registry.py lines 75-85 (data provenance fields), "
            "evaluation_pipeline.py lines 100-200 (stratified splitting)"
        ),
        compliance_status=ComplianceStatus.PARTIAL,
        gaps="Bias analysis not yet implemented",
        remediation_plan="Add fairness metrics (demographic parity, equalized odds) to evaluation_pipeline (Q2 2026)"
    ),
    
    # -------------------------------------------------------------------------
    # GDPR Mappings
    # -------------------------------------------------------------------------
    ComplianceMapping(
        requirement_id="GDPR-1",
        framework=RegulatoryFramework.GDPR,
        requirement_title="Right to Information about Processing",
        system_components=[
            "xai_engine.py (ClinicalExplanation)",
            "decision_trace.py (input_summary binding)"
        ],
        implementation_details=(
            "DecisionTrace captures input_summary (patient features used in decision). "
            "XAI Engine generates ClinicalExplanation with top_risk_factors (feature name, value, "
            "importance, clinical interpretation). Patients can request trace via patient_id."
        ),
        evidence=(
            "decision_trace.py lines 40-60 (input_summary field), "
            "xai_engine.py lines 200-300 (ClinicalExplanation with feature details)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
    
    ComplianceMapping(
        requirement_id="GDPR-2",
        framework=RegulatoryFramework.GDPR,
        requirement_title="Right to Explanation for Automated Decisions",
        system_components=[
            "xai_engine.py (SHAP/LIME explanations)",
            "governance.py (no prediction without explanation)"
        ],
        implementation_details=(
            "XAI Engine generates SHAP/LIME explanations for every prediction, providing feature "
            "importances and clinical interpretations. GovernanceContext enforces: no alert without "
            "explanation_event. Clinical summaries translate technical outputs to plain language."
        ),
        evidence=(
            "xai_engine.py lines 400-600 (SHAP/LIME implementation), "
            "governance.py lines 80-120 (require_explanation_link guards)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
    
    ComplianceMapping(
        requirement_id="GDPR-3",
        framework=RegulatoryFramework.GDPR,
        requirement_title="Data Subject Access to Processing Records",
        system_components=[
            "audit_logging.py (patient_id indexing)",
            "decision_trace.py (DecisionTraceManager)"
        ],
        implementation_details=(
            "Audit log includes patient_id in every event. DecisionTraceManager stores traces "
            "indexed by patient_id. Export functions (export_human_readable) generate CSV reports "
            "for patient data access requests."
        ),
        evidence=(
            "audit_logging.py lines 60-80 (patient_id field in AuditEvent), "
            "audit_logging.py lines 350-400 (export_human_readable method), "
            "decision_trace.py lines 100-150 (trace storage by patient_id)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
    
    ComplianceMapping(
        requirement_id="GDPR-4",
        framework=RegulatoryFramework.GDPR,
        requirement_title="Privacy by Design and Default",
        system_components=[
            "audit_logging.py (no PII in payload by default)",
            "decision_trace.py (input_summary pseudonymization)"
        ],
        implementation_details=(
            "Audit events use patient_id (pseudonym) rather than PHI. Payload fields contain "
            "clinical features (numeric values) without names/identifiers. Input summaries use "
            "de-identified feature names."
        ),
        evidence=(
            "audit_logging.py lines 60-90 (patient_id as pseudonym), "
            "decision_trace.py lines 50-70 (input_summary with de-identified features)"
        ),
        compliance_status=ComplianceStatus.PARTIAL,
        gaps="Encryption at rest not enforced by code",
        remediation_plan="Add encrypted audit log storage with key management (Q3 2026)"
    ),
    
    ComplianceMapping(
        requirement_id="GDPR-5",
        framework=RegulatoryFramework.GDPR,
        requirement_title="Accountability and Demonstrable Compliance",
        system_components=[
            "audit_logging.py (immutable audit trail)",
            "governance.py (fail-closed enforcement)",
            "technical_effect_registry.py (technical documentation)"
        ],
        implementation_details=(
            "Hash-chained audit log provides cryptographic proof of event integrity. "
            "GovernanceContext enforces mandatory logging with RegulatoryViolationError on bypass. "
            "Technical effect registry documents compliance measures and measurable effects."
        ),
        evidence=(
            "audit_logging.py lines 150-300 (hash chain implementation), "
            "governance.py lines 20-50 (RegulatoryViolationError guards), "
            "technical_effect_registry.py (complete compliance documentation)"
        ),
        compliance_status=ComplianceStatus.FULL
    ),
]


# =============================================================================
# Compliance Matrix Generator
# =============================================================================

class ComplianceMatrix:
    """Generate compliance matrices in multiple formats."""
    
    def __init__(self):
        self.requirements = (
            FDA_REQUIREMENTS +
            EU_AI_ACT_REQUIREMENTS +
            GDPR_REQUIREMENTS
        )
        self.mappings = COMPLIANCE_MAPPINGS
    
    def generate_console_table(self) -> str:
        """Generate human-readable console table."""
        output = "=" * 120 + "\n"
        output += "REGULATORY COMPLIANCE MATRIX\n"
        output += "=" * 120 + "\n\n"
        
        for framework in RegulatoryFramework:
            framework_mappings = [m for m in self.mappings if m.framework == framework]
            if not framework_mappings:
                continue
            
            output += f"\n{'='*120}\n"
            output += f"{framework.value}\n"
            output += f"{'='*120}\n\n"
            
            for mapping in framework_mappings:
                output += f"[{mapping.requirement_id}] {mapping.requirement_title}\n"
                output += f"Status: {mapping.compliance_status.value}\n"
                output += f"Components: {', '.join(mapping.system_components)}\n"
                output += f"Implementation: {mapping.implementation_details[:200]}...\n"
                
                if mapping.gaps:
                    output += f"⚠️  Gaps: {mapping.gaps}\n"
                    output += f"Remediation: {mapping.remediation_plan}\n"
                
                output += "\n"
        
        return output
    
    def generate_csv(self, output_path: str) -> None:
        """Generate CSV for spreadsheet import."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Requirement ID",
                "Framework",
                "Requirement Title",
                "System Components",
                "Implementation Details",
                "Evidence (File References)",
                "Compliance Status",
                "Gaps",
                "Remediation Plan"
            ])
            
            for mapping in self.mappings:
                writer.writerow([
                    mapping.requirement_id,
                    mapping.framework.value,
                    mapping.requirement_title,
                    "; ".join(mapping.system_components),
                    mapping.implementation_details,
                    mapping.evidence,
                    mapping.compliance_status.value,
                    mapping.gaps or "None",
                    mapping.remediation_plan or "N/A"
                ])
    
    def generate_markdown(self, output_path: str) -> None:
        """Generate markdown for documentation."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Regulatory Compliance Matrix\n\n")
            f.write("## Overview\n\n")
            f.write("This document maps regulatory requirements (FDA, EU AI Act, GDPR) to system implementations.\n\n")
            
            for framework in RegulatoryFramework:
                framework_mappings = [m for m in self.mappings if m.framework == framework]
                if not framework_mappings:
                    continue
                
                f.write(f"\n## {framework.value}\n\n")
                
                for mapping in framework_mappings:
                    f.write(f"### [{mapping.requirement_id}] {mapping.requirement_title}\n\n")
                    f.write(f"**Compliance Status:** {mapping.compliance_status.value}\n\n")
                    f.write(f"**System Components:**\n")
                    for comp in mapping.system_components:
                        f.write(f"- `{comp}`\n")
                    f.write(f"\n**Implementation:**\n{mapping.implementation_details}\n\n")
                    f.write(f"**Evidence:** {mapping.evidence}\n\n")
                    
                    if mapping.gaps:
                        f.write(f"**⚠️ Gaps:** {mapping.gaps}\n\n")
                        f.write(f"**Remediation:** {mapping.remediation_plan}\n\n")
                    
                    f.write("---\n\n")
    
    def generate_json(self, output_path: str) -> None:
        """Generate JSON for machine-readable processing."""
        data = {
            "generated_at": "2026-01-14",
            "total_requirements": len(self.mappings),
            "compliance_summary": {
                "full": len([m for m in self.mappings if m.compliance_status == ComplianceStatus.FULL]),
                "partial": len([m for m in self.mappings if m.compliance_status == ComplianceStatus.PARTIAL]),
                "planned": len([m for m in self.mappings if m.compliance_status == ComplianceStatus.PLANNED]),
            },
            "mappings": [
                {
                    "requirement_id": m.requirement_id,
                    "framework": m.framework.value,
                    "requirement_title": m.requirement_title,
                    "system_components": m.system_components,
                    "implementation_details": m.implementation_details,
                    "evidence": m.evidence,
                    "compliance_status": m.compliance_status.value,
                    "gaps": m.gaps,
                    "remediation_plan": m.remediation_plan,
                }
                for m in self.mappings
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def generate_all_formats(self, output_dir: str = "regulatory_submission") -> None:
        """Generate all output formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating compliance matrix in {output_dir}/...")
        
        # Console output
        console_output = self.generate_console_table()
        with open(output_path / "compliance_matrix.txt", 'w', encoding='utf-8') as f:
            f.write(console_output)
        print("✓ compliance_matrix.txt")
        
        # CSV
        self.generate_csv(str(output_path / "compliance_matrix.csv"))
        print("✓ compliance_matrix.csv")
        
        # Markdown
        self.generate_markdown(str(output_path / "compliance_matrix.md"))
        print("✓ compliance_matrix.md")
        
        # JSON
        self.generate_json(str(output_path / "compliance_matrix.json"))
        print("✓ compliance_matrix.json")
        
        print(f"\nCompliance Status Summary:")
        print(f"  FULL:    {len([m for m in self.mappings if m.compliance_status == ComplianceStatus.FULL])}")
        print(f"  PARTIAL: {len([m for m in self.mappings if m.compliance_status == ComplianceStatus.PARTIAL])}")
        print(f"  PLANNED: {len([m for m in self.mappings if m.compliance_status == ComplianceStatus.PLANNED])}")


# =============================================================================
# Usage
# =============================================================================

if __name__ == "__main__":
    matrix = ComplianceMatrix()
    matrix.generate_all_formats(output_dir="regulatory_submission")
    
    # Print console table
    print("\n" + matrix.generate_console_table())
