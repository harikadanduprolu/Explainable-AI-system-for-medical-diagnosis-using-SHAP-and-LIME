"""
Audit Logging Subsystem for Regulated Medical AI
================================================

Purpose
-------
Provides append-only, hash-chained audit logs that link predictions,
explanations, alerts, thresholds, clinician actions, and model versions.
Designed for FDA SaMD/MLDS (logging and traceability), EU AI Act (high-risk
logging, transparency, post-market monitoring), and Indian patent/medical
device requirements (technical effect and provenance).

Regulatory Justification
------------------------
- FDA SaMD/MLDS: Post-market surveillance, traceability of algorithm changes,
  and reconstruction of clinical decisions require immutable audit trails.
- EU AI Act (Title III, high-risk): Mandatory logging, traceability, and
  human oversight; this module links AI outputs, explanations, thresholds,
  and clinician interventions with integrity checks.
- Patent (IN/US/EU): Demonstrates a technical effect (secure, hash-chained,
  explainability-linked audit pipeline) that enables reproducibility and
  accountability; suitable for claims around regulated AI safety controls.

Key Properties
--------------
- Append-only JSONL with SHA-256 hash chaining (no silent overwrites).
- Dual outputs: machine-readable (JSONL) and human-readable summaries.
- Links: prediction_id ↔ explanation_id ↔ alert_id ↔ clinician_action_id.
- Includes model version, threshold in force, data hash, and actor identity.
- Integrity verification via chain validation; tamper-evident.

Usage (abridged)
----------------
    logger = AuditLogger(log_path="logs/audit.log", system_id="ED-AI-01")
    event = logger.log_event(
        event_type=AuditEventType.PREDICTION,
        patient_id="P123",
        disease="Sepsis",
        model_name="xgb_sepsis",
        model_version="1.3.2",
        prediction_id="pred-001",
        payload={"probability": 0.82, "threshold": 0.35},
        human_message="Sepsis risk predicted at 0.82 with threshold 0.35"
    )
    logger.verify_chain()  # Raises if tampering is detected

File Outputs
------------
- audit.log (JSONL, append-only, hash-chained)
- audit_summary.csv (optional human-readable export)

Failure Modes & Safeguards
--------------------------
- Tampering: Detected via hash chain; verify_chain() raises on mismatch.
- Partial writes: fsync on append; optional atomic rename (see _atomic_append).
- Clock skew: Timestamps include UTC and monotonic counter to preserve order.
- Concurrent writes: File locks (fcntl on POSIX, msvcrt on Windows) to prevent
  interleaving.

"""

from __future__ import annotations

import json
import os
import uuid
import hashlib
import time
import csv
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

try:
    import msvcrt  # Windows file locking
except ImportError:  # pragma: no cover
    msvcrt = None

try:
    import fcntl  # POSIX file locking
except ImportError:  # pragma: no cover
    fcntl = None


# =============================================================================
# Event Taxonomy
# =============================================================================

class AuditEventType(str, Enum):
    """Taxonomy of audit events for regulated clinical AI."""
    PREDICTION = "prediction"               # Model inference output
    EXPLANATION = "explanation"             # SHAP/LIME/Grad-CAM output
    ALERT = "alert"                         # Threshold-based clinical alert
    CLINICIAN_ACTION = "clinician_action"   # Acknowledgement, override, note
    MODEL_VERSION = "model_version"         # Model promotion/change
    DATA_INGEST = "data_ingest"             # Data load or preprocessing
    THRESHOLD_CHANGE = "threshold_change"   # Policy or threshold update
    EVALUATION = "evaluation"               # Offline eval outputs (AUROC etc.)
    FAILURE = "failure"                     # Errors, degraded mode


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AuditEvent:
    """Immutable audit event with hash chaining for tamper evidence."""
    event_id: str
    event_type: AuditEventType
    timestamp_utc: str
    monotonic_seq: int
    system_id: str
    actor_type: str  # system / clinician / admin
    actor_id: str
    patient_id: Optional[str]
    disease: Optional[str]
    model_name: Optional[str]
    model_version: Optional[str]
    prediction_id: Optional[str]
    explanation_id: Optional[str]
    alert_id: Optional[str]
    clinician_action_id: Optional[str]
    threshold: Optional[float]
    payload: Dict[str, Any]
    human_message: str
    data_hash: Optional[str]
    prev_hash: str
    record_hash: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)


# =============================================================================
# Audit Logger
# =============================================================================

class AuditLogger:
    """
    Append-only, hash-chained audit logger for regulated medical AI.

    Supports:
    - Immutable event recording with SHA-256 hash chaining
    - Reconstruction of decision pathways (prediction ↔ explanation ↔ alert ↔ action)
    - Dual outputs: JSONL (machine) and CSV summaries (human)
    - Chain verification for tamper evidence

    Regulatory mapping:
    - FDA SaMD/MLDS: Post-market surveillance, traceability of algorithm and
      threshold changes; reproducibility of decisions with context and actors.
    - EU AI Act: High-risk systems must log events enabling oversight,
      transparency, and incident investigation; this module captures each
      lifecycle event with integrity metadata.
    - Patent (IN/US/EU): Provides technical effect via integrity-preserving
      audit linking explainability artifacts to clinical actions.
    """

    def __init__(
        self,
        log_path: str,
        system_id: str,
        actor_id: str = "system",
        actor_type: str = "system",
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.system_id = system_id
        self.actor_id = actor_id
        self.actor_type = actor_type
        self._monotonic_seq = 0
        self._ensure_log_file()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_event(
        self,
        event_type: AuditEventType,
        payload: Dict[str, Any],
        human_message: str,
        patient_id: Optional[str] = None,
        disease: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        prediction_id: Optional[str] = None,
        explanation_id: Optional[str] = None,
        alert_id: Optional[str] = None,
        clinician_action_id: Optional[str] = None,
        threshold: Optional[float] = None,
        data_hash: Optional[str] = None,
    ) -> AuditEvent:
        """
        Append a new audit event with hash chaining.

        Returns the AuditEvent object (useful for propagating IDs).
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        monotonic_seq = self._next_seq()
        prev_hash = self._get_last_hash()

        event_id = str(uuid.uuid4())
        record_base = {
            "event_id": event_id,
            "event_type": event_type.value,
            "timestamp_utc": timestamp_utc,
            "monotonic_seq": monotonic_seq,
            "system_id": self.system_id,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "patient_id": patient_id,
            "disease": disease,
            "model_name": model_name,
            "model_version": model_version,
            "prediction_id": prediction_id,
            "explanation_id": explanation_id,
            "alert_id": alert_id,
            "clinician_action_id": clinician_action_id,
            "threshold": threshold,
            "payload": payload,
            "human_message": human_message,
            "data_hash": data_hash,
            "prev_hash": prev_hash,
        }

        record_hash = self._compute_hash(record_base)

        event = AuditEvent(
            record_hash=record_hash,
            **record_base,
        )

        self._atomic_append(event.to_json())
        return event

    def verify_chain(self) -> None:
        """Verify the entire log for tampering. Raises ValueError on failure."""
        last_hash = ""
        with self.log_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                obj = json.loads(line)
                expected_prev = obj.get("prev_hash", "")
                if expected_prev != last_hash:
                    raise ValueError(
                        f"Hash chain break at line {line_number}: expected prev_hash={last_hash}, got {expected_prev}"
                    )
                record_hash = obj.get("record_hash", "")
                computed = self._compute_hash({k: obj[k] for k in obj if k != "record_hash"})
                if record_hash != computed:
                    raise ValueError(
                        f"Record hash mismatch at line {line_number}: expected {record_hash}, computed {computed}"
                    )
                last_hash = record_hash

    def export_human_readable(self, output_csv: str) -> None:
        """Export a human-readable CSV summary for legal/regulator review."""
        with self.log_path.open("r", encoding="utf-8") as f, open(output_csv, "w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow([
                "event_id",
                "timestamp_utc",
                "event_type",
                "patient_id",
                "disease",
                "model_name",
                "model_version",
                "prediction_id",
                "explanation_id",
                "alert_id",
                "clinician_action_id",
                "threshold",
                "actor_type",
                "actor_id",
                "human_message",
            ])
            for line in f:
                obj = json.loads(line)
                writer.writerow([
                    obj.get("event_id"),
                    obj.get("timestamp_utc"),
                    obj.get("event_type"),
                    obj.get("patient_id"),
                    obj.get("disease"),
                    obj.get("model_name"),
                    obj.get("model_version"),
                    obj.get("prediction_id"),
                    obj.get("explanation_id"),
                    obj.get("alert_id"),
                    obj.get("clinician_action_id"),
                    obj.get("threshold"),
                    obj.get("actor_type"),
                    obj.get("actor_id"),
                    obj.get("human_message"),
                ])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_log_file(self) -> None:
        if not self.log_path.exists():
            self.log_path.touch()

    def _next_seq(self) -> int:
        self._monotonic_seq += 1
        return self._monotonic_seq

    def _get_last_hash(self) -> str:
        if not self.log_path.exists():
            return ""
        with self.log_path.open("rb") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode("utf-8")
        if not last_line:
            return ""
        try:
            obj = json.loads(last_line)
            return obj.get("record_hash", "")
        except json.JSONDecodeError:
            return ""

    def _compute_hash(self, record: Dict[str, Any]) -> str:
        record_json = json.dumps(record, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(record_json.encode("utf-8")).hexdigest()

    def _atomic_append(self, line: str) -> None:
        # Ensure line ends with newline
        line = line.rstrip("\n") + "\n"
        with self._file_lock(self.log_path) as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    # ------------------------------------------------------------------
    # File locking (Windows/POSIX)
    # ------------------------------------------------------------------

    def _file_lock(self, path: Path):
        return _FileLock(path)


class _FileLock:
    """Context manager for cross-platform file locking."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None

    def __enter__(self):
        self.file = open(self.path, "a+")
        if msvcrt:
            msvcrt.locking(self.file.fileno(), msvcrt.LK_LOCK, 1)
        elif fcntl:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            if msvcrt:
                try:
                    self.file.seek(0)
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            elif fcntl:
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()
        return False


# =============================================================================
# Example Usage and Sample Records
# =============================================================================

if __name__ == "__main__":
    logger = AuditLogger(log_path="logs/audit.log", system_id="ED-AI-01")

    # Prediction event
    pred_evt = logger.log_event(
        event_type=AuditEventType.PREDICTION,
        patient_id="P123",
        disease="Sepsis",
        model_name="xgb_sepsis",
        model_version="1.3.2",
        prediction_id="pred-001",
        payload={"probability": 0.82, "threshold": 0.35},
        human_message="Sepsis risk predicted at 0.82 with threshold 0.35"
    )

    # Explanation linked to prediction
    expl_evt = logger.log_event(
        event_type=AuditEventType.EXPLANATION,
        patient_id="P123",
        disease="Sepsis",
        model_name="xgb_sepsis",
        model_version="1.3.2",
        prediction_id=pred_evt.event_id,
        explanation_id="expl-001",
        payload={"top_features": ["lactate", "wbc", "hr"], "method": "shap"},
        human_message="SHAP explanation generated for prediction pred-001"
    )

    # Alert triggered
    alert_evt = logger.log_event(
        event_type=AuditEventType.ALERT,
        patient_id="P123",
        disease="Sepsis",
        model_name="xgb_sepsis",
        model_version="1.3.2",
        prediction_id=pred_evt.event_id,
        explanation_id=expl_evt.event_id,
        alert_id="alert-001",
        threshold=0.35,
        payload={"alert": "High sepsis risk", "probability": 0.82},
        human_message="Alert issued to charge nurse"
    )

    # Clinician action
    logger.log_event(
        event_type=AuditEventType.CLINICIAN_ACTION,
        patient_id="P123",
        disease="Sepsis",
        model_name="xgb_sepsis",
        model_version="1.3.2",
        prediction_id=pred_evt.event_id,
        explanation_id=expl_evt.event_id,
        alert_id=alert_evt.event_id,
        clinician_action_id="action-001",
        payload={"action": "Acknowledged", "notes": "Started fluids"},
        human_message="Clinician acknowledged alert and initiated fluids"
    )

    logger.verify_chain()
    logger.export_human_readable("logs/audit_summary.csv")
    print("Audit log created and verified.")
