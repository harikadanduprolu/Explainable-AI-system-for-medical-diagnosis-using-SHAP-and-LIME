import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Landing.css';

const stats = [
  { label: 'Matched Patients', value: '12,000+', note: 'Clinical + CXR pairs' },
  { label: 'Derived Features', value: '30+', note: 'Ratios, scores & flags' },
  { label: 'Monitored Diseases', value: '9', note: 'Sepsis through mortality' },
  { label: 'Explainability Modes', value: '3', note: 'SHAP, LIME, Multimodal' },
];

const featureHighlights = [
  {
    title: 'Multimodal Fusion',
    description:
      'Blends MIMIC-IV vitals/labs with MIMIC-CXR imaging evidence, producing agreement indices and governance alerts.',
  },
  {
    title: 'Clinician-Grade Explainability',
    description:
      'Top contributing features, what-if controls, and risk thresholds make every model decision auditable.',
  },
  {
    title: 'Regulatory Guardrails',
    description:
      'Audit logging, model registry metadata, and stratified validation follow FDA Software as a Medical Device expectations.',
  },
];

const workflow = [
  'Ingest MIMIC-IV v3.1 admissions, vitals, labs, and CheXpert-labeled CXR metadata.',
  'Engineer 30+ derived clinical markers (SIRS, SOFA-like, renal burden, hemodynamic ratios).',
  'Train disease-specific XGBoost ensembles with governance metadata and calibration metrics.',
  'Serve models through FastAPI with multimodal fusion + explainability endpoints.',
  'Deliver decisions via the React clinician workspace with cohort search, forms, and dashboards.',
];

function LandingPage() {
  return (
    <div className="landing">
      <header className="landing-hero">
        <div className="hero-content">
          <p className="hero-pill">Explainable Medical AI Platform</p>
          <h1>
            Predict multi-system risks with
            <span> transparent, multimodal intelligence.</span>
          </h1>
          <p className="hero-subtitle">
            Built on MIMIC-IV v3.1 and MIMIC-CXR-JPG, the system unifies structured EHR data,
            chest radiographs, and regulatory-grade observability so clinicians can trust every recommendation.
          </p>
          <div className="hero-cta">
            <Link className="btn-primary" to="/predict">
              Live Prediction Workspace
            </Link>
            <Link className="btn-secondary" to="/login">
              Clinician Login
            </Link>
          </div>
          <p className="hero-note">No PHI required. Use the sample cohort or plug in your own values.</p>
        </div>
        <div className="hero-card">
          <h3>Key Signals Tracked</h3>
          <ul>
            <li>Hemodynamic instability (shock index, MAP, pulse pressure)</li>
            <li>Renal burden (creatinine/BUN, eGFR proxies, renal flags)</li>
            <li>Metabolic stress (lactate ratios, glucose dynamics)</li>
            <li>Inflammation markers (SIRS, SOFA-like indicators)</li>
          </ul>
          <span className="card-footer">View detailed feature values in the prediction workspace.</span>
        </div>
      </header>

      <section className="landing-stats">
        {stats.map((stat) => (
          <div key={stat.label} className="stat-card">
            <div className="stat-value">{stat.value}</div>
            <div className="stat-label">{stat.label}</div>
            <div className="stat-note">{stat.note}</div>
          </div>
        ))}
      </section>

      <section className="landing-features">
        <div className="section-header">
          <h2>Why clinicians adopt this stack</h2>
          <p>
            Each layer—from ingestion to dashboard—was engineered for transparency, reproducibility, and rapid
            experimentation.
          </p>
        </div>
        <div className="feature-grid">
          {featureHighlights.map((feature) => (
            <div key={feature.title} className="feature-card">
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="landing-workflow">
        <div className="section-header">
          <h2>Workflow at a glance</h2>
          <p>Five steps tie together governed datasets, modeling, and clinician-facing web tooling.</p>
        </div>
        <ol className="workflow-list">
          {workflow.map((step, idx) => (
            <li key={step}>
              <span className="workflow-index">{idx + 1}</span>
              <p>{step}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="landing-cta">
        <div className="cta-card">
          <h2>Ready to explore?</h2>
          <p>
            Head to the prediction workspace for instant experimentation or sign in to unlock the clinician dashboard
            with longitudinal tracking, annotations, and deployment checklists.
          </p>
          <div className="hero-cta">
            <Link className="btn-primary" to="/predict">
              Try Predictions
            </Link>
            <Link className="btn-secondary" to="/clinical">
              Open Clinician Dashboard
            </Link>
          </div>
        </div>
      </section>

      <footer className="landing-footer">
        <div>© {new Date().getFullYear()} Explainable Medical AI System</div>
        <div className="footer-links">
          <a href="https://physionet.org/content/mimiciv/3.1/" target="_blank" rel="noreferrer">
            MIMIC-IV
          </a>
          <a href="https://physionet.org/content/mimic-cxr-jpg/2.1.0/" target="_blank" rel="noreferrer">
            MIMIC-CXR-JPG
          </a>
          <Link to="/predict">Prediction Workspace</Link>
        </div>
      </footer>
    </div>
  );
}

export default LandingPage;
