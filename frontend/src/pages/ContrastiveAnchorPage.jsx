import React, { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import '../styles/Clinical.css';

const DEFAULT_BASELINE_PATIENT = {
  patientId: '',
  age: 45,
  gender: 0,
  heart_rate: 72,
  systolic_bp: 120,
  diastolic_bp: 80,
  temperature: 98.6,
  respiratory_rate: 16,
  wbc_count: 7.5,
  hemoglobin: 14.0,
  platelet_count: 250,
  creatinine: 1.0,
  bun: 15,
  glucose: 95,
  lactate: 1.2,
};

const FEATURE_LABELS = {
  age: 'Age',
  gender: 'Gender',
  heart_rate: 'Heart Rate',
  systolic_bp: 'Systolic BP',
  diastolic_bp: 'Diastolic BP',
  temperature: 'Temperature',
  respiratory_rate: 'Respiratory Rate',
  wbc_count: 'WBC Count',
  hemoglobin: 'Hemoglobin',
  platelet_count: 'Platelet Count',
  creatinine: 'Creatinine',
  bun: 'BUN',
  glucose: 'Glucose',
  lactate: 'Lactate',
};

const DISEASE_PRIORITIES = {
  sepsis: ['lactate', 'temperature', 'wbc_count', 'systolic_bp', 'heart_rate', 'respiratory_rate'],
  kidney_failure: ['creatinine', 'bun', 'systolic_bp', 'lactate'],
  diabetes: ['glucose', 'systolic_bp', 'creatinine'],
  anemia: ['hemoglobin', 'platelet_count', 'creatinine'],
  thrombocytopenia: ['platelet_count', 'hemoglobin'],
  hypertension: ['systolic_bp', 'diastolic_bp', 'heart_rate', 'creatinine'],
  mortality: ['lactate', 'systolic_bp', 'creatinine', 'heart_rate', 'respiratory_rate'],
};

const FEATURE_ORDER = [
  'lactate',
  'temperature',
  'wbc_count',
  'systolic_bp',
  'heart_rate',
  'respiratory_rate',
  'creatinine',
  'bun',
  'glucose',
  'hemoglobin',
  'platelet_count',
  'diastolic_bp',
  'age',
  'gender',
];

const formatPercent = (value) => `${(Number(value) * 100).toFixed(1)}%`;

const formatFeature = (value) => (FEATURE_LABELS[value] || value).replace(/_/g, ' ');

const ContrastiveAnchorPage = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const baselinePatient = useMemo(() => {
    const routeState = location.state || {};
    return {
      ...DEFAULT_BASELINE_PATIENT,
      ...(routeState.baselinePatient || {}),
    };
  }, [location.state]);

  const availableDiseases = useMemo(
    () => location.state?.availableDiseases || [],
    [location.state]
  );

  const diseases = availableDiseases.length > 0 ? availableDiseases : Object.keys(DISEASE_PRIORITIES);

  const [selectedDisease, setSelectedDisease] = useState(diseases[0] || 'sepsis');
  const [focusFeature, setFocusFeature] = useState('lactate');
  const [topK, setTopK] = useState(5);
  const [contextSamples, setContextSamples] = useState(24);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    const prioritized = DISEASE_PRIORITIES[selectedDisease] || FEATURE_ORDER;
    if (prioritized.length > 0) {
      setFocusFeature(prioritized[0]);
    }
  }, [selectedDisease]);

  const focusOptions = useMemo(() => {
    const prioritized = DISEASE_PRIORITIES[selectedDisease] || FEATURE_ORDER;
    const merged = [
      ...prioritized,
      ...FEATURE_ORDER.filter((feature) => !prioritized.includes(feature)),
    ];
    return merged;
  }, [selectedDisease]);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('access_token');
      const prioritized = DISEASE_PRIORITIES[selectedDisease] || FEATURE_ORDER;
      const orderedFocus = [focusFeature, ...prioritized.filter((feature) => feature !== focusFeature)];

      const response = await fetch('/api/contrastive-anchor', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          patient_id: baselinePatient.patientId || null,
          baseline_features: baselinePatient,
          disease: selectedDisease,
          focus_features: orderedFocus,
          top_k: Number(topK),
          context_samples: Number(contextSamples),
        }),
      });

      if (!response.ok) {
        throw new Error('Contrastive anchor analysis failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to generate contrastive anchor explanation');
    } finally {
      setLoading(false);
    }
  };

  const activeBoundaries = result?.clinical_boundaries || {};

  return (
    <div className="clinical-app">
      <div className="clinical-header">
        <div className="header-content">
          <div className="header-title">
            <div className="header-icon">🧭</div>
            <div>
              <h1>Contrastive Anchor Explanation</h1>
              <div className="header-subtitle">Bounded local explanation with plausibility and stability scoring</div>
            </div>
          </div>
          <div className="user-info">
            <div className="user-badge">
              {user?.full_name ? `Dr. ${user.full_name}` : user?.username || 'Clinician'}
            </div>
            <button className="btn-logout" onClick={handleLogout}>Logout</button>
          </div>
        </div>
      </div>

      <div className="clinical-container">
        <div className="card" style={{ marginBottom: '1rem' }}>
          <div className="btn-group">
            <button className="btn btn-secondary" onClick={() => navigate('/clinical')}>
              <span>←</span><span>Back to Dashboard</span>
            </button>
            <button className="btn btn-secondary" onClick={() => navigate('/clinical/what-if', {
              state: {
                baselinePatient,
                availableDiseases: diseases,
              },
            })}>
              <span>🔄</span><span>Open What-If Analysis</span>
            </button>
            <button className="btn btn-secondary" onClick={() => navigate('/clinical/constrained-response-surface', {
              state: {
                baselinePatient,
                availableDiseases: diseases,
              },
            })}>
              <span>📈</span><span>Open Response Surface</span>
            </button>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title"><span>🧭</span> Contrastive Anchor Controls</h2>
            <p>Choose a disease and feature focus, then generate a clinically bounded local anchor explanation.</p>
          </div>

          <div className="patient-info-grid">
            <div className="info-field">
              <label>Disease to Explain</label>
              <select value={selectedDisease} onChange={(e) => setSelectedDisease(e.target.value)}>
                {diseases.map((disease) => (
                  <option key={disease} value={disease}>
                    {disease.replace(/_/g, ' ').toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <div className="info-field">
              <label>Focus Feature</label>
              <select value={focusFeature} onChange={(e) => setFocusFeature(e.target.value)}>
                {focusOptions.map((feature) => (
                  <option key={feature} value={feature}>
                    {formatFeature(feature)}
                  </option>
                ))}
              </select>
            </div>

            <div className="info-field">
              <label>Top Features</label>
              <input
                type="number"
                min="1"
                max="8"
                step="1"
                value={topK}
                onChange={(e) => setTopK(Math.max(1, Math.min(8, Number(e.target.value) || 1)))}
              />
            </div>

            <div className="info-field">
              <label>Local Context Samples</label>
              <input
                type="number"
                min="8"
                max="64"
                step="1"
                value={contextSamples}
                onChange={(e) => setContextSamples(Math.max(8, Math.min(64, Number(e.target.value) || 8)))}
              />
            </div>
          </div>

          <div className="btn-group">
            <button className="btn btn-primary" onClick={handleAnalyze} disabled={loading}>
              {loading ? (
                <><span className="spinner"></span><span>Generating...</span></>
              ) : (
                <><span>🧭</span><span>Generate Anchor Explanation</span></>
              )}
            </button>
          </div>
        </div>

        {result && (
          <>
            <div className="card" style={{ background: 'var(--bg)', marginBottom: '1rem' }}>
              <div className="card-header">
                <h2 className="card-title"><span>📍</span> Anchor Summary</h2>
              </div>
              <div className="risk-summary">
                <div className="risk-card low-risk">
                  <div className="risk-card-header">Baseline Risk</div>
                  <div className="risk-score">{formatPercent(result.baseline_risk)}</div>
                  <div className="risk-label">{result.risk_category}</div>
                </div>
                <div className="risk-card medium-risk">
                  <div className="risk-card-header">Representative Anchor Risk</div>
                  <div className="risk-score">{formatPercent(result.representative_anchor_risk)}</div>
                  <div className="risk-label">{result.anchor_strategy}</div>
                </div>
                <div className="risk-card high-risk">
                  <div className="risk-card-header">Risk Shift</div>
                  <div className="risk-score">{result.risk_delta > 0 ? '+' : ''}{formatPercent(result.risk_delta)}</div>
                  <div className="risk-label">{result.confidence >= 0.75 ? 'High confidence' : 'Moderate confidence'}</div>
                </div>
              </div>

              <div style={{ marginTop: '0.9rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem' }}>
                <div>
                  <strong>Confidence:</strong> {formatPercent(result.confidence)}
                </div>
                <div>
                  <strong>Context samples:</strong> {result.local_context_size}
                </div>
                <div>
                  <strong>Focus features:</strong> {result.focus_features.join(', ')}
                </div>
                <div>
                  <strong>Relative shift:</strong> {result.risk_delta_percent > 0 ? '+' : ''}{result.risk_delta_percent.toFixed(1)}%
                </div>
              </div>
            </div>

            <div className="card" style={{ background: 'var(--bg)', marginBottom: '1rem' }}>
              <h4 style={{ marginBottom: '0.75rem' }}>Clinical Summary</h4>
              <div style={{ lineHeight: 1.7 }}>{result.clinical_summary}</div>
            </div>

            <div className="card" style={{ background: 'var(--bg)', marginBottom: '1rem' }}>
              <h4 style={{ marginBottom: '0.75rem' }}>Feature-Level Anchor Results</h4>
              <div className="feature-list">
                {result.feature_contributions.map((item, idx) => {
                  const boundary = activeBoundaries[item.feature_name] || {};
                  return (
                    <div key={`${item.feature_name}-${idx}`} className="feature-item" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                      <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
                        <div className="feature-name">
                          {idx + 1}. {formatFeature(item.feature_name)}
                        </div>
                        <div style={{ fontWeight: 700, color: item.importance >= 0 ? 'var(--secondary)' : 'var(--danger)' }}>
                          {item.importance >= 0 ? '+' : ''}{item.importance.toFixed(3)}
                        </div>
                      </div>

                      <div style={{ marginTop: '0.4rem', fontSize: '0.92rem' }}>
                        Current: <strong>{item.current_value.toFixed(2)}</strong> {'→'} Anchor: <strong>{item.anchor_value.toFixed(2)}</strong>
                      </div>
                      <div style={{ marginTop: '0.35rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                        Baseline risk {formatPercent(item.baseline_risk)} | Anchor risk {formatPercent(item.anchor_risk)} | Delta {item.risk_delta > 0 ? '+' : ''}{formatPercent(item.risk_delta)}
                      </div>
                      <div style={{ marginTop: '0.35rem', fontSize: '0.9rem' }}>
                        Stability {formatPercent(item.stability_score)} | Plausibility {formatPercent(item.plausibility_score)} | {item.constraint_status.replace(/_/g, ' ')}
                      </div>
                      <div style={{ marginTop: '0.35rem', fontSize: '0.9rem' }}>
                        {item.recommendation}
                      </div>
                      <div style={{ marginTop: '0.35rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        Bounds: {boundary.absolute_low !== undefined ? `${boundary.absolute_low.toFixed(2)} - ${boundary.absolute_high.toFixed(2)}` : 'n/a'}
                        {boundary.preferred_low !== undefined ? ` | Preferred: ${boundary.preferred_low.toFixed(2)} - ${boundary.preferred_high.toFixed(2)}` : ''}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {result.clinical_recommendations?.length > 0 && (
              <div className="card" style={{ background: 'var(--bg)' }}>
                <h4 style={{ marginBottom: '0.75rem' }}>Clinical Recommendations</h4>
                <ul style={{ marginLeft: '1.25rem', color: 'var(--text)', lineHeight: 1.8 }}>
                  {result.clinical_recommendations.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ContrastiveAnchorPage;
