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

const ConstrainedResponseSurfacePage = () => {
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
  const [numPoints, setNumPoints] = useState(11);
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

      const response = await fetch('/api/constrained-response-surface', {
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
          num_points: Number(numPoints),
        }),
      });

      if (!response.ok) {
        throw new Error('Constrained response-surface analysis failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to generate constrained response-surface explanation');
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
            <div className="header-icon">📈</div>
            <div>
              <h1>Constrained Response Surface</h1>
              <div className="header-subtitle">Model-agnostic bounded response curves for clinically plausible interventions</div>
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
            <button className="btn btn-secondary" onClick={() => navigate('/clinical/contrastive-anchor', {
              state: {
                baselinePatient,
                availableDiseases: diseases,
              },
            })}>
              <span>🧭</span><span>Open Contrastive Anchor</span>
            </button>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title"><span>📈</span> Response Surface Controls</h2>
            <p>Choose a disease and feature focus, then sample bounded local curves to measure intervention sensitivity.</p>
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
              <label>Curve Sample Points</label>
              <input
                type="number"
                min="7"
                max="31"
                step="2"
                value={numPoints}
                onChange={(e) => {
                  const raw = Number(e.target.value) || 7;
                  const clipped = Math.max(7, Math.min(31, raw));
                  const odd = clipped % 2 === 0 ? clipped + 1 : clipped;
                  setNumPoints(Math.min(31, odd));
                }}
              />
            </div>
          </div>

          <div className="btn-group">
            <button className="btn btn-primary" onClick={handleAnalyze} disabled={loading}>
              {loading ? (
                <><span className="spinner"></span><span>Generating...</span></>
              ) : (
                <><span>📈</span><span>Generate Response Surface</span></>
              )}
            </button>
          </div>
        </div>

        {result && (
          <>
            <div className="card" style={{ background: 'var(--bg)', marginBottom: '1rem' }}>
              <div className="card-header">
                <h2 className="card-title"><span>📊</span> Surface Summary</h2>
              </div>
              <div className="risk-summary">
                <div className="risk-card low-risk">
                  <div className="risk-card-header">Baseline Risk</div>
                  <div className="risk-score">{formatPercent(result.baseline_risk)}</div>
                  <div className="risk-label">{result.risk_category}</div>
                </div>
                <div className="risk-card medium-risk">
                  <div className="risk-card-header">Best Local Bounded Risk</div>
                  <div className="risk-score">{formatPercent(result.representative_best_risk)}</div>
                  <div className="risk-label">{result.analysis_strategy}</div>
                </div>
                <div className="risk-card high-risk">
                  <div className="risk-card-header">Potential Reduction</div>
                  <div className="risk-score">{result.risk_delta > 0 ? '+' : ''}{formatPercent(result.risk_delta)}</div>
                  <div className="risk-label">{result.confidence >= 0.75 ? 'High confidence' : 'Moderate confidence'}</div>
                </div>
              </div>

              <div style={{ marginTop: '0.9rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem' }}>
                <div>
                  <strong>Confidence:</strong> {formatPercent(result.confidence)}
                </div>
                <div>
                  <strong>Points per curve:</strong> {result.local_context_size}
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
              <h4 style={{ marginBottom: '0.75rem' }}>Feature-Level Response Curves</h4>
              <div className="feature-list">
                {result.feature_surfaces.map((item, idx) => {
                  const boundary = activeBoundaries[item.feature_name] || {};
                  const previewPoints = item.sample_points?.slice(0, 6) || [];
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
                        Current: <strong>{item.current_value.toFixed(2)}</strong> {'→'} Best: <strong>{item.best_value.toFixed(2)}</strong> ({item.suggested_direction})
                      </div>
                      <div style={{ marginTop: '0.35rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                        Baseline risk {formatPercent(item.baseline_risk)} | Best risk {formatPercent(item.best_risk)} | Delta {item.risk_delta > 0 ? '+' : ''}{formatPercent(item.risk_delta)}
                      </div>
                      <div style={{ marginTop: '0.35rem', fontSize: '0.9rem' }}>
                        Confidence {formatPercent(item.confidence)} | Monotonicity {formatPercent(item.monotonicity)} | Nonlinearity {item.nonlinearity.toFixed(3)}
                      </div>
                      <div style={{ marginTop: '0.35rem', fontSize: '0.9rem' }}>
                        Response area {item.response_area.toFixed(3)} | Elasticity mean {item.elasticity_mean.toFixed(3)} | Elasticity max {item.elasticity_max.toFixed(3)}
                      </div>
                      <div style={{ marginTop: '0.35rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        Bounds: {boundary.absolute_low !== undefined ? `${boundary.absolute_low.toFixed(2)} - ${boundary.absolute_high.toFixed(2)}` : 'n/a'}
                        {boundary.preferred_low !== undefined ? ` | Preferred: ${boundary.preferred_low.toFixed(2)} - ${boundary.preferred_high.toFixed(2)}` : ''}
                      </div>
                      {previewPoints.length > 0 && (
                        <div style={{ marginTop: '0.45rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                          Curve preview: {previewPoints.map((p) => `${p.feature_value.toFixed(2)}→${(p.risk_score * 100).toFixed(1)}%`).join(' | ')}
                        </div>
                      )}
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

export default ConstrainedResponseSurfacePage;
