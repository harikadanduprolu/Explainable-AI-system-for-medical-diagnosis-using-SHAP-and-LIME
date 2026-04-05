import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import '../styles/Clinical.css';

const ClinicalDashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [patientData, setPatientData] = useState({
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
    lactate: 1.2
  });

  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [clinicalNotes, setClinicalNotes] = useState('');
  const [alert, setAlert] = useState(null);

  const samplePatients = [
    {
      name: "Preset: Healthy Patient",
      data: { age: 45, gender: 0, heart_rate: 72, systolic_bp: 120, diastolic_bp: 80, temperature: 98.6, respiratory_rate: 16, wbc_count: 7.5, hemoglobin: 14.0, platelet_count: 250, creatinine: 1.0, bun: 15, glucose: 95, lactate: 1.2 }
    },
    {
      name: "Preset: High Sepsis Risk",
      data: { age: 68, gender: 1, heart_rate: 115, systolic_bp: 95, diastolic_bp: 65, temperature: 101.5, respiratory_rate: 24, wbc_count: 16.5, hemoglobin: 10.5, platelet_count: 150, creatinine: 2.1, bun: 40, glucose: 180, lactate: 3.2 }
    },
    {
      name: "Preset: Kidney Failure Risk",
      data: { age: 72, gender: 1, heart_rate: 88, systolic_bp: 135, diastolic_bp: 85, temperature: 98.4, respiratory_rate: 18, wbc_count: 8.2, hemoglobin: 9.5, platelet_count: 220, creatinine: 4.5, bun: 85, glucose: 110, lactate: 1.8 }
    },
    {
      name: "Preset: Diabetes Risk",
      data: { age: 55, gender: 0, heart_rate: 82, systolic_bp: 140, diastolic_bp: 90, temperature: 98.7, respiratory_rate: 17, wbc_count: 7.8, hemoglobin: 12.5, platelet_count: 280, creatinine: 1.2, bun: 18, glucose: 325, lactate: 1.5 }
    }
  ];

  const diseaseNames = {
    sepsis: 'Sepsis',
    kidney_failure: 'Kidney Failure',
    diabetes: 'Hyperglycemia Risk',
    anemia: 'Anemia',
    thrombocytopenia: 'Thrombocytopenia',
    hypertension: 'Hypertension Risk',
    mortality: 'Mortality Risk'
  };

  const featureNames = {
    age: 'Age', gender: 'Gender', heart_rate: 'Heart Rate',
    systolic_bp: 'Systolic BP', diastolic_bp: 'Diastolic BP',
    temperature: 'Temperature', respiratory_rate: 'Respiratory Rate',
    wbc_count: 'WBC Count', hemoglobin: 'Hemoglobin',
    platelet_count: 'Platelet Count', creatinine: 'Creatinine',
    bun: 'Blood Urea Nitrogen', glucose: 'Glucose', lactate: 'Lactate'
  };

  const handleInputChange = (field, value) => {
    setPatientData({ ...patientData, [field]: parseFloat(value) || value });
  };

  const loadSample = (index) => {
    setPatientData({ ...patientData, ...samplePatients[index].data });
    showAlert(`Loaded: ${samplePatients[index].name}`, 'success');
  };

  const analyzePrediction = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          patient_id: patientData.patientId || null,
          features: { ...patientData, patientId: undefined }
        })
      });

      if (!response.ok) throw new Error('Analysis failed');

      const data = await response.json();
      setPredictions(data);
      showAlert('Analysis complete', 'success');
      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (error) {
      showAlert(error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const clearForm = () => {
    loadSample(0);
    setPredictions(null);
    setClinicalNotes('');
  };

  const showAlert = (message, type) => {
    setAlert({ message, type });
    setTimeout(() => setAlert(null), 5000);
  };

  const getRiskClass = (category) => {
    if (category === 'CRITICAL' || category === 'HIGH') return 'high-risk';
    if (category === 'MODERATE') return 'medium-risk';
    return 'low-risk';
  };

  const getRiskColor = (category) => {
    if (category === 'CRITICAL' || category === 'HIGH') return '#dc3545';
    if (category === 'MODERATE') return '#ffc107';
    return '#28a745';
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const openWhatIfAnalysis = () => {
    navigate('/clinical/what-if', {
      state: {
        baselinePatient: { ...patientData },
        availableDiseases: predictions?.predictions?.map((pred) => pred.disease) || [],
      },
    });
  };

  const openContrastiveAnchorView = () => {
    navigate('/clinical/contrastive-anchor', {
      state: {
        baselinePatient: { ...patientData },
        availableDiseases: predictions?.predictions?.map((pred) => pred.disease) || [],
      },
    });
  };

  const openConstrainedResponseSurfaceView = () => {
    navigate('/clinical/constrained-response-surface', {
      state: {
        baselinePatient: { ...patientData },
        availableDiseases: predictions?.predictions?.map((pred) => pred.disease) || [],
      },
    });
  };

  const renderRecommendations = (text) => {
    if (!text) return null;

    const sections = text.split('###').filter(Boolean);

    return sections.map((section, index) => {
      const [title, ...content] = section.split('\n');
      return (
        <div key={index} className="rec-section" style={{ marginBottom: '0.85rem' }}>
          <h4 className="rec-title" style={{ marginBottom: '0.5rem' }}>🩺 {title.trim()}</h4>
          <ul className="rec-list" style={{ margin: 0, paddingLeft: '1rem' }}>
            {content
              .filter((line) => line.trim() !== '')
              .map((line, idx) => (
                <li key={idx}>{line.replace(/^[-*]\s?/, '').trim()}</li>
              ))}
          </ul>
        </div>
      );
    });
  };

  return (
    <div className="clinical-app">
      {/* Header */}
      <div className="clinical-header">
        <div className="header-content">
          <div className="header-title">
            <div className="header-icon">🏥</div>
            <div>
              <h1>Clinical Decision Support System</h1>
              <div className="header-subtitle">AI-Powered Multi-Disease Risk Assessment</div>
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

      {/* Main Container */}
      <div className="clinical-container">
        {/* Alert */}
        {alert && (
          <div className={`alert alert-${alert.type}`}>
            {alert.message}
          </div>
        )}

        {/* Patient Information */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title"><span>👤</span> Patient Information</h2>
            <div className="sample-patients">
              {samplePatients.map((sample, idx) => (
                <button key={idx} className="sample-btn" onClick={() => loadSample(idx)}>
                  {sample.name}
                </button>
              ))}
            </div>
          </div>
          <div className="patient-info-grid">
            <div className="info-field">
              <label>Patient ID</label>
              <input type="text" value={patientData.patientId} onChange={(e) => handleInputChange('patientId', e.target.value)} placeholder="Enter Patient ID" />
            </div>
            <div className="info-field">
              <label>Age (years)</label>
              <input type="number" value={patientData.age} onChange={(e) => handleInputChange('age', e.target.value)} min="18" max="100" />
            </div>
            <div className="info-field">
              <label>Gender</label>
              <select value={patientData.gender} onChange={(e) => handleInputChange('gender', e.target.value)}>
                <option value="0">Female</option>
                <option value="1">Male</option>
              </select>
            </div>
          </div>
        </div>

        {/* Vital Signs & Labs */}
        <div className="two-column-layout">
          {/* Vital Signs */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title"><span>💓</span> Vital Signs</h2>
            </div>
            <div className="vitals-grid">
              <div className="vital-input">
                <label>Heart Rate</label>
                <input type="number" value={patientData.heart_rate} onChange={(e) => handleInputChange('heart_rate', e.target.value)} step="1" />
                <span className="unit">bpm</span>
              </div>
              <div className="vital-input">
                <label>Systolic BP</label>
                <input type="number" value={patientData.systolic_bp} onChange={(e) => handleInputChange('systolic_bp', e.target.value)} step="1" />
                <span className="unit">mmHg</span>
              </div>
              <div className="vital-input">
                <label>Diastolic BP</label>
                <input type="number" value={patientData.diastolic_bp} onChange={(e) => handleInputChange('diastolic_bp', e.target.value)} step="1" />
                <span className="unit">mmHg</span>
              </div>
              <div className="vital-input">
                <label>Temperature</label>
                <input type="number" value={patientData.temperature} onChange={(e) => handleInputChange('temperature', e.target.value)} step="0.1" />
                <span className="unit">°F</span>
              </div>
              <div className="vital-input">
                <label>Respiratory Rate</label>
                <input type="number" value={patientData.respiratory_rate} onChange={(e) => handleInputChange('respiratory_rate', e.target.value)} step="1" />
                <span className="unit">/min</span>
              </div>
            </div>
          </div>

          {/* Laboratory Values */}
          <div className="card">
            <div className="card-header">
              <h2 className="card-title"><span>🔬</span> Laboratory Values</h2>
            </div>
            <div className="vitals-grid">
              <div className="vital-input">
                <label>WBC Count</label>
                <input type="number" value={patientData.wbc_count} onChange={(e) => handleInputChange('wbc_count', e.target.value)} step="0.1" />
                <span className="unit">K/µL</span>
              </div>
              <div className="vital-input">
                <label>Hemoglobin</label>
                <input type="number" value={patientData.hemoglobin} onChange={(e) => handleInputChange('hemoglobin', e.target.value)} step="0.1" />
                <span className="unit">g/dL</span>
              </div>
              <div className="vital-input">
                <label>Platelet Count</label>
                <input type="number" value={patientData.platelet_count} onChange={(e) => handleInputChange('platelet_count', e.target.value)} step="1" />
                <span className="unit">K/µL</span>
              </div>
              <div className="vital-input">
                <label>Creatinine</label>
                <input type="number" value={patientData.creatinine} onChange={(e) => handleInputChange('creatinine', e.target.value)} step="0.1" />
                <span className="unit">mg/dL</span>
              </div>
              <div className="vital-input">
                <label>BUN</label>
                <input type="number" value={patientData.bun} onChange={(e) => handleInputChange('bun', e.target.value)} step="1" />
                <span className="unit">mg/dL</span>
              </div>
              <div className="vital-input">
                <label>Glucose</label>
                <input type="number" value={patientData.glucose} onChange={(e) => handleInputChange('glucose', e.target.value)} step="1" />
                <span className="unit">mg/dL</span>
              </div>
              <div className="vital-input">
                <label>Lactate</label>
                <input type="number" value={patientData.lactate} onChange={(e) => handleInputChange('lactate', e.target.value)} step="0.1" />
                <span className="unit">mmol/L</span>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="card">
          <div className="btn-group">
            <button className="btn btn-primary" onClick={analyzePrediction} disabled={loading}>
              {loading ? (
                <><span className="spinner"></span><span>Analyzing...</span></>
              ) : (
                <><span>🔍</span><span>Analyze Patient</span></>
              )}
            </button>
            <button className="btn btn-secondary" onClick={clearForm}>
              <span>🔄</span><span>Clear Form</span>
            </button>
            <button className="btn btn-secondary" onClick={openWhatIfAnalysis}>
              <span>🔄</span><span>Open What-If Analysis</span>
            </button>
            <button className="btn btn-secondary" onClick={openContrastiveAnchorView}>
              <span>🧭</span><span>Contrastive Anchor View</span>
            </button>
            <button className="btn btn-secondary" onClick={openConstrainedResponseSurfaceView}>
              <span>📈</span><span>Response Surface View</span>
            </button>
          </div>
        </div>

        {/* Results Section */}
        {predictions && (
          <div id="results-section">
            {/* AI-Generated Recommendations */}
            {predictions?.clinical_recommendations && (
              <div
                className="card"
                style={{
                  marginBottom: '1.5rem',
                  padding: '1rem',
                  backgroundColor: 'rgba(76, 175, 80, 0.1)',
                  borderRadius: '8px',
                  borderLeft: '4px solid #4CAF50'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '1.2rem', marginRight: '0.5rem' }}>🤖</span>
                  <h3 style={{ margin: 0 }}>AI-Generated Recommendations</h3>
                </div>

                <div
                  style={{
                    backgroundColor: 'white',
                    padding: '1rem',
                    borderRadius: '6px',
                    maxHeight: '350px',
                    overflowY: 'auto',
                    fontSize: '0.95rem',
                    lineHeight: '1.6'
                  }}
                >
                  {renderRecommendations(
                    predictions.clinical_recommendations.recommendations
                  )}
                </div>

                {predictions.clinical_recommendations.high_risk_diseases?.length > 0 && (
                  <div
                    style={{
                      marginTop: '0.75rem',
                      fontSize: '0.9rem',
                      color: '#dc3545'
                    }}
                  >
                    <strong>⚠️ High Risk:</strong>{' '}
                    {predictions.clinical_recommendations.high_risk_diseases.join(', ')}
                  </div>
                )}
              </div>
            )}

            {/* Risk Summary */}
            <div className="card">
              <div className="card-header">
                <h2 className="card-title"><span>📊</span> Risk Assessment Summary</h2>
                <button className="btn btn-export" onClick={() => window.print()}>
                  <span>📄</span><span>Export Report</span>
                </button>
              </div>
              <div className="risk-summary">
                {predictions.predictions.map((pred, idx) => (
                  <div key={idx} className={`risk-card ${getRiskClass(pred.risk_category)}`}>
                    <div className="risk-card-header">{diseaseNames[pred.disease]}</div>
                    <div className="risk-score">{(pred.risk_score * 100).toFixed(1)}%</div>
                    <div className="risk-label">{pred.risk_category}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Detailed Analysis */}
            <div className="card">
              <div className="card-header">
                <h2 className="card-title"><span>🔬</span> Detailed Clinical Analysis</h2>
              </div>
              <div className="tabs">
                {predictions.predictions.map((pred, idx) => (
                  <button key={idx} className={`tab ${activeTab === idx ? 'active' : ''}`} onClick={() => setActiveTab(idx)}>
                    {diseaseNames[pred.disease]}
                  </button>
                ))}
              </div>
              {predictions.predictions.map((pred, idx) => (
                <div key={idx} className={`tab-content ${activeTab === idx ? 'active' : ''}`}>
                  <div style={{ marginBottom: '20px' }}>
                    <strong>Model Type:</strong> {pred.model_type}<br />
                    <strong>Prediction Threshold:</strong> {(pred.threshold * 100).toFixed(1)}%<br />
                    <strong>Risk Score:</strong> {(pred.risk_score * 100).toFixed(2)}%<br />
                    <strong>Risk Category:</strong> <span style={{ color: getRiskColor(pred.risk_category), fontWeight: 600 }}>{pred.risk_category}</span>
                  </div>
                  <h3 style={{ marginBottom: '15px', fontSize: '16px' }}>Top Contributing Factors</h3>
                  <div className="feature-list">
                    {pred.top_features.slice(0, 5).map((feature, fidx) => (
                      <div key={fidx} className="feature-item">
                        <span className="feature-name">{featureNames[feature.feature_name]}</span>
                        <div className="feature-value">
                          <div className="feature-bar">
                            <div className="feature-bar-fill" style={{ width: `${Math.abs(feature.importance) * 100}%` }}></div>
                          </div>
                          <span className="feature-score">
                            {feature.importance > 0 ? '+' : ''}{feature.importance.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Clinical Notes */}
            <div className="card">
              <div className="card-header">
                <h2 className="card-title"><span>📝</span> Clinical Notes & Recommendations</h2>
              </div>
              <div className="clinical-notes">
                <textarea value={clinicalNotes} onChange={(e) => setClinicalNotes(e.target.value)} placeholder="Enter clinical notes, observations, and treatment recommendations..." />
              </div>
              <div className="btn-group">
                <button className="btn btn-primary" onClick={() => { localStorage.setItem('last_clinical_notes', clinicalNotes); showAlert('Notes saved', 'success'); }}>
                  <span>💾</span><span>Save Notes</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ClinicalDashboard;
