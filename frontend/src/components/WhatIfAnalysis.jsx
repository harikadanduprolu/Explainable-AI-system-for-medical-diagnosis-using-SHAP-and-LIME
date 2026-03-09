import React, { useState } from 'react';

function WhatIfAnalysis({ baselinePatient }) {
  const [selectedFeature, setSelectedFeature] = useState('temperature');
  const [newValue, setNewValue] = useState('');
  const [selectedDisease, setSelectedDisease] = useState('sepsis');
  const [whatIfResult, setWhatIfResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const features = [
    { key: 'temperature', label: 'Temperature (°F)' },
    { key: 'heart_rate', label: 'Heart Rate (bpm)' },
    { key: 'lactate', label: 'Lactate (mmol/L)' },
    { key: 'wbc_count', label: 'WBC Count (K/µL)' },
    { key: 'creatinine', label: 'Creatinine (mg/dL)' },
    { key: 'glucose', label: 'Glucose (mg/dL)' },
    { key: 'systolic_bp', label: 'Systolic BP (mmHg)' },
  ];

  const diseases = [
    'sepsis',
    'kidney_failure',
    'heart_disease',
    'diabetes',
    'anemia',
    'mortality',
  ];

  const handleAnalyze = async () => {
    if (!newValue) {
      alert('Please enter a new value');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/whatif', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          baseline_features: baselinePatient,
          modified_features: {
            [selectedFeature]: parseFloat(newValue),
          },
          disease: selectedDisease,
        }),
      });

      if (!response.ok) {
        throw new Error('What-if analysis failed');
      }

      const data = await response.json();
      setWhatIfResult(data);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to perform what-if analysis');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>🔄 What-If Scenario Analysis</h2>
        <p>Explore how changing clinical parameters affects disease risk</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
        <div className="form-group">
          <label>Disease to Analyze</label>
          <select
            className="form-group input"
            style={{ padding: '0.75rem', borderRadius: '8px', border: '2px solid var(--border)' }}
            value={selectedDisease}
            onChange={(e) => setSelectedDisease(e.target.value)}
          >
            {diseases.map((d) => (
              <option key={d} value={d}>
                {d.replace(/_/g, ' ').toUpperCase()}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Feature to Modify</label>
          <select
            className="form-group input"
            style={{ padding: '0.75rem', borderRadius: '8px', border: '2px solid var(--border)' }}
            value={selectedFeature}
            onChange={(e) => setSelectedFeature(e.target.value)}
          >
            {features.map((f) => (
              <option key={f.key} value={f.key}>
                {f.label}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>
            Current Value: <strong>{baselinePatient[selectedFeature]}</strong>
          </label>
          <input
            type="number"
            placeholder="Enter new value"
            value={newValue}
            onChange={(e) => setNewValue(e.target.value)}
            style={{ padding: '0.75rem', borderRadius: '8px', border: '2px solid var(--border)' }}
            step="0.1"
          />
        </div>
      </div>

      <button onClick={handleAnalyze} className="btn btn-primary" disabled={loading}>
        {loading ? 'Analyzing...' : '🔍 Analyze Impact'}
      </button>

      {whatIfResult && (
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>Analysis Results</h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
            <div className="card" style={{ background: 'var(--bg)' }}>
              <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                Baseline Risk
              </div>
              <div style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--primary)' }}>
                {(whatIfResult.baseline_risk * 100).toFixed(1)}%
              </div>
            </div>

            <div className="card" style={{ background: 'var(--bg)' }}>
              <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                New Risk
              </div>
              <div style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--primary)' }}>
                {(whatIfResult.new_risk * 100).toFixed(1)}%
              </div>
            </div>

            <div className="card" style={{ background: 'var(--bg)' }}>
              <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                Risk Change
              </div>
              <div
                style={{
                  fontSize: '2rem',
                  fontWeight: '700',
                  color: whatIfResult.risk_delta < 0 ? 'var(--secondary)' : 'var(--danger)',
                }}
              >
                {whatIfResult.risk_delta > 0 ? '+' : ''}
                {(whatIfResult.risk_delta * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div
            className={`alert ${whatIfResult.risk_delta < 0 ? 'alert-success' : 'alert-warning'}`}
          >
            <strong>{whatIfResult.recommendation}</strong>
          </div>

          <div className="card" style={{ background: 'var(--bg)', marginTop: '1rem' }}>
            <h4 style={{ marginBottom: '0.75rem' }}>Modified Features</h4>
            {Object.entries(whatIfResult.modified_features).map(([feature, values]) => (
              <div key={feature} style={{ marginBottom: '0.5rem' }}>
                <strong>{feature.replace(/_/g, ' ')}:</strong>{' '}
                <span style={{ color: 'var(--text-secondary)' }}>{values.old}</span>
                {' → '}
                <span style={{ color: 'var(--primary)', fontWeight: '600' }}>{values.new}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default WhatIfAnalysis;
