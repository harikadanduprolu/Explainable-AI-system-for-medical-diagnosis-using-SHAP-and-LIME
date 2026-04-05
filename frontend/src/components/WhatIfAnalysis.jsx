import React, { useEffect, useMemo, useState } from 'react';

const FEATURE_GUIDANCE = {
  temperature: {
    increase: 'Higher temperature is often associated with stronger inflammatory/infectious burden, which can raise sepsis-related risk.',
    decrease: 'Lowering fever may reduce physiologic stress and can improve risk when infection control is in progress.',
    actions: 'Review infection source control, antipyretic strategy, and serial lactate/WBC trends.'
  },
  heart_rate: {
    increase: 'Rising heart rate can signal hemodynamic stress, pain, hypovolemia, or worsening systemic illness.',
    decrease: 'Reducing tachycardia may indicate improved perfusion, volume status, or symptom control.',
    actions: 'Reassess volume status, pain/anxiety drivers, oxygenation, and rhythm abnormalities.'
  },
  lactate: {
    increase: 'Higher lactate generally reflects worse tissue hypoperfusion and often increases acute risk estimates.',
    decrease: 'Lactate clearance is typically favorable and may reduce short-term deterioration risk.',
    actions: 'Track lactate clearance trajectory, optimize perfusion, and escalate if persistent elevation remains.'
  },
  wbc_count: {
    increase: 'WBC elevation can suggest active inflammation/infection and may shift risk upward.',
    decrease: 'Normalization can indicate improving inflammatory response and treatment effect.',
    actions: 'Correlate with infection markers, culture results, and current antimicrobial strategy.'
  },
  creatinine: {
    increase: 'Rising creatinine indicates worsening renal function and may increase kidney/mortality risk.',
    decrease: 'Creatinine improvement is usually favorable for renal and global risk profiles.',
    actions: 'Review nephrotoxic exposures, fluid balance, urine output, and renal support thresholds.'
  },
  glucose: {
    increase: 'Hyperglycemia is linked to stress physiology and can worsen outcomes in high-risk patients.',
    decrease: 'Better glycemic control may improve metabolic stability and reduce risk contribution.',
    actions: 'Use protocolized glucose management and monitor for overcorrection/hypoglycemia.'
  },
  systolic_bp: {
    increase: 'Improved systolic pressure may indicate better perfusion support in hypotensive states.',
    decrease: 'Dropping systolic pressure can indicate hemodynamic compromise and increased acute risk.',
    actions: 'Check MAP/perfusion endpoints, vasopressor needs, and fluid responsiveness.'
  }
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

const TARGET_RANGES = {
  temperature: { low: 97.0, high: 99.0 },
  heart_rate: { low: 60.0, high: 100.0 },
  systolic_bp: { low: 100.0, high: 140.0 },
  respiratory_rate: { low: 12.0, high: 20.0 },
  wbc_count: { low: 4.0, high: 11.0 },
  hemoglobin: { low: 12.0, high: 16.0 },
  platelet_count: { low: 150.0, high: 400.0 },
  creatinine: { low: 0.6, high: 1.2 },
  bun: { low: 7.0, high: 20.0 },
  glucose: { low: 70.0, high: 140.0 },
  lactate: { low: 0.8, high: 2.0 },
};

const FEATURE_ACTIONS = {
  anemia: {
    hemoglobin: 'Increase hemoglobin and evaluate iron stores/supplementation plan.',
    platelet_count: 'Support marrow/hemostatic stability and reassess bleeding risk.',
  },
  sepsis: {
    lactate: 'Improve perfusion and monitor lactate clearance.',
    temperature: 'Control infection burden and fever trajectory.',
    wbc_count: 'Reassess inflammatory response and antimicrobial strategy.',
  },
  kidney_failure: {
    creatinine: 'Reduce kidney stress and avoid nephrotoxic exposure.',
    bun: 'Optimize renal support and monitor uremic burden.',
  },
};

const makeTargetValue = (feature, currentValue) => {
  const range = TARGET_RANGES[feature];
  if (!range) return currentValue;
  if (currentValue < range.low) return range.low;
  if (currentValue > range.high) return range.high;
  return currentValue;
};

const buildCounterfactualPlan = (disease, baselinePatient) => {
  const priorities = DISEASE_PRIORITIES[disease] || [];
  return priorities
    .map((feature) => {
      const currentValue = Number(baselinePatient?.[feature]);
      if (Number.isNaN(currentValue)) {
        return null;
      }

      const targetValue = makeTargetValue(feature, currentValue);
      if (targetValue === currentValue) {
        return null;
      }

      const direction = targetValue > currentValue ? 'Increase' : 'Decrease';
      const action = FEATURE_ACTIONS[disease]?.[feature] || `Move ${feature.replace(/_/g, ' ')} toward a clinically preferred range.`;

      return {
        feature,
        currentValue,
        targetValue,
        direction,
        action,
      };
    })
    .filter(Boolean)
    .slice(0, 3);
};

function WhatIfAnalysis({ baselinePatient, availableDiseases = [] }) {
  const [selectedFeature, setSelectedFeature] = useState('temperature');
  const [newValue, setNewValue] = useState('');
  const fallbackDiseases = [
    'sepsis',
    'kidney_failure',
    'diabetes',
    'anemia',
    'thrombocytopenia',
    'hypertension',
    'mortality',
  ];
  const diseases = availableDiseases.length > 0 ? availableDiseases : fallbackDiseases;
  const [selectedDisease, setSelectedDisease] = useState(diseases[0] || 'sepsis');
  const [whatIfResult, setWhatIfResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const suggestedCounterfactuals = useMemo(
    () => buildCounterfactualPlan(selectedDisease, baselinePatient),
    [selectedDisease, baselinePatient]
  );

  const features = [
    { key: 'temperature', label: 'Temperature (°F)' },
    { key: 'heart_rate', label: 'Heart Rate (bpm)' },
    { key: 'lactate', label: 'Lactate (mmol/L)' },
    { key: 'wbc_count', label: 'WBC Count (K/µL)' },
    { key: 'creatinine', label: 'Creatinine (mg/dL)' },
    { key: 'glucose', label: 'Glucose (mg/dL)' },
    { key: 'systolic_bp', label: 'Systolic BP (mmHg)' },
  ];

  const baselineValue = Number(baselinePatient[selectedFeature]);
  const proposedValue = newValue === '' ? null : Number(newValue);
  const delta = proposedValue === null ? null : proposedValue - baselineValue;
  const isIncrease = delta !== null && delta > 0;
  const isDecrease = delta !== null && delta < 0;
  const guidance = FEATURE_GUIDANCE[selectedFeature];

  const formatRisk = (value) => `${(Number(value) * 100).toFixed(1)}%`;

  useEffect(() => {
    const defaultSuggestion = suggestedCounterfactuals[0];
    if (defaultSuggestion) {
      setSelectedFeature(defaultSuggestion.feature);
      setNewValue(String(defaultSuggestion.targetValue));
    }
  }, [selectedDisease, baselinePatient, suggestedCounterfactuals]);

  const handleAnalyze = async () => {
    if (!newValue) {
      alert('Please enter a new value');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/what-if', {
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
            Current Value: <strong>{baselineValue}</strong>
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

      <div className="card" style={{ background: 'var(--bg)', marginBottom: '1rem' }}>
        <h4 style={{ marginBottom: '0.5rem' }}>Clinical Preview (Before Running Simulation)</h4>
        <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
          Disease focus: <strong>{selectedDisease.replace(/_/g, ' ')}</strong>
        </div>
        {suggestedCounterfactuals.length > 0 && (
          <div style={{ marginBottom: '1rem' }}>
            <div style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.5rem' }}>
              Suggested counterfactual plan
            </div>
            <div className="feature-list">
              {suggestedCounterfactuals.map((item, idx) => (
                <div key={item.feature} className="feature-item" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                  <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
                    <div className="feature-name">
                      {idx + 1}. {item.feature.replace(/_/g, ' ')}: {item.direction} to {item.targetValue.toFixed(2)}
                    </div>
                    <div style={{ fontWeight: 600, color: 'var(--secondary)' }}>
                      recommended
                    </div>
                  </div>
                  <div style={{ marginTop: '0.35rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    {item.action}
                  </div>
                  <div style={{ marginTop: '0.35rem', fontSize: '0.9rem' }}>
                    If {item.feature.replace(/_/g, ' ')} changes from {item.currentValue.toFixed(2)} to {item.targetValue.toFixed(2)}, the model is expected to move in the safer direction.
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {proposedValue === null ? (
          <div style={{ fontSize: '0.9rem' }}>
            Pick a suggested action above or enter your own proposed value to preview directional impact.
          </div>
        ) : (
          <>
            <div style={{ fontSize: '0.9rem', marginBottom: '0.5rem' }}>
              Proposed change: <strong>{baselineValue}</strong> {' -> '} <strong>{proposedValue}</strong>
              {' '}({delta > 0 ? '+' : ''}{delta.toFixed(2)})
            </div>
            <div style={{ fontSize: '0.9rem', marginBottom: '0.5rem' }}>
              {isIncrease && guidance ? guidance.increase : null}
              {isDecrease && guidance ? guidance.decrease : null}
              {!isIncrease && !isDecrease ? 'No net change entered; risk is expected to remain close to baseline.' : null}
            </div>
            {guidance && (
              <div style={{ fontSize: '0.9rem' }}>
                Suggested clinical checks: {guidance.actions}
              </div>
            )}
          </>
        )}
      </div>

      <button onClick={handleAnalyze} className="btn btn-primary" disabled={loading}>
        {loading ? 'Analyzing...' : '🔍 Analyze Impact'}
      </button>

      {whatIfResult && (
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>Analysis Results</h3>

          <div className="card" style={{ background: 'var(--bg)', marginBottom: '1rem' }}>
            <h4 style={{ marginBottom: '0.75rem' }}>Risk Comparison (Before vs After)</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ padding: '0.75rem', borderRadius: '8px', background: '#fff', border: '1px solid var(--border)' }}>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Before Update</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--primary)' }}>{formatRisk(whatIfResult.baseline_risk)}</div>
              </div>

              <div style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', fontWeight: 700 }} aria-hidden="true">
                →
              </div>

              <div style={{ padding: '0.75rem', borderRadius: '8px', background: '#fff', border: '1px solid var(--border)' }}>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>After Update</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--primary)' }}>{formatRisk(whatIfResult.new_risk)}</div>
              </div>
            </div>

            <div style={{ marginTop: '0.8rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Delta:</span>
              <span
                style={{
                  fontSize: '0.95rem',
                  fontWeight: 700,
                  color: whatIfResult.risk_delta < 0 ? 'var(--secondary)' : 'var(--danger)',
                }}
              >
                {whatIfResult.risk_delta > 0 ? '+' : ''}{formatRisk(whatIfResult.risk_delta)}
              </span>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                ({whatIfResult.risk_delta_percent > 0 ? '+' : ''}{whatIfResult.risk_delta_percent.toFixed(1)}% relative)
              </span>
            </div>
          </div>

          {whatIfResult.clinical_summary && (
            <div className="card" style={{ background: 'var(--bg)', marginBottom: '1rem' }}>
              <h4 style={{ marginBottom: '0.5rem' }}>What Should Change</h4>
              <div style={{ fontSize: '0.95rem' }}>{whatIfResult.clinical_summary}</div>
            </div>
          )}

          {whatIfResult.recommended_changes && whatIfResult.recommended_changes.length > 0 && (
            <div className="card" style={{ background: 'var(--bg)', marginBottom: '1rem' }}>
              <h4 style={{ marginBottom: '0.75rem' }}>Ranked Risk-Reducing Actions</h4>
              <div className="feature-list">
                {whatIfResult.recommended_changes.map((item, idx) => (
                  <div key={idx} className="feature-item" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
                      <div className="feature-name">
                        {idx + 1}. {item.feature.replace(/_/g, ' ')}: {item.direction} to {Number(item.target_value).toFixed(2)}
                      </div>
                      <div style={{ fontWeight: 600, color: item.risk_delta < 0 ? 'var(--secondary)' : 'var(--danger)' }}>
                        {formatRisk(item.risk_delta)}
                      </div>
                    </div>
                    <div style={{ marginTop: '0.35rem', fontSize: '0.9rem' }}>
                      <strong>Risk:</strong> {formatRisk(whatIfResult.baseline_risk)} {' -> '} {formatRisk(item.expected_risk)}
                    </div>
                    <div style={{ marginTop: '0.35rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                      {item.action} | impact: {item.impact_label}
                    </div>
                    <div style={{ marginTop: '0.35rem', fontSize: '0.9rem' }}>
                      If {item.feature.replace(/_/g, ' ')} {item.direction.toLowerCase()}, risk is expected to {item.risk_delta < 0 ? 'decrease' : 'increase'} by {Math.abs(item.risk_delta * 100).toFixed(1)} percentage points.
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {whatIfResult.counterfactual_explanations && whatIfResult.counterfactual_explanations.length > 0 && (
            <div className="card" style={{ background: 'var(--bg)', marginBottom: '1rem' }}>
              <h4 style={{ marginBottom: '0.75rem' }}>Detailed Counterfactuals</h4>
              <ul style={{ marginLeft: '1.25rem', color: 'var(--text)', lineHeight: 1.8 }}>
                {whatIfResult.counterfactual_explanations.map((line, idx) => (
                  <li key={idx}>{line}</li>
                ))}
              </ul>
            </div>
          )}
          
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
