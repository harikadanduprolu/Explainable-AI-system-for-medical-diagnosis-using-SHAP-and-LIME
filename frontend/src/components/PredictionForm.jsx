import React, { useState, useEffect } from 'react';

const FEATURE_INFO = {
  age: { label: 'Age', unit: 'years', min: 18, max: 100, step: 1 },
  gender: { label: 'Gender', unit: '(0=F, 1=M)', min: 0, max: 1, step: 1 },
  heart_rate: { label: 'Heart Rate', unit: 'bpm', min: 40, max: 200, step: 1 },
  systolic_bp: { label: 'Systolic BP', unit: 'mmHg', min: 70, max: 250, step: 1 },
  diastolic_bp: { label: 'Diastolic BP', unit: 'mmHg', min: 40, max: 150, step: 1 },
  temperature: { label: 'Temperature', unit: '°F', min: 95, max: 106, step: 0.1 },
  respiratory_rate: { label: 'Respiratory Rate', unit: '/min', min: 8, max: 50, step: 1 },
  wbc_count: { label: 'WBC Count', unit: 'K/µL', min: 1, max: 50, step: 0.1 },
  hemoglobin: { label: 'Hemoglobin', unit: 'g/dL', min: 5, max: 20, step: 0.1 },
  platelet_count: { label: 'Platelet Count', unit: 'K/µL', min: 20, max: 700, step: 1 },
  creatinine: { label: 'Creatinine', unit: 'mg/dL', min: 0.3, max: 15, step: 0.1 },
  bun: { label: 'BUN', unit: 'mg/dL', min: 5, max: 200, step: 1 },
  glucose: { label: 'Glucose', unit: 'mg/dL', min: 50, max: 700, step: 1 },
  lactate: { label: 'Lactate', unit: 'mmol/L', min: 0.5, max: 25, step: 0.1 },
};

const DEFAULT_VALUES = {
  age: 65,
  gender: 1,
  heart_rate: 80,
  systolic_bp: 120,
  diastolic_bp: 80,
  temperature: 98.6,
  respiratory_rate: 16,
  wbc_count: 7.5,
  hemoglobin: 13.5,
  platelet_count: 250,
  creatinine: 1.0,
  bun: 15,
  glucose: 100,
  lactate: 1.5,
};

const IMAGING_FINDINGS = [
  { key: 'pneumonia', label: 'Pneumonia' },
  { key: 'edema', label: 'Pulmonary Edema' },
  { key: 'cardiomegaly', label: 'Cardiomegaly' },
  { key: 'pleural_effusion', label: 'Pleural Effusion' },
];

const createDefaultImagingMap = () =>
  IMAGING_FINDINGS.reduce((acc, finding) => ({ ...acc, [finding.key]: 0.2 }), {});

function PredictionForm({ onSubmit, initialData, loading }) {
  const [formData, setFormData] = useState(initialData || DEFAULT_VALUES);
  const [imagingEnabled, setImagingEnabled] = useState(false);
  const [imagingData, setImagingData] = useState({
    dicom_id: '',
    source: 'manual-entry',
    findings: createDefaultImagingMap(),
  });

  useEffect(() => {
    if (initialData) {
      setFormData(initialData);
    }
  }, [initialData]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: parseFloat(value) || 0,
    }));
  };

  const handleImagingToggle = (e) => {
    setImagingEnabled(e.target.checked);
  };

  const handleImagingFieldChange = (e) => {
    const { name, value } = e.target;
    if (name === 'dicom_id') {
      setImagingData((prev) => ({ ...prev, dicom_id: value }));
    } else {
      setImagingData((prev) => ({
        ...prev,
        findings: {
          ...prev.findings,
          [name]: Number(value),
        },
      }));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const payload = {
      features: formData,
    };

    if (imagingEnabled) {
      const normalizedFindings = Object.entries(imagingData.findings).reduce(
        (acc, [key, val]) => ({
          ...acc,
          [key]: Math.min(Math.max(parseFloat(val) || 0, 0), 1),
        }),
        {}
      );
      payload.imaging = {
        dicom_id: imagingData.dicom_id?.trim() || undefined,
        source: imagingData.source,
        findings: normalizedFindings,
      };
    }

    onSubmit(payload);
  };

  const resetForm = () => {
    setFormData(DEFAULT_VALUES);
    setImagingEnabled(false);
    setImagingData({
      dicom_id: '',
      source: 'manual-entry',
      findings: createDefaultImagingMap(),
    });
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>Patient Clinical Data</h2>
        <p>Enter patient vitals and laboratory values for disease risk prediction</p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          {Object.entries(FEATURE_INFO).map(([key, info]) => (
            <div key={key} className="form-group">
              <label htmlFor={key}>
                {info.label} <small>{info.unit}</small>
              </label>
              <input
                type="number"
                id={key}
                name={key}
                value={formData[key]}
                onChange={handleChange}
                min={info.min}
                max={info.max}
                step={info.step}
                required
              />
            </div>
          ))}
        </div>

        <div className="card" style={{ marginTop: '1.5rem' }}>
          {/* <div className="card-header">
            <h3 style={{ marginBottom: '0.25rem' }}>Chest X-ray Evidence (MIMIC-CXR)</h3>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 500 }}>
              <input
                type="checkbox"
                checked={imagingEnabled}
                onChange={handleImagingToggle}
              />
              Provide imaging probabilities to activate the fusion engine
            </label>
          </div> */}

          {imagingEnabled && (
            <div className="form-grid" style={{ marginTop: '1rem' }}>
              <div className="form-group" style={{ gridColumn: 'span 2' }}>
                <label htmlFor="dicom_id">DICOM ID</label>
                <input
                  type="text"
                  id="dicom_id"
                  name="dicom_id"
                  value={imagingData.dicom_id}
                  onChange={handleImagingFieldChange}
                  placeholder="e.g., 1.2.826..."
                />
              </div>
              {IMAGING_FINDINGS.map((finding) => (
                <div key={finding.key} className="form-group">
                  <label htmlFor={finding.key}>
                    {finding.label} Score
                    <small style={{ display: 'block', color: 'var(--text-secondary)' }}>
                      0 (absent) {"→"} 1 (strong evidence)
                    </small>
                  </label>
                  <input
                    type="range"
                    id={finding.key}
                    name={finding.key}
                    min={0}
                    max={1}
                    step={0.05}
                    value={imagingData.findings[finding.key]}
                    onChange={handleImagingFieldChange}
                  />
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    {(imagingData.findings[finding.key] * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? (
              <>
                <div className="loading-spinner" style={{ width: '20px', height: '20px' }}></div>
                Analyzing...
              </>
            ) : (
              '🔮 Predict Diseases'
            )}
          </button>
          <button type="button" onClick={resetForm} className="btn btn-secondary">
            🔄 Reset
          </button>
        </div>
      </form>
    </div>
  );
}

export default PredictionForm;
