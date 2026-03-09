import React, { useState, useEffect } from 'react';

function SamplePatients({ onSelectSample }) {
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSamples();
  }, []);

  const fetchSamples = async () => {
    try {
      const response = await fetch('/api/sample-patients');
      const data = await response.json();
      setSamples(data.samples);
    } catch (error) {
      console.error('Error fetching samples:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading">
        <div className="loading-spinner"></div>
        <p>Loading sample patients...</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <h2>👥 Sample Patient Profiles</h2>
        <p>Click on a patient profile to load their data and view predictions</p>
      </div>

      <div className="sample-patients">
        {samples.map((sample, idx) => (
          <div
            key={idx}
            className="sample-card"
            onClick={() => onSelectSample(sample)}
          >
            <h4>{sample.name}</h4>
            <p>
              Age: {sample.features.age} | HR: {sample.features.heart_rate} bpm
            </p>
            <p>
              Temp: {sample.features.temperature}°F | Lactate: {sample.features.lactate} mmol/L
            </p>
            <p style={{ fontSize: '0.8rem', marginTop: '0.5rem', fontStyle: 'italic' }}>
              Click to analyze →
            </p>
          </div>
        ))}
      </div>

      <div className="alert alert-info" style={{ marginTop: '1.5rem' }}>
        <strong>💡 Tip:</strong> Sample patients represent different clinical scenarios
        commonly seen in ICU settings. Each profile is designed to demonstrate the system's
        ability to assess various disease risk levels.
      </div>
    </div>
  );
}

export default SamplePatients;
