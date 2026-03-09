import React, { useState } from 'react';
import PredictionForm from './components/PredictionForm';
import Results from './components/Results';
import WhatIfAnalysis from './components/WhatIfAnalysis';
import SamplePatients from './components/SamplePatients';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [predictions, setPredictions] = useState(null);
  const [currentPatient, setCurrentPatient] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePrediction = async (patientData) => {
    setLoading(true);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          patient_id: `P${Date.now()}`,
          features: patientData,
        }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPredictions(data);
      setCurrentPatient(patientData);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to get predictions. Please check if the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const loadSamplePatient = (sample) => {
    setCurrentPatient(sample.features);
    handlePrediction(sample.features);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>🏥 Explainable Medical AI System</h1>
          <p>Multi-Disease Prediction with SHAP & LIME Explanations</p>
        </div>
      </header>

      <main className="main-container">
        <div className="tabs">
          <button
            className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`}
            onClick={() => setActiveTab('predict')}
          >
            🔮 Predict
          </button>
          <button
            className={`tab-btn ${activeTab === 'whatif' ? 'active' : ''}`}
            onClick={() => setActiveTab('whatif')}
            disabled={!currentPatient}
          >
            🔄 What-If Analysis
          </button>
          <button
            className={`tab-btn ${activeTab === 'samples' ? 'active' : ''}`}
            onClick={() => setActiveTab('samples')}
          >
            👥 Sample Patients
          </button>
        </div>

        {activeTab === 'predict' && (
          <>
            <PredictionForm
              onSubmit={handlePrediction}
              initialData={currentPatient}
              loading={loading}
            />
            {predictions && !loading && <Results data={predictions} />}
          </>
        )}

        {activeTab === 'whatif' && currentPatient && (
          <WhatIfAnalysis baselinePatient={currentPatient} />
        )}

        {activeTab === 'samples' && (
          <SamplePatients onSelectSample={loadSamplePatient} />
        )}
      </main>
    </div>
  );
}

export default App;
