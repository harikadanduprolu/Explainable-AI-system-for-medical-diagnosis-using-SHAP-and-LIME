import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Login from './pages/Login';
import Signup from './pages/Signup';
import ClinicalDashboard from './pages/ClinicalDashboard';
import LandingPage from './pages/LandingPage';
import PredictionForm from './components/PredictionForm';
import Results from './components/Results';
import WhatIfAnalysis from './components/WhatIfAnalysis';
import SamplePatients from './components/SamplePatients';
import './index.css';

// Protected Route Component
function ProtectedRoute({ children }) {
  const { user, token, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="app">
        <main className="main-container">
          <div className="card">
            <div className="loading">
              <div className="loading-spinner" style={{ marginBottom: '0.5rem' }}></div>
              <p>Restoring secure session...</p>
            </div>
          </div>
        </main>
      </div>
    );
  }
  
  if (!user || !token) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  
  return children;
}

// Original Prediction App Component
function PredictionApp() {
  const [activeTab, setActiveTab] = useState('predict');
  const [predictions, setPredictions] = useState(null);
  const [currentPatient, setCurrentPatient] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePrediction = async (payload) => {
    const { features, imaging } = payload;
    setLoading(true);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          patient_id: `P${Date.now()}`,
          features,
          imaging,
        }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPredictions(data);
      setCurrentPatient(features);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to get predictions. Please check if the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const loadSamplePatient = (sample) => {
    setCurrentPatient(sample.features);
    handlePrediction({ features: sample.features });
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

// Main App with Routing
function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route 
            path="/clinical" 
            element={
              <ProtectedRoute>
                <ClinicalDashboard />
              </ProtectedRoute>
            } 
          />
          <Route path="/predict" element={<PredictionApp />} />
          <Route path="/" element={<LandingPage />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
