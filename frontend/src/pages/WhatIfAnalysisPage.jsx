import React, { useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import WhatIfAnalysis from '../components/WhatIfAnalysis';
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

const WhatIfAnalysisPage = () => {
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

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="clinical-app">
      <div className="clinical-header">
        <div className="header-content">
          <div className="header-title">
            <div className="header-icon">🔄</div>
            <div>
              <h1>What-If Analysis</h1>
              <div className="header-subtitle">Route-based clinical counterfactual exploration</div>
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
          </div>
        </div>

        <WhatIfAnalysis baselinePatient={baselinePatient} availableDiseases={availableDiseases} />
      </div>
    </div>
  );
};

export default WhatIfAnalysisPage;