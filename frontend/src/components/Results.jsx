import React from 'react';

function Results({ data }) {
  if (!data || !data.predictions) {
    return null;
  }

  const getRiskColor = (category) => {
    const colors = {
      LOW: 'low',
      MODERATE: 'moderate',
      HIGH: 'high',
      CRITICAL: 'critical',
    };
    return colors[category] || 'moderate';
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>📊 Prediction Results</h2>
        <p>
          Overall Risk: <span className={`risk-badge ${getRiskColor(data.overall_risk_category)}`}>
            {data.overall_risk_category}
          </span>
        </p>
      </div>

      <div className="results-grid">
        {data.predictions.map((pred) => (
          <div
            key={pred.disease}
            className={`result-item ${getRiskColor(pred.risk_category)}`}
          >
            <div className="result-info">
              <div className="result-disease">
                {pred.disease.replace(/_/g, ' ')}
              </div>
              <div>
                <span className={`risk-badge ${getRiskColor(pred.risk_category)}`}>
                  {pred.risk_category}
                </span>
                <small style={{ marginLeft: '1rem', color: 'var(--text-secondary)' }}>
                  Model: {pred.model_type} | Threshold: {pred.threshold.toFixed(3)}
                </small>
              </div>
              
              {pred.top_features && pred.top_features.length > 0 && (
                <details style={{ marginTop: '0.75rem' }}>
                  <summary style={{ cursor: 'pointer', fontSize: '0.9rem', fontWeight: '500' }}>
                    Top Contributing Features
                  </summary>
                  <div className="feature-list">
                    {pred.top_features.map((feat, idx) => (
                      <div key={idx} className="feature-item">
                        <div className="feature-name">{feat.feature_name.replace(/_/g, ' ')}</div>
                        <div className="feature-value">
                          Importance: {feat.importance.toFixed(3)}
                        </div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                          Value: {feat.value.toFixed(2)}
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
            <div className="result-score" style={{ color: `var(--${getRiskColor(pred.risk_category) === 'low' ? 'secondary' : getRiskColor(pred.risk_category) === 'critical' ? 'danger' : 'warning'})` }}>
              {(pred.risk_score * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>

      <div className="alert alert-info" style={{ marginTop: '1.5rem' }}>
        <strong>ℹ️ Note:</strong> These predictions are based on machine learning models trained on MIMIC-III data.
        Results should be reviewed by qualified healthcare professionals and not used as sole basis for medical decisions.
      </div>
    </div>
  );
}

export default Results;
