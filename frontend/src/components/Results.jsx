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

  const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;

  const getScoreColor = (category) => {
    switch (category) {
      case 'CRITICAL':
        return 'var(--danger)';
      case 'HIGH':
      case 'MODERATE':
        return 'var(--warning)';
      default:
        return 'var(--secondary)';
    }
  };

  const renderFusionSummary = () => {
    const summary = data.multimodal_summary;
    if (!summary) {
      return null;
    }

    return (
      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div className="card-header">
          <h3>[Multimodal Evidence Fusion]</h3>
          <p>
            Consistency Index:{' '}
            <strong>{formatPercent(summary.consistency_index)}</strong>
          </p>
          <small style={{ color: 'var(--text-secondary)' }}>
            Sources: {summary.data_sources.tabular} + {summary.data_sources.imaging}
          </small>
        </div>

        <div className="table-responsive">
          <table className="table">
            <thead>
              <tr>
                <th>Disease</th>
                <th>Tabular Risk</th>
                <th>CXR Signal</th>
                <th>Severity Mod.</th>
                <th>Fused Score</th>
                <th>Agreement</th>
                <th>Flag</th>
              </tr>
            </thead>
            <tbody>
              {summary.fused_predictions.map((fused) => (
                <tr key={fused.disease}>
                  <td>{fused.disease.replace(/_/g, ' ')}</td>
                  <td>{formatPercent(fused.tabular_risk)}</td>
                  <td>
                    {fused.imaging_signal !== null && fused.imaging_signal !== undefined
                      ? formatPercent(fused.imaging_signal)
                      : '--'}
                  </td>
                  <td>
                    {fused.severity_modifier !== null && fused.severity_modifier !== undefined
                      ? formatPercent(fused.severity_modifier)
                      : '--'}
                  </td>
                  <td>{formatPercent(fused.fused_score)}</td>
                  <td>{formatPercent(fused.agreement_index)}</td>
                  <td>{fused.governance_flag || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {summary.alerts.length > 0 && (
          <div className="alert alert-warning" style={{ marginTop: '1rem' }}>
            <strong>Alerts:</strong> {summary.alerts.join(', ')}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>[Prediction Results]</h2>
        <p>
          Overall Risk:{' '}
          <span className={`risk-badge ${getRiskColor(data.overall_risk_category)}`}>
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
                  <summary style={{ cursor: 'pointer', fontSize: '0.9rem', fontWeight: 500 }}>
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

              {pred.shap_top_features && pred.shap_top_features.length > 0 && (
                <details style={{ marginTop: '0.75rem' }}>
                  <summary style={{ cursor: 'pointer', fontSize: '0.9rem', fontWeight: 500 }}>
                    SHAP Local Explanation
                  </summary>
                  <div className="feature-list">
                    {pred.shap_top_features.map((feat, idx) => (
                      <div key={idx} className="feature-item">
                        <div className="feature-name">{feat.feature_name.replace(/_/g, ' ')}</div>
                        <div className="feature-value">
                          SHAP: {feat.importance > 0 ? '+' : ''}{feat.importance.toFixed(4)}
                        </div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                          Value: {feat.value.toFixed(2)}
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              )}

              {pred.lime_top_features && pred.lime_top_features.length > 0 && (
                <details style={{ marginTop: '0.75rem' }}>
                  <summary style={{ cursor: 'pointer', fontSize: '0.9rem', fontWeight: 500 }}>
                    LIME Local Explanation
                  </summary>
                  <div className="feature-list">
                    {pred.lime_top_features.map((feat, idx) => (
                      <div key={idx} className="feature-item">
                        <div className="feature-name">{feat.feature_name.replace(/_/g, ' ')}</div>
                        <div className="feature-value">
                          LIME: {feat.importance > 0 ? '+' : ''}{feat.importance.toFixed(4)}
                        </div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                          Value: {feat.value.toFixed(2)}
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              )}

              {pred.explanation_methods && pred.explanation_methods.length > 0 && (
                <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Methods: {pred.explanation_methods.join(', ')}
                </div>
              )}

              {pred.explanation_warnings && pred.explanation_warnings.length > 0 && (
                <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--warning)' }}>
                  {pred.explanation_warnings.join(' | ')}
                </div>
              )}

              {pred.clinical_decision_support_report && (
                <details style={{ marginTop: '0.75rem' }}>
                  <summary style={{ cursor: 'pointer', fontSize: '0.9rem', fontWeight: 500 }}>
                    Clinical Decision Support Insight
                  </summary>
                  <pre
                    style={{
                      marginTop: '0.5rem',
                      whiteSpace: 'pre-wrap',
                      fontFamily: 'inherit',
                      fontSize: '0.9rem',
                      lineHeight: 1.45,
                      color: 'var(--text-primary)',
                    }}
                  >
                    {pred.clinical_decision_support_report}
                  </pre>
                </details>
              )}
            </div>
            <div
              className="result-score"
              style={{ color: getScoreColor(pred.risk_category) }}
            >
              {(pred.risk_score * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>

      {renderFusionSummary()}

      <div className="alert alert-info" style={{ marginTop: '1.5rem' }}>
        <strong>[Note]</strong> These predictions are based on machine learning models trained on MIMIC data.
        Results should be reviewed by qualified healthcare professionals and not used as sole basis for medical decisions.
      </div>
    </div>
  );
}

export default Results;
