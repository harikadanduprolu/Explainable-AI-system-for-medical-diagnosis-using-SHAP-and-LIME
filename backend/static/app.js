// ============================================================================
// Explainable Medical AI - Frontend JavaScript
// ============================================================================

const API_BASE = window.location.origin;
let currentPatientData = null;

// ============================================================================
// Navigation
// ============================================================================

document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const targetView = btn.dataset.view;
        
        // Update active nav
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update active view
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        document.getElementById(`${targetView}-view`).classList.add('active');
    });
});

// ============================================================================
// Form Handling - Prediction
// ============================================================================

const patientForm = document.getElementById('patient-form');
const predictBtn = document.getElementById('predict-btn');
const resultsPlaceholder = document.getElementById('results-placeholder');
const resultsContainer = document.getElementById('results-container');

patientForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Collect form data
    const formData = new FormData(patientForm);
    const features = {};
    
    formData.forEach((value, key) => {
        features[key] = parseFloat(value);
    });
    
    // Store for what-if analysis
    currentPatientData = features;
    
    // Show loading state
    predictBtn.disabled = true;
    predictBtn.querySelector('.btn-text').style.display = 'none';
    predictBtn.querySelector('.btn-loading').style.display = 'inline';
    
    try {
        // Make API request
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                patient_id: `PATIENT-${Date.now()}`,
                features: features
            })
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error analyzing patient data. Please check your inputs and try again.');
    } finally {
        // Reset button
        predictBtn.disabled = false;
        predictBtn.querySelector('.btn-text').style.display = 'inline';
        predictBtn.querySelector('.btn-loading').style.display = 'none';
    }
});

// ============================================================================
// Display Results
// ============================================================================

function displayResults(data) {
    // Hide placeholder, show results
    resultsPlaceholder.style.display = 'none';
    resultsContainer.style.display = 'block';
    
    // Overall risk
    const overallRiskDisplay = document.getElementById('overall-risk-display');
    overallRiskDisplay.innerHTML = `
        <div class="risk-badge risk-${data.overall_risk_category}">
            ${data.overall_risk_category} RISK
        </div>
    `;
    
    // Timestamp
    const timestamp = new Date(data.timestamp).toLocaleString();
    document.getElementById('analysis-timestamp').textContent = `Analysis completed: ${timestamp}`;
    
    // Disease predictions
    const diseasePredictions = document.getElementById('disease-predictions');
    diseasePredictions.innerHTML = '';
    
    // Sort by risk score (highest first)
    const sortedPredictions = [...data.predictions].sort((a, b) => b.risk_score - a.risk_score);
    
    sortedPredictions.forEach(prediction => {
        const diseaseCard = createDiseaseCard(prediction);
        diseasePredictions.appendChild(diseaseCard);
    });
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function createDiseaseCard(prediction) {
    const card = document.createElement('div');
    card.className = `disease-card risk-${prediction.risk_category}`;
    
    const riskPercent = (prediction.risk_score * 100).toFixed(1);
    const riskColor = getRiskColor(prediction.risk_category);
    
    // Format disease name
    const diseaseName = prediction.disease.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    
    card.innerHTML = `
        <div class="disease-header">
            <span class="disease-name">${diseaseName}</span>
            <span class="risk-score" style="color: ${riskColor}">${riskPercent}%</span>
        </div>
        <div class="risk-bar">
            <div class="risk-bar-fill" style="width: ${riskPercent}%; background: ${riskColor}"></div>
        </div>
        <div style="margin-top: 8px;">
            <span style="font-size: 0.875rem; color: var(--gray-600);">Risk Category:</span>
            <strong style="color: ${riskColor}; margin-left: 8px;">${prediction.risk_category}</strong>
        </div>
        <div class="feature-importance">
            <strong style="font-size: 0.875rem; color: var(--gray-700); display: block; margin-bottom: 8px;">
                Top Contributing Factors:
            </strong>
            ${createFeatureList(prediction.top_features)}
        </div>
    `;
    
    return card;
}

function createFeatureList(features) {
    if (!features || features.length === 0) {
        return '<p style="color: var(--gray-600); font-size: 0.875rem;">No feature importance data available</p>';
    }
    
    return features.map(feature => {
        const featureName = feature.feature_name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        const importance = (feature.importance * 100).toFixed(1);
        const arrow = feature.direction === 'increases' ? '↑' : '↓';
        const color = feature.direction === 'increases' ? 'var(--danger)' : 'var(--success)';
        
        return `
            <div class="feature-item">
                <span class="feature-name">${featureName}</span>
                <span class="feature-value">
                    <span style="color: ${color}; margin-right: 4px;">${arrow}</span>
                    ${importance}% importance
                </span>
            </div>
        `;
    }).join('');
}

function getRiskColor(category) {
    const colors = {
        'LOW': 'var(--risk-low)',
        'MODERATE': 'var(--risk-moderate)',
        'HIGH': 'var(--risk-high)',
        'CRITICAL': 'var(--risk-critical)'
    };
    return colors[category] || 'var(--gray-600)';
}

// ============================================================================
// Sample Patients
// ============================================================================

const loadSampleBtn = document.getElementById('load-sample-btn');
const sampleModal = document.getElementById('sample-modal');
const modalClose = document.querySelector('.modal-close');

loadSampleBtn.addEventListener('click', async () => {
    try {
        const response = await fetch(`${API_BASE}/api/sample-patients`);
        const data = await response.json();
        
        displaySamplePatients(data.samples);
        sampleModal.classList.add('active');
    } catch (error) {
        console.error('Error loading samples:', error);
        alert('Error loading sample patients');
    }
});

modalClose.addEventListener('click', () => {
    sampleModal.classList.remove('active');
});

sampleModal.addEventListener('click', (e) => {
    if (e.target === sampleModal) {
        sampleModal.classList.remove('active');
    }
});

function displaySamplePatients(samples) {
    const list = document.getElementById('sample-patients-list');
    list.innerHTML = '';
    
    samples.forEach(sample => {
        const item = document.createElement('div');
        item.className = 'sample-item';
        item.innerHTML = `
            <div class="sample-name">${sample.name}</div>
            <div class="sample-description">Click to load this patient profile</div>
        `;
        
        item.addEventListener('click', () => {
            loadSampleData(sample.features);
            sampleModal.classList.remove('active');
        });
        
        list.appendChild(item);
    });
}

function loadSampleData(features) {
    Object.keys(features).forEach(key => {
        const input = document.getElementById(key);
        if (input) {
            input.value = features[key];
        }
    });
    
    // Scroll to form
    patientForm.scrollIntoView({ behavior: 'smooth' });
}

// ============================================================================
// What-If Analysis
// ============================================================================

const whatifAnalyzeBtn = document.getElementById('whatif-analyze-btn');
const copyFromPredictBtn = document.getElementById('copy-from-predict');

// Copy data from prediction form
copyFromPredictBtn.addEventListener('click', () => {
    if (!currentPatientData) {
        alert('Please run a prediction first');
        return;
    }
    
    // Copy key features to what-if form
    document.getElementById('wi_age').value = currentPatientData.age || '';
    document.getElementById('wi_glucose').value = currentPatientData.glucose || '';
    document.getElementById('wi_creatinine').value = currentPatientData.creatinine || '';
    document.getElementById('wi_heart_rate').value = currentPatientData.heart_rate || '';
});

// Analyze what-if scenario
whatifAnalyzeBtn.addEventListener('click', async () => {
    // Get baseline features (simplified - in real app would get all 14 features)
    const baselineFeatures = {
        age: parseFloat(document.getElementById('wi_age').value),
        gender: currentPatientData?.gender || 1,
        heart_rate: parseFloat(document.getElementById('wi_heart_rate').value),
        systolic_bp: currentPatientData?.systolic_bp || 120,
        diastolic_bp: currentPatientData?.diastolic_bp || 80,
        temperature: currentPatientData?.temperature || 98.6,
        respiratory_rate: currentPatientData?.respiratory_rate || 16,
        wbc_count: currentPatientData?.wbc_count || 7.5,
        hemoglobin: currentPatientData?.hemoglobin || 14,
        platelet_count: currentPatientData?.platelet_count || 250,
        creatinine: parseFloat(document.getElementById('wi_creatinine').value),
        bun: currentPatientData?.bun || 15,
        glucose: parseFloat(document.getElementById('wi_glucose').value),
        lactate: currentPatientData?.lactate || 1.2
    };
    
    const disease = document.getElementById('target-disease').value;
    const modifyFeature = document.getElementById('modify-feature').value;
    const newValue = parseFloat(document.getElementById('new-value').value);
    
    if (!disease || !modifyFeature || isNaN(newValue)) {
        alert('Please fill all fields');
        return;
    }
    
    try {
        whatifAnalyzeBtn.disabled = true;
        whatifAnalyzeBtn.textContent = 'Analyzing...';
        
        const response = await fetch(`${API_BASE}/api/whatif`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                baseline_features: baselineFeatures,
                modified_features: {
                    [modifyFeature]: newValue
                },
                disease: disease
            })
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        const data = await response.json();
        displayWhatIfResults(data);
        
    } catch (error) {
        console.error('What-if analysis error:', error);
        alert('Error performing what-if analysis');
    } finally {
        whatifAnalyzeBtn.disabled = false;
        whatifAnalyzeBtn.textContent = 'Analyze Scenario';
    }
});

function displayWhatIfResults(data) {
    const resultsDiv = document.getElementById('whatif-results');
    const resultsContent = document.getElementById('whatif-results-content');
    
    const baselinePercent = (data.baseline_risk * 100).toFixed(1);
    const newPercent = (data.new_risk * 100).toFixed(1);
    const deltaPercent = data.risk_delta_percent.toFixed(1);
    const deltaClass = data.risk_delta > 0 ? 'delta-positive' : 'delta-negative';
    const deltaSymbol = data.risk_delta > 0 ? '+' : '';
    
    const diseaseName = data.disease.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    
    resultsContent.innerHTML = `
        <div style="margin-bottom: 20px;">
            <h4 style="color: var(--gray-700); margin-bottom: 12px;">${diseaseName} Risk Analysis</h4>
        </div>
        
        <div class="comparison-grid">
            <div class="comparison-item">
                <div class="comparison-label">Baseline Risk</div>
                <div class="comparison-value">${baselinePercent}%</div>
            </div>
            <div class="comparison-item">
                <div class="comparison-label">New Risk</div>
                <div class="comparison-value">${newPercent}%</div>
            </div>
        </div>
        
        <div style="text-align: center; padding: 20px; background: var(--gray-50); border-radius: 8px; margin: 16px 0;">
            <div class="comparison-label">Risk Change</div>
            <div class="comparison-value ${deltaClass}">
                ${deltaSymbol}${deltaPercent}%
            </div>
        </div>
        
        <div style="background: var(--gray-50); padding: 16px; border-radius: 8px;">
            <strong style="color: var(--gray-700);">Changes Made:</strong>
            ${Object.entries(data.modified_features).map(([feature, values]) => `
                <div style="margin-top: 8px;">
                    <strong>${feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</strong>
                    ${values.old.toFixed(1)} → ${values.new.toFixed(1)}
                </div>
            `).join('')}
        </div>
        
        <div style="margin-top: 16px; padding: 12px; background: var(--gray-50); border-radius: 6px;">
            <strong style="color: var(--gray-700);">Recommendation:</strong>
            <p style="margin-top: 8px; color: var(--gray-600);">${data.recommendation}</p>
        </div>
    `;
    
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================================================
// Initialization
// ============================================================================

console.log('🚀 Explainable Medical AI - Frontend Loaded');
console.log('API Base:', API_BASE);
