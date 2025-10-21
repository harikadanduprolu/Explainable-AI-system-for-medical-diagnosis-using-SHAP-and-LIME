# The above code is a Python script that starts with a shebang line `#!/usr/bin/env python3` which is
# used to specify the interpreter that should be used to run the script.
# The above code is a Python script that starts with a shebang line `#!/usr/bin/env python3` which is
# used to specify the interpreter that should be used to run the script. In this case, it specifies
# that the script should be run using Python 3.
#!/usr/bin/env python3
"""
Interactive Explainable Medical Diagnosis Dashboard

This creates a web-based dashboard for exploring explainable AI predictions
for medical diagnosis using Dash and Plotly.
"""

import warnings
warnings.filterwarnings('ignore')

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

# Check if results file exists
RESULTS_FILE = 'complete_multi_disease_results.json'

def load_results():
    """Load results from the complete demo."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    else:
        # Create sample data for dashboard demo
        return create_sample_results()

def create_sample_results():
    """Create sample results for dashboard demo."""
    return {
        'system_metadata': {
            'timestamp': datetime.now().isoformat(),
            'data_source': 'Demo Data',
            'patients_analyzed': 5,
            'diseases_modeled': ['sepsis', 'kidney_failure', 'cardiovascular', 'mortality'],
            'features_used': ['age', 'heart_rate', 'blood_pressure', 'temperature', 'glucose', 'creatinine']
        },
        'model_performance': {
            'sepsis': {'auc': 0.711, 'accuracy': 0.897, 'f1_score': 0.0, 'prevalence': 0.098},
            'kidney_failure': {'auc': 0.807, 'accuracy': 0.760, 'f1_score': 0.486, 'prevalence': 0.303},
            'cardiovascular': {'auc': 0.706, 'accuracy': 0.810, 'f1_score': 0.374, 'prevalence': 0.212},
            'mortality': {'auc': 0.508, 'accuracy': 0.843, 'f1_score': 0.041, 'prevalence': 0.159}
        },
        'patient_analysis': {
            'total_patients': 5,
            'risk_distribution': {'CRITICAL': 0, 'HIGH': 1, 'MODERATE': 1, 'LOW': 3},
            'critical_alerts': 0,
            'high_risk_interventions': 1
        },
        'patient_reports': [
            {
                'patient_id': 0,
                'risk_category': 'HIGH',
                'overall_risk_score': 0.326,
                'diseases': {
                    'sepsis': {'risk_score': 0.030, 'risk_category': 'LOW RISK'},
                    'kidney_failure': {'risk_score': 0.812, 'risk_category': 'HIGH RISK'},
                    'cardiovascular': {'risk_score': 0.107, 'risk_category': 'LOW RISK'},
                    'mortality': {'risk_score': 0.355, 'risk_category': 'LOW RISK'}
                },
                'recommendations': [
                    '🫘 HIGH KIDNEY FAILURE RISK - Nephrology consultation',
                    '🧪 Monitor: Creatinine, BUN, electrolytes',
                    '💧 Assess: Fluid balance and medication nephrotoxicity'
                ]
            }
        ]
    }

# Load data
results = load_results()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Explainable Medical AI Dashboard"

# Define colors
colors = {
    'background': '#f8f9fa',
    'text': '#212529',
    'primary': '#0d6efd',
    'success': '#198754',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#0dcaf0'
}

# Dashboard layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏥 Explainable Medical AI Dashboard", 
                style={'textAlign': 'center', 'color': colors['primary'], 'marginBottom': '30px'}),
        html.P(f"Multi-Disease Prediction System | Analysis of {results['system_metadata']['patients_analyzed']} patients",
               style={'textAlign': 'center', 'fontSize': '18px', 'color': colors['text']})
    ], style={'backgroundColor': colors['background'], 'padding': '20px', 'marginBottom': '20px'}),
    
    # Summary Cards Row
    html.Div([
        html.Div([
            html.Div([
                html.H3(f"{results['patient_analysis']['total_patients']}", style={'color': colors['primary']}),
                html.P("Patients Analyzed", style={'margin': '0'})
            ], className='card-body', style={'textAlign': 'center', 'padding': '20px', 
                                           'backgroundColor': 'white', 'borderRadius': '10px', 
                                           'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ], style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
        
        html.Div([
            html.Div([
                html.H3(f"{len(results['system_metadata']['diseases_modeled'])}", style={'color': colors['success']}),
                html.P("Disease Models", style={'margin': '0'})
            ], className='card-body', style={'textAlign': 'center', 'padding': '20px', 
                                           'backgroundColor': 'white', 'borderRadius': '10px', 
                                           'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ], style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
        
        html.Div([
            html.Div([
                html.H3(f"{results['patient_analysis']['high_risk_interventions']}", style={'color': colors['warning']}),
                html.P("High-Risk Patients", style={'margin': '0'})
            ], className='card-body', style={'textAlign': 'center', 'padding': '20px', 
                                           'backgroundColor': 'white', 'borderRadius': '10px', 
                                           'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ], style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
        
        html.Div([
            html.Div([
                html.H3(f"{results['patient_analysis']['critical_alerts']}", style={'color': colors['danger']}),
                html.P("Critical Alerts", style={'margin': '0'})
            ], className='card-body', style={'textAlign': 'center', 'padding': '20px', 
                                           'backgroundColor': 'white', 'borderRadius': '10px', 
                                           'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ], style={'width': '23%', 'display': 'inline-block', 'margin': '1%'})
    ], style={'marginBottom': '30px'}),
    
    # Charts Row
    html.Div([
        # Model Performance Chart
        html.Div([
            html.H4("📊 Model Performance (AUC Scores)", style={'color': colors['text'], 'marginBottom': '15px'}),
            dcc.Graph(id='model-performance-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%', 'padding': '20px',
                 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        
        # Risk Distribution Chart
        html.Div([
            html.H4("⚠️ Patient Risk Distribution", style={'color': colors['text'], 'marginBottom': '15px'}),
            dcc.Graph(id='risk-distribution-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%', 'padding': '20px',
                 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
    ], style={'marginBottom': '30px'}),
    
    # Disease Prevalence Chart
    html.Div([
        html.H4("🦠 Disease Prevalence Analysis", style={'color': colors['text'], 'marginBottom': '15px'}),
        dcc.Graph(id='disease-prevalence-chart')
    ], style={'width': '96%', 'margin': '2%', 'padding': '20px',
             'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Patient Details Section
    html.Div([
        html.H4("👤 Patient Analysis Details", style={'color': colors['text'], 'marginBottom': '15px'}),
        html.Div(id='patient-details')
    ], style={'width': '96%', 'margin': '2%', 'padding': '20px',
             'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Footer
    html.Div([
        html.P(f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
               f"📊 Data Source: {results['system_metadata']['data_source']}",
               style={'textAlign': 'center', 'color': colors['text'], 'margin': '0'})
    ], style={'backgroundColor': colors['background'], 'padding': '20px', 'marginTop': '30px'})
    
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa', 'minHeight': '100vh'})

# Callbacks for interactive charts
@app.callback(
    Output('model-performance-chart', 'figure'),
    Input('model-performance-chart', 'id')
)
def update_model_performance_chart(_):
    """Create model performance chart."""
    diseases = list(results['model_performance'].keys())
    auc_scores = [results['model_performance'][disease]['auc'] for disease in diseases]
    
    fig = go.Figure(data=[
        go.Bar(
            x=diseases,
            y=auc_scores,
            marker_color=['#198754' if score >= 0.7 else '#ffc107' if score >= 0.6 else '#dc3545' 
                         for score in auc_scores],
            text=[f'{score:.3f}' for score in auc_scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="AUC Scores by Disease",
        xaxis_title="Disease",
        yaxis_title="AUC Score",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

@app.callback(
    Output('risk-distribution-chart', 'figure'),
    Input('risk-distribution-chart', 'id')
)
def update_risk_distribution_chart(_):
    """Create risk distribution pie chart."""
    risk_dist = results['patient_analysis']['risk_distribution']
    
    # Filter out zero values
    labels = [k for k, v in risk_dist.items() if v > 0]
    values = [v for v in risk_dist.values() if v > 0]
    
    color_map = {
        'CRITICAL': '#dc3545',
        'HIGH': '#fd7e14', 
        'MODERATE': '#ffc107',
        'LOW': '#198754'
    }
    
    colors_list = [color_map.get(label, '#6c757d') for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors_list),
        textinfo='label+percent',
        textfont_size=12
    )])
    
    fig.update_layout(
        title="Risk Categories",
        height=400,
        showlegend=True
    )
    
    return fig

@app.callback(
    Output('disease-prevalence-chart', 'figure'),
    Input('disease-prevalence-chart', 'id')
)
def update_disease_prevalence_chart(_):
    """Create disease prevalence chart."""
    diseases = list(results['model_performance'].keys())
    prevalences = [results['model_performance'][disease]['prevalence'] * 100 
                  for disease in diseases]
    
    fig = go.Figure(data=[
        go.Bar(
            x=diseases,
            y=prevalences,
            marker_color=colors['info'],
            text=[f'{prev:.1f}%' for prev in prevalences],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Disease Prevalence in Dataset",
        xaxis_title="Disease",
        yaxis_title="Prevalence (%)",
        height=400,
        showlegend=False
    )
    
    return fig

@app.callback(
    Output('patient-details', 'children'),
    Input('patient-details', 'id')
)
def update_patient_details(_):
    """Create patient details section."""
    if not results.get('patient_reports'):
        return html.P("No patient data available.")
    
    patient_cards = []
    
    for patient in results['patient_reports'][:3]:  # Show first 3 patients
        # Risk category styling
        risk_color = {
            'CRITICAL': colors['danger'],
            'HIGH': colors['warning'], 
            'MODERATE': colors['info'],
            'LOW': colors['success']
        }.get(patient['risk_category'], colors['text'])
        
        # Disease risk scores
        disease_rows = []
        for disease, data in patient['diseases'].items():
            disease_rows.append(
                html.Tr([
                    html.Td(disease.replace('_', ' ').title()),
                    html.Td(f"{data['risk_score']:.3f}"),
                    html.Td(data['risk_category'], 
                           style={'color': risk_color if 'HIGH' in data['risk_category'] else colors['success']})
                ])
            )
        
        # Recommendations
        rec_items = [html.Li(rec) for rec in patient.get('recommendations', [])]
        
        patient_card = html.Div([
            html.H5(f"Patient {patient['patient_id']}", style={'color': colors['primary']}),
            html.P(f"Overall Risk: {patient['risk_category']}", 
                   style={'color': risk_color, 'fontWeight': 'bold'}),
            html.P(f"Risk Score: {patient['overall_risk_score']:.3f}"),
            
            html.H6("Disease-Specific Risks:", style={'marginTop': '15px'}),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Disease"),
                        html.Th("Risk Score"),
                        html.Th("Category")
                    ])
                ]),
                html.Tbody(disease_rows)
            ], style={'width': '100%', 'marginBottom': '15px'}),
            
            html.H6("Clinical Recommendations:"),
            html.Ul(rec_items, style={'marginLeft': '20px'})
            
        ], style={
            'border': f'2px solid {risk_color}',
            'borderRadius': '10px',
            'padding': '15px',
            'margin': '10px',
            'backgroundColor': '#f8f9fa'
        })
        
        patient_cards.append(patient_card)
    
    return html.Div(patient_cards)

# Run the app
if __name__ == '__main__':
    print("🚀 Starting Explainable Medical AI Dashboard...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050")
    print("🔍 Features:")
    print("   - Model performance visualization")
    print("   - Patient risk distribution")
    print("   - Disease prevalence analysis") 
    print("   - Detailed patient reports")
    print("   - Interactive explainable AI insights")
    print("\n💡 Tip: Keep this terminal open while using the dashboard")
    print("🛑 Press Ctrl+C to stop the dashboard")
    
    app.run(debug=True, host='127.0.0.1', port=8050)
