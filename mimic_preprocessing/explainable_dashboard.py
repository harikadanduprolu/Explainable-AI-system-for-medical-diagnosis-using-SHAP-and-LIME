#!/usr/bin/env python3
"""
Interactive Dashboard for Explainable Medical Diagnosis

This module creates an interactive web dashboard using Dash to visualize
model predictions, SHAP explanations, and LIME interpretations for clinical use.

Features:
- Patient selection and risk assessment display
- Interactive SHAP and LIME visualizations
- Clinical recommendations
- Model performance monitoring
- Feature importance analysis

Author: GitHub Copilot
License: Academic/Research Use Only
"""

import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional
import dash_bootstrap_components as dbc

# Import our explainable diagnosis system
from .explainable_medical_diagnosis import ExplainableMedicalDiagnosis

logger = logging.getLogger(__name__)

class ExplainableDiagnosisDashboard:
    """
    Interactive dashboard for explainable medical diagnosis visualization.
    """
    
    def __init__(self, config_path: str, results_dir: str):
        """
        Initialize the dashboard.
        
        Args:
            config_path: Path to configuration file
            results_dir: Directory containing saved results
        """
        self.config_path = config_path
        self.results_dir = Path(results_dir)
        self.diagnosis_system = None
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Load results if available
        self.results = self._load_results()
        self.patient_reports = self._load_patient_reports()
        
        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
    
    def _load_results(self) -> Dict:
        """Load saved results from file."""
        results_file = self.results_dir / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_patient_reports(self) -> Dict:
        """Load all patient reports."""
        reports = {}
        for report_file in self.results_dir.glob('clinical_report_patient_*.json'):
            patient_id = report_file.stem.split('_')[-1]
            with open(report_file, 'r') as f:
                reports[patient_id] = json.load(f)
        return reports
    
    def _setup_layout(self):
        """Setup the dashboard layout."""
        
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Explainable Medical Diagnosis Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Control Panel"),
                        dbc.CardBody([
                            html.Label("Select Patient:"),
                            dcc.Dropdown(
                                id='patient-dropdown',
                                options=[{'label': f'Patient {pid}', 'value': pid} 
                                        for pid in self.patient_reports.keys()],
                                value=list(self.patient_reports.keys())[0] if self.patient_reports else None
                            ),
                            html.Br(),
                            html.Label("Visualization Type:"),
                            dcc.RadioItems(
                                id='viz-type',
                                options=[
                                    {'label': 'SHAP Analysis', 'value': 'shap'},
                                    {'label': 'LIME Explanation', 'value': 'lime'},
                                    {'label': 'Feature Importance', 'value': 'importance'},
                                    {'label': 'Model Performance', 'value': 'performance'}
                                ],
                                value='shap',
                                labelStyle={'display': 'block'}
                            )
                        ])
                    ])
                ], width=3),
                
                # Main Content Area
                dbc.Col([
                    # Patient Risk Summary
                    dbc.Card([
                        dbc.CardHeader("Patient Risk Assessment"),
                        dbc.CardBody(id='patient-summary')
                    ], className="mb-3"),
                    
                    # Visualization Area
                    dbc.Card([
                        dbc.CardHeader("Analysis Visualization"),
                        dbc.CardBody([
                            dcc.Graph(id='main-visualization')
                        ])
                    ])
                ], width=9)
            ], className="mb-4"),
            
            # Additional Information Row
            dbc.Row([
                # Clinical Recommendations
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Clinical Recommendations"),
                        dbc.CardBody(id='clinical-recommendations')
                    ])
                ], width=6),
                
                # Feature Details
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Feature Analysis"),
                        dbc.CardBody(id='feature-details')
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Model Performance Summary
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Model Performance Metrics"),
                        dbc.CardBody(id='performance-metrics')
                    ])
                ])
            ])
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            Output('patient-summary', 'children'),
            Input('patient-dropdown', 'value')
        )
        def update_patient_summary(patient_id):
            if not patient_id or patient_id not in self.patient_reports:
                return "No patient selected"
            
            report = self.patient_reports[patient_id]
            prediction = report['prediction']
            
            # Risk score gauge
            risk_score = prediction['risk_score']
            risk_category = prediction['risk_category']
            
            # Color coding for risk levels
            if risk_score > 0.7:
                color = "danger"
            elif risk_score > 0.3:
                color = "warning"
            else:
                color = "success"
            
            return [
                dbc.Row([
                    dbc.Col([
                        html.H4(f"Patient {patient_id}"),
                        html.P(f"Risk Score: {risk_score:.3f}"),
                        dbc.Badge(f"Risk Level: {risk_category}", color=color, className="mb-2"),
                        html.P(f"Actual Outcome: {'Event' if prediction['actual_outcome'] else 'No Event'}")
                    ], width=6),
                    dbc.Col([
                        # Create a simple risk gauge
                        self._create_risk_gauge(risk_score)
                    ], width=6)
                ])
            ]
        
        @self.app.callback(
            Output('main-visualization', 'figure'),
            [Input('patient-dropdown', 'value'),
             Input('viz-type', 'value')]
        )
        def update_main_visualization(patient_id, viz_type):
            if not patient_id or patient_id not in self.patient_reports:
                return go.Figure()
            
            if viz_type == 'shap':
                return self._create_shap_plot(patient_id)
            elif viz_type == 'lime':
                return self._create_lime_plot(patient_id)
            elif viz_type == 'importance':
                return self._create_feature_importance_plot()
            elif viz_type == 'performance':
                return self._create_performance_plot()
            
            return go.Figure()
        
        @self.app.callback(
            Output('clinical-recommendations', 'children'),
            Input('patient-dropdown', 'value')
        )
        def update_recommendations(patient_id):
            if not patient_id or patient_id not in self.patient_reports:
                return "No recommendations available"
            
            recommendations = self.patient_reports[patient_id]['clinical_recommendations']
            
            return [
                html.Ol([
                    html.Li(rec) for rec in recommendations
                ])
            ]
        
        @self.app.callback(
            Output('feature-details', 'children'),
            Input('patient-dropdown', 'value')
        )
        def update_feature_details(patient_id):
            if not patient_id or patient_id not in self.patient_reports:
                return "No feature details available"
            
            report = self.patient_reports[patient_id]
            risk_factors = report.get('top_risk_factors', [])
            
            if not risk_factors:
                return "No risk factors available"
            
            # Create a table of top risk factors
            data = []
            for factor in risk_factors[:10]:
                data.append({
                    'Feature': factor['feature'],
                    'SHAP Value': f"{factor['shap_value']:.3f}",
                    'Impact': factor['impact']
                })
            
            return dash_table.DataTable(
                data=data,
                columns=[
                    {"name": "Feature", "id": "Feature"},
                    {"name": "SHAP Value", "id": "SHAP Value"},
                    {"name": "Impact", "id": "Impact"}
                ],
                style_cell={'textAlign': 'left'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Impact} = Increases Risk'},
                        'backgroundColor': '#ffcccc',
                    },
                    {
                        'if': {'filter_query': '{Impact} = Decreases Risk'},
                        'backgroundColor': '#ccffcc',
                    }
                ]
            )
        
        @self.app.callback(
            Output('performance-metrics', 'children'),
            Input('viz-type', 'value')  # Dummy input to trigger update
        )
        def update_performance_metrics(_):
            if not self.results or 'model_performance' not in self.results:
                return "No performance metrics available"
            
            perf = self.results['model_performance']
            
            return dbc.Row([
                dbc.Col([
                    html.H6("Accuracy"),
                    html.H4(f"{perf.get('accuracy', 0):.3f}")
                ], width=3),
                dbc.Col([
                    html.H6("AUC"),
                    html.H4(f"{perf.get('auc', 0):.3f}")
                ], width=3),
                dbc.Col([
                    html.H6("F1-Score"),
                    html.H4(f"{perf.get('f1_score', 0):.3f}")
                ], width=3),
                dbc.Col([
                    html.H6("CV AUC"),
                    html.H4(f"{perf.get('cv_mean', 0):.3f} ± {perf.get('cv_std', 0):.3f}")
                ], width=3)
            ])
    
    def _create_risk_gauge(self, risk_score: float) -> dcc.Graph:
        """Create a risk gauge visualization."""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        
        return dcc.Graph(figure=fig)
    
    def _create_shap_plot(self, patient_id: str) -> go.Figure:
        """Create SHAP waterfall plot for a patient."""
        if patient_id not in self.patient_reports:
            return go.Figure()
        
        report = self.patient_reports[patient_id]
        shap_contributions = report.get('shap_contributions', {})
        
        if not shap_contributions:
            return go.Figure().add_annotation(
                text="No SHAP data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Sort by absolute contribution
        sorted_features = sorted(shap_contributions.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:15]
        
        features, values = zip(*sorted_features)
        colors = ['red' if v < 0 else 'green' for v in values]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=features,
            x=values,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"SHAP Feature Contributions - Patient {patient_id}",
            xaxis_title="SHAP Value",
            yaxis_title="Features",
            height=600
        )
        
        return fig
    
    def _create_lime_plot(self, patient_id: str) -> go.Figure:
        """Create LIME explanation plot for a patient."""
        if patient_id not in self.patient_reports:
            return go.Figure()
        
        report = self.patient_reports[patient_id]
        lime_explanation = report.get('lime_explanation', {})
        
        if not lime_explanation or 'feature_contributions' not in lime_explanation:
            return go.Figure().add_annotation(
                text="No LIME data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        features, values = zip(*lime_explanation['feature_contributions'])
        colors = ['red' if v < 0 else 'green' for v in values]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=features,
            x=values,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"LIME Feature Contributions - Patient {patient_id}",
            xaxis_title="Contribution to Prediction",
            yaxis_title="Features",
            height=600
        )
        
        return fig
    
    def _create_feature_importance_plot(self) -> go.Figure:
        """Create global feature importance plot."""
        if not self.results or 'feature_importance' not in self.results:
            return go.Figure().add_annotation(
                text="No feature importance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        importance_data = self.results['feature_importance'].get('shap', [])
        
        if not importance_data:
            return go.Figure().add_annotation(
                text="No SHAP feature importance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Convert to DataFrame if it's a list of dicts
        if isinstance(importance_data, list):
            df_importance = pd.DataFrame(importance_data)
        else:
            df_importance = pd.DataFrame(importance_data)
        
        # Take top 20 features
        top_features = df_importance.head(20)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_features['feature'],
            x=top_features['importance'],
            orientation='h',
            marker_color='steelblue',
            text=[f"{v:.3f}" for v in top_features['importance']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Global Feature Importance (SHAP)",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Features",
            height=600
        )
        
        return fig
    
    def _create_performance_plot(self) -> go.Figure:
        """Create model performance visualization."""
        if not self.results or 'model_performance' not in self.results:
            return go.Figure().add_annotation(
                text="No performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        perf = self.results['model_performance']
        
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy', 'AUC', 'F1-Score', 'Cross-Validation'],
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        # Accuracy gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=perf.get('accuracy', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 0.7], 'color': "lightgray"},
                            {'range': [0.7, 1], 'color': "gray"}]}
        ), row=1, col=1)
        
        # AUC gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=perf.get('auc', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkgreen"},
                   'steps': [{'range': [0, 0.7], 'color': "lightgray"},
                            {'range': [0.7, 1], 'color': "gray"}]}
        ), row=1, col=2)
        
        # F1-Score gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=perf.get('f1_score', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkred"},
                   'steps': [{'range': [0, 0.7], 'color': "lightgray"},
                            {'range': [0.7, 1], 'color': "gray"}]}
        ), row=2, col=1)
        
        # CV scores (if available)
        if 'classification_report' in perf:
            classes = list(perf['classification_report'].keys())[:-3]  # Exclude avg metrics
            precision = [perf['classification_report'][c]['precision'] for c in classes]
            recall = [perf['classification_report'][c]['recall'] for c in classes]
            
            fig.add_trace(go.Bar(
                x=classes,
                y=precision,
                name='Precision',
                marker_color='lightblue'
            ), row=2, col=2)
            
            fig.add_trace(go.Bar(
                x=classes,
                y=recall,
                name='Recall',
                marker_color='lightcoral'
            ), row=2, col=2)
        
        fig.update_layout(height=600, title="Model Performance Dashboard")
        
        return fig
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the dashboard server."""
        logger.info(f"Starting dashboard server on port {port}")
        self.app.run_server(debug=debug, port=port)


def create_dashboard_from_results(config_path: str, results_dir: str, 
                                port: int = 8050, debug: bool = True):
    """
    Create and run dashboard from existing results.
    
    Args:
        config_path: Path to configuration file
        results_dir: Directory containing results
        port: Port to run server on
        debug: Whether to run in debug mode
    """
    dashboard = ExplainableDiagnosisDashboard(config_path, results_dir)
    dashboard.run_server(debug=debug, port=port)


if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Run Explainable Diagnosis Dashboard")
    parser.add_argument('config', help="Configuration file path")
    parser.add_argument('results_dir', help="Results directory path")
    parser.add_argument('--port', type=int, default=8050, help="Port to run server on")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode")
    
    args = parser.parse_args()
    
    create_dashboard_from_results(args.config, args.results_dir, args.port, args.debug)
