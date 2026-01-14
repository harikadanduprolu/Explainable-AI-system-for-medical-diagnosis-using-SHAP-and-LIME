"""
Clinician-Facing Dashboard Design for Unified Explainable AI System
====================================================================

A comprehensive UX specification for healthcare professionals using the
multi-disease explainable diagnosis system.

Design Principles:
- Clinical workflow integration
- Information hierarchy (critical → detailed)
- Trust and transparency
- Rapid decision support
- Minimal cognitive load

Target Users:
- Emergency physicians
- Intensivists
- Hospitalists
- Nurses
- Clinical decision support teams

Use Cases:
1. Risk assessment for new patients
2. Monitoring high-risk patients
3. Exploring intervention scenarios
4. Understanding AI recommendations
5. Documentation and audit trail
"""

# ============================================================================
# DASHBOARD LAYOUT STRUCTURE
# ============================================================================

DASHBOARD_LAYOUT = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER: Patient Info + Navigation                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ┌─────────────────┐  ┌──────────────────────────────────────────────────┐ │
│ │                 │  │                                                  │ │
│ │   PANEL 1:      │  │        PANEL 2: PREDICTION CONFIDENCE            │ │
│ │   RISK SUMMARY  │  │                                                  │ │
│ │   (Critical)    │  │  Disease Predictions + Confidence Scores         │ │
│ │                 │  │                                                  │ │
│ └─────────────────┘  └──────────────────────────────────────────────────┘ │
│                                                                             │
│ ┌───────────────────────────────────────────────────────────────────────┐  │
│ │                                                                       │  │
│ │   PANEL 3: FEATURE IMPORTANCE (SHAP/LIME)                            │  │
│ │                                                                       │  │
│ │   Top Contributing Factors (Interactive Bar Chart)                   │  │
│ │                                                                       │  │
│ └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│ ┌────────────────────────────┐  ┌──────────────────────────────────────┐  │
│ │                            │  │                                      │  │
│ │   PANEL 4:                 │  │   PANEL 5:                           │  │
│ │   CLINICAL SUMMARY         │  │   WHAT-IF ANALYSIS                   │  │
│ │                            │  │                                      │  │
│ │   Plain English            │  │   Interactive Scenario Testing       │  │
│ │   Recommendations          │  │                                      │  │
│ │                            │  │                                      │  │
│ └────────────────────────────┘  └──────────────────────────────────────┘  │
│                                                                             │
│ ┌───────────────────────────────────────────────────────────────────────┐  │
│ │                                                                       │  │
│ │   PANEL 6: PATIENT TIMELINE & HISTORY                                │  │
│ │                                                                       │  │
│ │   Historical predictions + interventions over time                   │  │
│ │                                                                       │  │
│ └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# PANEL 1: RISK SUMMARY (Critical Alert Panel)
# ============================================================================

PANEL_1_RISK_SUMMARY = {
    "name": "Risk Summary Panel",
    "position": "Top-left, prominent position",
    "size": "300px × 400px",
    "purpose": "Immediate risk assessment at-a-glance",
    
    "components": [
        {
            "component": "Risk Gauge (Primary)",
            "type": "Radial gauge / Semicircle",
            "data_shown": [
                "Overall risk score (0-100%)",
                "Risk category (Low/Moderate/High/Critical)",
                "Confidence interval (±X%)"
            ],
            "visual_design": {
                "colors": {
                    "Low": "#4CAF50 (Green)",
                    "Moderate": "#FFC107 (Amber)",
                    "High": "#FF9800 (Orange)",
                    "Critical": "#F44336 (Red)"
                },
                "gauge_segments": [
                    "0-25%: Green zone",
                    "25-50%: Amber zone",
                    "50-75%: Orange zone",
                    "75-100%: Red zone"
                ],
                "animation": "Smooth needle movement (0.5s ease)",
                "font": "Large, bold percentage (48px)"
            },
            "interactions": [
                "Hover: Show confidence interval tooltip",
                "Click: Expand detailed breakdown"
            ]
        },
        {
            "component": "Disease Risk Badges",
            "type": "Compact status indicators",
            "data_shown": [
                "Sepsis: XX%",
                "Acute Kidney Injury: XX%",
                "Cardiovascular Event: XX%",
                "Mortality Risk: XX%"
            ],
            "visual_design": {
                "layout": "Vertical stack of pills",
                "badge_design": "Icon + Disease Name + Risk %",
                "color_coding": "Background color based on risk level",
                "icons": "Disease-specific medical icons"
            },
            "interactions": [
                "Hover: Show last prediction time",
                "Click: Navigate to disease-specific view"
            ]
        },
        {
            "component": "Trend Indicator",
            "type": "Sparkline + Arrow",
            "data_shown": [
                "Risk trend (↑ increasing, ↓ decreasing, → stable)",
                "Mini time-series chart (last 24h)",
                "Change magnitude (+5% in 2h)"
            ],
            "visual_design": {
                "arrows": {
                    "↑ Red": "Risk increasing",
                    "↓ Green": "Risk decreasing",
                    "→ Gray": "Stable"
                },
                "sparkline_color": "Follows current risk level"
            },
            "interactions": [
                "Hover: Show detailed trend data",
                "Click: Open timeline panel"
            ]
        },
        {
            "component": "Last Updated Timestamp",
            "type": "Text label",
            "data_shown": "Updated: 2 minutes ago",
            "visual_design": {
                "position": "Bottom of panel",
                "font": "Small, gray (12px)"
            }
        },
        {
            "component": "Refresh Button",
            "type": "Icon button",
            "data_shown": "↻ Refresh icon",
            "interactions": [
                "Click: Recompute predictions",
                "Loading state: Spinning animation"
            ]
        }
    ],
    
    "trust_indicators": [
        "Confidence score visible",
        "Timestamp for recency",
        "Trend shows consistency",
        "Model version tooltip"
    ],
    
    "alerts": {
        "critical_risk": {
            "trigger": "Risk ≥ 75%",
            "visual": "Pulsing red border + bell icon",
            "sound": "Optional audio alert",
            "action": "Suggest immediate review"
        },
        "rapid_deterioration": {
            "trigger": "Risk increased >15% in <1h",
            "visual": "Yellow warning banner",
            "action": "Show contributing factors"
        }
    }
}


# ============================================================================
# PANEL 2: PREDICTION CONFIDENCE (Multi-Disease View)
# ============================================================================

PANEL_2_PREDICTION_CONFIDENCE = {
    "name": "Prediction Confidence Panel",
    "position": "Top-right, beside Risk Summary",
    "size": "700px × 400px",
    "purpose": "Detailed disease-specific predictions with confidence",
    
    "components": [
        {
            "component": "Disease Cards Grid",
            "type": "4×1 card layout",
            "data_shown": "One card per disease (Sepsis, AKI, CV, Mortality)",
            
            "card_structure": [
                {
                    "header": {
                        "disease_icon": "Medical symbol (syringe, kidney, heart, cross)",
                        "disease_name": "Sepsis / Acute Kidney Injury / etc.",
                        "model_type": "XGBoost / Random Forest badge"
                    },
                    "body": {
                        "risk_probability": "Large percentage (36px bold)",
                        "confidence_bar": "Horizontal progress bar (0-100%)",
                        "risk_category": "Colored pill badge (Low/Moderate/High/Critical)",
                        "prevalence_note": "Population baseline: 9.8%"
                    },
                    "footer": {
                        "explain_button": "🔍 View Explanation",
                        "what_if_button": "🎯 What-If Analysis"
                    }
                }
            ],
            
            "visual_design": {
                "card_background": "White with colored left border (risk-based)",
                "hover_state": "Subtle elevation shadow",
                "spacing": "16px gap between cards",
                "border_colors": "Match risk level colors"
            },
            
            "interactions": [
                "Hover card: Show last prediction time + model version",
                "Click 'View Explanation': Navigate to SHAP panel",
                "Click 'What-If': Open scenario builder",
                "Click card header: Toggle expanded details"
            ]
        },
        {
            "component": "Model Performance Indicators",
            "type": "Collapsible section per card",
            "data_shown": [
                "ROC-AUC: 0.91",
                "Precision: 0.85",
                "Recall: 0.82",
                "Last trained: 2025-12-15"
            ],
            "visual_design": {
                "collapsed_by_default": True,
                "expand_icon": "▼ / ▲",
                "font": "Small, monospace (10px)"
            },
            "interactions": [
                "Click expand: Show full metrics",
                "Hover metric: Tooltip with definition"
            ]
        },
        {
            "component": "Comparison Toggle",
            "type": "Toggle switch",
            "data_shown": "Compare with previous prediction",
            "interactions": [
                "Toggle ON: Show delta (△ +5%)",
                "Color code: Green (↓) or Red (↑)"
            ]
        }
    ],
    
    "trust_indicators": [
        "Confidence bars visible",
        "Model type transparency",
        "Performance metrics available",
        "Population prevalence context",
        "Timestamp per prediction"
    ],
    
    "tooltips": {
        "confidence_bar": "Model confidence: How certain the AI is about this prediction. Higher is more confident.",
        "prevalence": "Expected rate in general population. This patient's risk is compared to this baseline.",
        "model_type": "XGBoost: Tree-based ensemble model known for high accuracy in healthcare.",
        "explain_button": "See which clinical factors are driving this prediction (SHAP analysis)"
    }
}


# ============================================================================
# PANEL 3: FEATURE IMPORTANCE (SHAP/LIME Visualization)
# ============================================================================

PANEL_3_FEATURE_IMPORTANCE = {
    "name": "Feature Importance Panel",
    "position": "Center, full width",
    "size": "1000px × 350px",
    "purpose": "Show which factors drive the prediction (explainability)",
    
    "components": [
        {
            "component": "Explainer Method Tabs",
            "type": "Tab navigation",
            "tabs": ["SHAP Values", "LIME Explanation", "Both (Comparison)"],
            "visual_design": {
                "active_tab": "Underlined + bold",
                "inactive_tab": "Gray text"
            },
            "interactions": [
                "Click tab: Switch visualization method",
                "Tooltip: Explain difference between SHAP and LIME"
            ]
        },
        {
            "component": "Horizontal Bar Chart (Primary Viz)",
            "type": "Interactive horizontal bars",
            "data_shown": [
                "Top 8 features (sorted by |importance|)",
                "Feature name (left axis)",
                "Importance score (bar length)",
                "Direction: Red bars (increase risk) / Green bars (decrease risk)",
                "Feature value annotation (e.g., 'Temp: 38.5°C')"
            ],
            
            "visual_design": {
                "bar_colors": {
                    "positive_impact": "#F44336 (Red) - Increases risk",
                    "negative_impact": "#4CAF50 (Green) - Decreases risk"
                },
                "bar_height": "32px per feature",
                "font": "14px for labels, 12px for values",
                "x_axis": "Importance score (-1.0 to +1.0 for SHAP)",
                "baseline_line": "Vertical line at x=0 (dashed gray)"
            },
            
            "interactions": [
                "Hover bar: Show detailed tooltip",
                "Click bar: Highlight feature in other panels",
                "Drag to reorder: Allow manual prioritization (optional)"
            ],
            
            "tooltip_content": {
                "example": {
                    "feature": "Body Temperature",
                    "value": "38.5°C",
                    "normal_range": "36.1 - 37.2°C",
                    "status": "HIGH (fever)",
                    "shap_value": "+0.15",
                    "interpretation": "Elevated temperature (fever) strongly indicates infection, increasing sepsis risk by ~15%.",
                    "percentile": "Patient is in 85th percentile for this feature",
                    "actionable": "✓ ACTIONABLE: Can be reduced with antipyretics"
                }
            }
        },
        {
            "component": "Feature Value Display",
            "type": "Inline text annotations",
            "data_shown": "Current value next to each bar (e.g., '110 bpm')",
            "visual_design": {
                "color_coding": {
                    "abnormal_high": "Red text",
                    "abnormal_low": "Blue text",
                    "normal": "Black text"
                },
                "badge": "⚠️ icon for abnormal values"
            }
        },
        {
            "component": "Clinical Translation Panel",
            "type": "Expandable text section",
            "data_shown": [
                "Plain English explanation of top 3 features",
                "Clinical context (why this matters)",
                "Recommended actions"
            ],
            "visual_design": {
                "background": "Light blue (#E3F2FD)",
                "icon": "💬 speech bubble",
                "font": "16px, easy-to-read"
            },
            "example_text": "This patient's HIGH RISK is primarily driven by:\n\n1. **Body Temperature (38.5°C)** - Fever indicates possible infection\n2. **White Blood Cell Count (15,000)** - Elevated immune response\n3. **Blood Lactate (2.5)** - Tissue hypoperfusion\n\nRecommendation: Consider early sepsis protocol and broad-spectrum antibiotics."
        },
        {
            "component": "View Toggle Controls",
            "type": "Button group",
            "options": [
                "Top 5",
                "Top 10",
                "All Features",
                "Only Abnormal"
            ],
            "interactions": [
                "Click: Filter displayed features"
            ]
        },
        {
            "component": "Export Button",
            "type": "Icon button",
            "data_shown": "📄 Export",
            "interactions": [
                "Click: Download PNG or PDF of visualization",
                "Options: Include in patient report"
            ]
        }
    ],
    
    "trust_indicators": [
        "Bidirectional bars (increase vs decrease risk)",
        "Actual feature values shown",
        "Normal ranges provided",
        "Clinical interpretation included",
        "Method transparency (SHAP vs LIME)"
    ],
    
    "tooltips": {
        "shap_tab": "SHAP Values: Global feature importance based on Shapley values from game theory. Consistent across all predictions.",
        "lime_tab": "LIME: Local linear approximation of the model's behavior around this specific patient.",
        "positive_bar": "This feature INCREASES the predicted risk. Higher values push risk up.",
        "negative_bar": "This feature DECREASES the predicted risk. Acts as a protective factor.",
        "abnormal_icon": "This value is outside the normal clinical range."
    }
}


# ============================================================================
# PANEL 4: CLINICAL SUMMARY (Recommendations)
# ============================================================================

PANEL_4_CLINICAL_SUMMARY = {
    "name": "Clinical Summary & Recommendations Panel",
    "position": "Bottom-left",
    "size": "450px × 350px",
    "purpose": "Plain English summary and actionable recommendations",
    
    "components": [
        {
            "component": "Summary Card",
            "type": "Text block with icons",
            "sections": [
                {
                    "header": "🏥 Clinical Assessment",
                    "content": "Patient shows HIGH sepsis risk (75%) due to fever (38.5°C), tachycardia (110 bpm), and elevated WBC (15,000). Risk has increased 15% in the last 2 hours.",
                    "font": "16px, line-height 1.6"
                },
                {
                    "header": "📋 Recommended Actions",
                    "content": [
                        "1. ⚡ URGENT: Monitor vitals every 1-2 hours",
                        "2. 💊 Consider broad-spectrum antibiotics if infection confirmed",
                        "3. 🌡️ Administer antipyretics (acetaminophen 1000mg)",
                        "4. 💉 Obtain blood cultures before antibiotics",
                        "5. 📊 Repeat lactate in 2 hours"
                    ],
                    "visual_design": {
                        "numbering": "Auto-numbered list",
                        "icons": "Emoji or medical icons",
                        "urgent_items": "Red background for priority items"
                    }
                },
                {
                    "header": "🎯 Target Metrics",
                    "content": [
                        "Temperature: ↓ to 37.0°C",
                        "Heart Rate: ↓ to 85 bpm",
                        "Lactate: ↓ to <2.0 mmol/L"
                    ],
                    "visual_design": {
                        "arrows": "Direction indicators (↑↓→)",
                        "color": "Green for targets"
                    }
                }
            ]
        },
        {
            "component": "Confidence Disclaimer",
            "type": "Info banner",
            "data_shown": "⚠️ AI Suggestion - Clinical judgment required. Model confidence: 89%",
            "visual_design": {
                "background": "#FFF3E0 (Light orange)",
                "border": "1px solid #FF9800",
                "font": "12px italic"
            }
        },
        {
            "component": "Copy to EMR Button",
            "type": "Action button",
            "data_shown": "📋 Copy to EMR / Export Note",
            "interactions": [
                "Click: Copy formatted summary to clipboard",
                "Success: Green checkmark animation"
            ]
        },
        {
            "component": "Feedback Buttons",
            "type": "Thumbs up/down",
            "data_shown": "Was this helpful? 👍 / 👎",
            "interactions": [
                "Click: Record clinician feedback",
                "Optional: Add comment"
            ]
        }
    ],
    
    "trust_indicators": [
        "Confidence score displayed",
        "Disclaimer about AI assistance",
        "Recommendations marked as suggestions",
        "Clinical judgment emphasized"
    ],
    
    "tooltips": {
        "summary": "AI-generated summary based on top risk factors and clinical guidelines.",
        "recommendations": "Suggested actions derived from sepsis protocols and feature importance.",
        "confidence": "Model's certainty about its prediction. Lower confidence = more uncertainty."
    }
}


# ============================================================================
# PANEL 5: WHAT-IF ANALYSIS (Interactive Scenario Testing)
# ============================================================================

PANEL_5_WHATIF_ANALYSIS = {
    "name": "What-If Analysis Panel",
    "position": "Bottom-right",
    "size": "550px × 350px",
    "purpose": "Explore alternative scenarios and intervention effects",
    
    "components": [
        {
            "component": "Scenario Builder",
            "type": "Interactive form",
            "sections": [
                {
                    "header": "Modify Patient Parameters",
                    "controls": [
                        {
                            "feature": "Temperature",
                            "current_value": "38.5°C",
                            "control_type": "Slider + Text input",
                            "range": "35.0 - 42.0°C",
                            "step": "0.1",
                            "constraints_shown": "Max change: ±3.0°C",
                            "visual": {
                                "slider_color": "Blue",
                                "constraint_markers": "Red zone markers at limits",
                                "normal_range_highlight": "Green band (36.1-37.2)"
                            }
                        },
                        {
                            "feature": "Heart Rate",
                            "current_value": "110 bpm",
                            "control_type": "Slider + Text input",
                            "range": "40 - 200 bpm",
                            "step": "1",
                            "constraints_shown": "Max change: ±40 bpm"
                        },
                        {
                            "feature": "WBC Count",
                            "current_value": "15,000",
                            "control_type": "Slider + Text input",
                            "range": "1,000 - 50,000",
                            "step": "100",
                            "constraints_shown": "Max change: ±5,000 (slow)"
                        }
                    ],
                    "visual_design": {
                        "layout": "Vertical stack",
                        "label_width": "150px",
                        "slider_width": "250px",
                        "spacing": "16px between controls"
                    }
                },
                {
                    "header": "Quick Presets",
                    "type": "Button group",
                    "presets": [
                        "✓ Normalize All",
                        "💊 Post-Treatment",
                        "⏱️ 4 Hours Later",
                        "🔄 Reset"
                    ],
                    "interactions": [
                        "Click preset: Apply predefined changes",
                        "Tooltip: Explain what each preset does"
                    ]
                }
            ]
        },
        {
            "component": "Risk Delta Display",
            "type": "Comparison card",
            "data_shown": {
                "baseline": {
                    "label": "Current Risk",
                    "value": "75%",
                    "color": "Red"
                },
                "arrow": "→",
                "modified": {
                    "label": "Predicted Risk",
                    "value": "45%",
                    "color": "Orange"
                },
                "delta": {
                    "label": "Change",
                    "value": "-30% (↓)",
                    "color": "Green",
                    "visual": "Large bold text with arrow"
                }
            },
            "visual_design": {
                "layout": "Horizontal three-column",
                "background": "Light gray (#F5F5F5)",
                "padding": "16px",
                "border_radius": "8px"
            }
        },
        {
            "component": "Plausibility Indicator",
            "type": "Status badge",
            "data_shown": {
                "realistic": "✓ REALISTIC - Achievable with standard treatment",
                "challenging": "⚠️ CHALLENGING - Requires intensive care",
                "unlikely": "⚠️ UNLIKELY - Aggressive intervention needed",
                "impossible": "✗ IMPOSSIBLE - Violates clinical constraints"
            },
            "visual_design": {
                "realistic": "Green badge",
                "challenging": "Amber badge",
                "unlikely": "Orange badge",
                "impossible": "Red badge"
            }
        },
        {
            "component": "Intervention Summary",
            "type": "Text list",
            "data_shown": [
                "Changes applied:",
                "• Temperature: 38.5°C → 37.0°C (↓ 1.5°C)",
                "• Heart Rate: 110 → 85 bpm (↓ 25 bpm)",
                "",
                "Feasibility:",
                "• Temperature: ACTIONABLE (antipyretics, 1-2h)",
                "• Heart Rate: ACTIONABLE (beta-blockers if appropriate)"
            ],
            "visual_design": {
                "font": "12px monospace",
                "actionable_color": "Green",
                "non_actionable_color": "Gray"
            }
        },
        {
            "component": "Action Buttons",
            "type": "Button row",
            "buttons": [
                {
                    "label": "Apply Scenario",
                    "action": "Compute new risk",
                    "visual": "Primary blue button"
                },
                {
                    "label": "Save Scenario",
                    "action": "Save to patient history",
                    "visual": "Secondary gray button"
                },
                {
                    "label": "Get Suggestions",
                    "action": "Use SHAP-guided recommendations",
                    "visual": "Secondary button with ✨ icon"
                }
            ]
        }
    ],
    
    "trust_indicators": [
        "Constraints visualized (red zones)",
        "Plausibility assessment shown",
        "Feasibility per intervention",
        "Delta calculation transparent"
    ],
    
    "tooltips": {
        "slider": "Drag to adjust value. Red zones = unsafe. Green zone = normal range.",
        "plausibility": "Clinical feasibility assessment based on physiological constraints.",
        "preset_normalize": "Sets all abnormal values to their normal range midpoint.",
        "preset_treatment": "Simulates expected values 2 hours after standard sepsis treatment.",
        "constraints": "Maximum allowed change based on clinical guidelines and safety limits."
    }
}


# ============================================================================
# PANEL 6: PATIENT TIMELINE & HISTORY
# ============================================================================

PANEL_6_TIMELINE = {
    "name": "Patient Timeline & History Panel",
    "position": "Bottom, full width",
    "size": "1000px × 250px",
    "purpose": "Track risk evolution and interventions over time",
    
    "components": [
        {
            "component": "Timeline Chart",
            "type": "Line chart with annotations",
            "data_shown": [
                "X-axis: Time (last 24 hours / 7 days / custom)",
                "Y-axis: Risk probability (0-100%)",
                "Line: Risk trajectory",
                "Points: Prediction timestamps",
                "Annotations: Clinical interventions"
            ],
            
            "visual_design": {
                "line_color": "Dynamic (follows current risk level)",
                "line_width": "3px",
                "points": "8px circles",
                "risk_zones": {
                    "background_bands": [
                        "0-25%: Light green (#E8F5E9)",
                        "25-50%: Light amber (#FFF8E1)",
                        "50-75%: Light orange (#FFE0B2)",
                        "75-100%: Light red (#FFEBEE)"
                    ]
                },
                "interventions": {
                    "marker": "Vertical dashed line",
                    "label": "Icon + short text",
                    "examples": [
                        "💊 Antibiotics started",
                        "🌡️ Antipyretics given",
                        "🏥 ICU transfer",
                        "📊 Lab results updated"
                    ]
                }
            },
            
            "interactions": [
                "Hover point: Show exact risk + time + confidence",
                "Hover intervention: Show details (drug, dose, time)",
                "Click point: Jump to that prediction snapshot",
                "Zoom: Scroll to zoom time range",
                "Pan: Drag to move timeline"
            ]
        },
        {
            "component": "Time Range Selector",
            "type": "Button group",
            "options": ["6 Hours", "24 Hours", "7 Days", "Custom"],
            "interactions": [
                "Click: Adjust visible time range",
                "Custom: Date picker dialog"
            ]
        },
        {
            "component": "Event Log Table",
            "type": "Collapsible data table",
            "data_shown": {
                "columns": [
                    "Timestamp",
                    "Event Type",
                    "Risk Before",
                    "Risk After",
                    "Delta",
                    "Notes"
                ],
                "example_rows": [
                    {
                        "timestamp": "2026-01-13 10:30",
                        "event": "Antibiotics started (Vancomycin)",
                        "risk_before": "78%",
                        "risk_after": "75%",
                        "delta": "-3%",
                        "notes": "Blood cultures obtained"
                    },
                    {
                        "timestamp": "2026-01-13 08:15",
                        "event": "Initial prediction",
                        "risk_before": "-",
                        "risk_after": "78%",
                        "delta": "-",
                        "notes": "ED admission"
                    }
                ]
            },
            "visual_design": {
                "collapsed_by_default": True,
                "expand_toggle": "▼ Show Event Log",
                "row_highlight": "Hover effect",
                "delta_color": "Green (↓) or Red (↑)"
            },
            "interactions": [
                "Click row: Highlight corresponding point on timeline",
                "Sort: Click column headers",
                "Filter: Search box for events"
            ]
        },
        {
            "component": "Trend Analysis Summary",
            "type": "Text card",
            "data_shown": [
                "Overall trend: ↓ Decreasing (Good)",
                "Rate of change: -10% per hour",
                "Stable since: 2 hours ago",
                "Next prediction: 15 minutes"
            ],
            "visual_design": {
                "position": "Top-right of panel",
                "background": "White card",
                "font": "14px"
            }
        }
    ],
    
    "trust_indicators": [
        "Historical consistency visible",
        "Intervention effects traceable",
        "Prediction timestamps shown",
        "Model updates logged"
    ],
    
    "tooltips": {
        "timeline": "Risk trajectory over time. Points = new predictions. Lines = interventions.",
        "intervention_marker": "Clinical action taken. Hover for details.",
        "risk_zone": "Color bands show risk categories (Green=Low, Red=Critical).",
        "gap": "No prediction during this time (patient may have been transferred or data unavailable)."
    }
}


# ============================================================================
# HEADER: PATIENT INFO & NAVIGATION
# ============================================================================

HEADER_COMPONENT = {
    "name": "Dashboard Header",
    "position": "Top, full width",
    "size": "100% × 80px",
    "purpose": "Patient context and navigation",
    
    "components": [
        {
            "component": "Patient Info Card",
            "type": "Horizontal info strip",
            "data_shown": [
                "Patient ID: P12345",
                "Name: John Doe (if available)",
                "Age: 65 years",
                "Gender: Male",
                "Admission: 2026-01-13 06:00 (4h ago)",
                "Location: ED Bed 7"
            ],
            "visual_design": {
                "layout": "Left-aligned, horizontal",
                "separator": " | ",
                "font": "14px",
                "icon": "👤 patient icon"
            }
        },
        {
            "component": "Navigation Tabs",
            "type": "Tab bar",
            "tabs": [
                "Overview (current)",
                "Sepsis Details",
                "AKI Details",
                "Cardiovascular",
                "Mortality Risk",
                "Comparative Analysis"
            ],
            "visual_design": {
                "active_tab": "Blue underline + bold",
                "inactive_tab": "Gray text"
            },
            "interactions": [
                "Click tab: Switch disease-specific view"
            ]
        },
        {
            "component": "Global Actions",
            "type": "Button group (right-aligned)",
            "buttons": [
                {
                    "icon": "🔔",
                    "label": "Alerts (2)",
                    "badge": "Red notification dot",
                    "action": "Open alerts panel"
                },
                {
                    "icon": "⚙️",
                    "label": "Settings",
                    "action": "Open preferences"
                },
                {
                    "icon": "📊",
                    "label": "Export Report",
                    "action": "Generate PDF report"
                },
                {
                    "icon": "❓",
                    "label": "Help",
                    "action": "Open documentation"
                }
            ]
        }
    ]
}


# ============================================================================
# COLOR CODING SYSTEM
# ============================================================================

COLOR_SYSTEM = {
    "risk_levels": {
        "Low": {
            "primary": "#4CAF50",  # Green
            "light": "#E8F5E9",
            "dark": "#2E7D32",
            "usage": "Risk < 25%"
        },
        "Moderate": {
            "primary": "#FFC107",  # Amber
            "light": "#FFF8E1",
            "dark": "#F57F17",
            "usage": "Risk 25-50%"
        },
        "High": {
            "primary": "#FF9800",  # Orange
            "light": "#FFE0B2",
            "dark": "#E65100",
            "usage": "Risk 50-75%"
        },
        "Critical": {
            "primary": "#F44336",  # Red
            "light": "#FFEBEE",
            "dark": "#B71C1C",
            "usage": "Risk > 75%"
        }
    },
    
    "semantic_colors": {
        "positive_change": "#4CAF50",  # Green (risk decreased)
        "negative_change": "#F44336",  # Red (risk increased)
        "neutral": "#9E9E9E",          # Gray
        "actionable": "#2196F3",       # Blue (can be modified)
        "fixed": "#9E9E9E",            # Gray (cannot be modified)
        "warning": "#FF9800",          # Orange
        "info": "#2196F3",             # Blue
        "success": "#4CAF50"           # Green
    },
    
    "accessibility": {
        "contrast_ratio": "WCAG AA compliant (4.5:1 minimum)",
        "colorblind_safe": "Use patterns + text labels in addition to color",
        "alternatives": "Icons accompany color coding"
    }
}


# ============================================================================
# INTERACTION PATTERNS
# ============================================================================

INTERACTION_PATTERNS = {
    "hover_behaviors": [
        "Tooltips appear after 300ms delay",
        "Cards elevate with shadow on hover",
        "Chart points highlight on hover",
        "Buttons show subtle scale effect (1.05x)"
    ],
    
    "click_behaviors": [
        "Primary actions: Immediate execution",
        "Dangerous actions: Confirmation dialog",
        "Loading states: Spinner + disabled state",
        "Success: Green checkmark animation (1s)"
    ],
    
    "keyboard_navigation": [
        "Tab: Navigate between interactive elements",
        "Enter/Space: Activate focused element",
        "Esc: Close modals/dialogs",
        "Arrow keys: Adjust sliders"
    ],
    
    "responsive_behaviors": [
        "< 1200px: Stack panels vertically",
        "< 768px: Mobile layout (single column)",
        "Touch: Larger hit targets (44×44px minimum)"
    ],
    
    "real_time_updates": [
        "WebSocket connection for live data",
        "Auto-refresh every 5 minutes (configurable)",
        "Visual pulse on new data",
        "Alert sound for critical changes (optional)"
    ]
}


# ============================================================================
# TRUST & TRANSPARENCY FEATURES
# ============================================================================

TRUST_FEATURES = {
    "model_transparency": [
        "Model type visible (XGBoost, Random Forest)",
        "Training date shown",
        "Performance metrics available",
        "Version number displayed"
    ],
    
    "uncertainty_communication": [
        "Confidence scores always visible",
        "Confidence intervals shown",
        "Low confidence warnings",
        "Prediction ranges (min-max)"
    ],
    
    "clinical_context": [
        "Population baseline comparison",
        "Normal ranges displayed",
        "Percentile information",
        "Historical trends"
    ],
    
    "disclaimers": [
        "AI assistance disclaimer visible",
        "Clinical judgment required",
        "Not a diagnostic tool alone",
        "Intended for decision support"
    ],
    
    "audit_trail": [
        "All predictions logged",
        "User interactions recorded",
        "Timestamp every action",
        "Exportable for review"
    ]
}


# ============================================================================
# RESPONSIVE BREAKPOINTS
# ============================================================================

RESPONSIVE_DESIGN = {
    "desktop": {
        "breakpoint": "> 1200px",
        "layout": "6-panel grid as shown",
        "sidebar": "Visible"
    },
    
    "tablet": {
        "breakpoint": "768px - 1200px",
        "layout": "Stacked 2-column",
        "changes": [
            "Risk Summary + Prediction Confidence side-by-side",
            "Other panels stack vertically",
            "Timeline remains full width"
        ]
    },
    
    "mobile": {
        "breakpoint": "< 768px",
        "layout": "Single column",
        "changes": [
            "All panels stack vertically",
            "Collapsible sections by default",
            "Swipe gestures for navigation",
            "Bottom navigation bar"
        ]
    }
}


# ============================================================================
# COMPONENT LIBRARY
# ============================================================================

COMPONENT_LIST = {
    "data_visualization": [
        "Radial risk gauge",
        "Horizontal bar charts (SHAP)",
        "Line chart (timeline)",
        "Sparklines (trends)",
        "Progress bars (confidence)",
        "Heatmaps (feature correlations)"
    ],
    
    "inputs": [
        "Sliders (with constraints)",
        "Text inputs (numeric)",
        "Dropdowns (disease selection)",
        "Toggle switches",
        "Radio buttons",
        "Date/time pickers"
    ],
    
    "display_elements": [
        "Cards (disease predictions)",
        "Badges (risk levels)",
        "Pills (status indicators)",
        "Icons (medical symbols)",
        "Tooltips (explanations)",
        "Modals (detailed views)"
    ],
    
    "navigation": [
        "Tab bars",
        "Breadcrumbs",
        "Sidebar menu",
        "Pagination",
        "Back/forward buttons"
    ],
    
    "feedback": [
        "Loading spinners",
        "Success checkmarks",
        "Error messages",
        "Warning banners",
        "Progress indicators",
        "Toast notifications"
    ]
}


# ============================================================================
# ACCESSIBILITY FEATURES
# ============================================================================

ACCESSIBILITY = {
    "wcag_compliance": "Level AA",
    
    "features": [
        "Screen reader support (ARIA labels)",
        "Keyboard navigation",
        "Focus indicators",
        "Alt text for all images",
        "Color-independent information",
        "Adjustable text size",
        "High contrast mode",
        "Reduced motion option"
    ],
    
    "testing": [
        "Automated accessibility testing (axe)",
        "Manual screen reader testing",
        "Keyboard-only navigation testing",
        "Color contrast verification"
    ]
}


# ============================================================================
# USAGE EXAMPLE: COMPLETE WORKFLOW
# ============================================================================

USAGE_WORKFLOW = """
CLINICIAN WORKFLOW: Emergency Department Sepsis Case
====================================================

1. PATIENT ARRIVAL (t=0)
   - Patient admitted to ED Bed 7
   - Vitals entered into system
   - Dashboard loads with initial prediction
   
   [PANEL 1: RISK SUMMARY]
   ⚠️ HIGH RISK (75%) - Red gauge
   - Sepsis: 75% ↑
   - AKI: 35%
   - CV: 28%
   - Mortality: 18%
   
   🔔 ALERT: "High sepsis risk detected. Recommend immediate evaluation."

2. REVIEW EXPLANATION (t=2 min)
   Clinician hovers over Risk Summary → Clicks "View Explanation"
   
   [PANEL 3: FEATURE IMPORTANCE]
   Top factors:
   1. Temperature: 38.5°C (+0.15 SHAP) ⚠️ HIGH
      💬 "Elevated temperature strongly indicates infection"
   2. WBC: 15,000 (+0.12 SHAP) ⚠️ HIGH
   3. Heart Rate: 110 bpm (+0.08 SHAP) ⚠️ HIGH
   
   [PANEL 4: CLINICAL SUMMARY]
   "Patient shows high sepsis risk due to fever, tachycardia, and elevated WBC.
   Recommend: Obtain blood cultures, consider antibiotics, monitor closely."

3. EXPLORE INTERVENTION (t=5 min)
   Clinician opens What-If panel
   
   [PANEL 5: WHAT-IF ANALYSIS]
   - Adjusts Temperature slider: 38.5°C → 37.0°C
   - Adjusts Heart Rate: 110 → 85 bpm
   - Clicks "Apply Scenario"
   
   Result:
   Risk: 75% → 45% (-30%) ✓ REALISTIC
   "Temperature reduction achievable with antipyretics (1-2h)"

4. IMPLEMENT TREATMENT (t=10 min)
   Clinician orders:
   - Acetaminophen 1000mg PO
   - Blood cultures × 2
   - Broad-spectrum antibiotics (pending cultures)
   - Repeat vitals in 1 hour
   
   Clicks "Copy to EMR" → Summary added to patient note

5. MONITOR PROGRESS (t=2 hours)
   System auto-refreshes with new vitals
   
   [PANEL 6: TIMELINE]
   Graph shows:
   - Initial risk: 75% (10:00)
   - Post-antipyretic: 68% (11:00) ↓ Improving
   - Current: 62% (12:00) ↓ Continuing to improve
   
   Interventions marked on timeline:
   💊 Antibiotics (10:15)
   🌡️ Antipyretics (10:10)

6. DISCHARGE DECISION (t=4 hours)
   Risk stabilized at 45% (Moderate)
   Clinician reviews trend, decides to admit to general floor
   Exports PDF report for handoff

OUTCOME: Early identification + appropriate treatment → Improved patient outcome
"""


# ============================================================================
# PRINT SUMMARY
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CLINICIAN DASHBOARD - UI WIREFRAME & COMPONENT SPECIFICATION")
    print("=" * 80)
    print()
    print("LAYOUT:")
    print(DASHBOARD_LAYOUT)
    print()
    print("=" * 80)
    print("PANELS OVERVIEW:")
    print("=" * 80)
    print()
    print("1. RISK SUMMARY - Critical at-a-glance risk assessment")
    print("   Size: 300×400px | Position: Top-left")
    print("   Components: Risk gauge, disease badges, trend indicator")
    print("   Trust: Confidence scores, timestamps, model version")
    print()
    print("2. PREDICTION CONFIDENCE - Multi-disease predictions")
    print("   Size: 700×400px | Position: Top-right")
    print("   Components: Disease cards (4), confidence bars, metrics")
    print("   Trust: Model transparency, performance metrics, prevalence context")
    print()
    print("3. FEATURE IMPORTANCE - SHAP/LIME explanations")
    print("   Size: 1000×350px | Position: Center")
    print("   Components: Bar chart, clinical translation, tooltips")
    print("   Trust: Bidirectional bars, actual values, normal ranges")
    print()
    print("4. CLINICAL SUMMARY - Plain English recommendations")
    print("   Size: 450×350px | Position: Bottom-left")
    print("   Components: Summary text, action list, target metrics")
    print("   Trust: Confidence disclaimer, AI assistance label")
    print()
    print("5. WHAT-IF ANALYSIS - Interactive scenario testing")
    print("   Size: 550×350px | Position: Bottom-right")
    print("   Components: Sliders, risk delta, plausibility indicator")
    print("   Trust: Constraints shown, feasibility assessment")
    print()
    print("6. PATIENT TIMELINE - Historical risk trajectory")
    print("   Size: 1000×250px | Position: Bottom full-width")
    print("   Components: Line chart, intervention markers, event log")
    print("   Trust: Historical consistency, intervention effects traceable")
    print()
    print("=" * 80)
    print("COLOR CODING:")
    print("=" * 80)
    print("  Low Risk (0-25%):      GREEN  (#4CAF50)")
    print("  Moderate (25-50%):     AMBER  (#FFC107)")
    print("  High (50-75%):         ORANGE (#FF9800)")
    print("  Critical (>75%):       RED    (#F44336)")
    print()
    print("  Increases Risk:        RED bars")
    print("  Decreases Risk:        GREEN bars")
    print("  Actionable Feature:    BLUE highlight")
    print("  Fixed Feature:         GRAY text")
    print()
    print("=" * 80)
    print("TRUST INDICATORS:")
    print("=" * 80)
    print("  ✓ Confidence scores always visible")
    print("  ✓ Model type and version transparent")
    print("  ✓ Timestamps for recency")
    print("  ✓ Performance metrics available")
    print("  ✓ Population baseline context")
    print("  ✓ Normal ranges displayed")
    print("  ✓ Clinical judgment disclaimer")
    print("  ✓ Audit trail of all actions")
    print()
    print("=" * 80)
    print("COMPONENT COUNT:")
    print("=" * 80)
    print(f"  Total Panels: 6")
    print(f"  Interactive Charts: 4 (gauge, bars, line, sliders)")
    print(f"  Buttons: 15+")
    print(f"  Input Controls: 8+ (sliders, toggles, dropdowns)")
    print(f"  Cards/Badges: 10+")
    print(f"  Tooltips: 30+")
    print()
    print("=" * 80)
    print("WIREFRAME COMPLETE")
    print("=" * 80)
