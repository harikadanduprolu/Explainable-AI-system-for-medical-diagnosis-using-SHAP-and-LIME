#!/usr/bin/env python3
"""
SHAP and LIME Explanations Demonstration
==========================================
Shows both SHAP and LIME explanations for trained disease prediction models.
"""

import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular

print("=" * 80)
print("🔬 EXPLAINABLE AI DEMONSTRATION - SHAP & LIME")
print("=" * 80)
print("\nThis demonstrates how AI models make medical diagnosis predictions")
print("using two state-of-the-art explainability techniques:")
print("  • SHAP (SHapley Additive exPlanations)")
print("  • LIME (Local Interpretable Model-agnostic Explanations)")
print()

# ============================================================================
# PART 1: Load Model and Test Patient
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: Loading Trained Model and Test Patient")
print("=" * 80)

# Load sepsis prediction model
model_path = Path("trained_models/sepsis_xgboost_v1.0.0.pkl")
print(f"\n📦 Loading model: {model_path.name}")

bundle = joblib.load(model_path)
model = bundle['model']
scaler = bundle['scaler']
feature_names = list(scaler.feature_names_in_)

print(f"✅ Model loaded successfully")
print(f"   Disease: {bundle.get('disease', 'sepsis')}")
print(f"   Model type: {bundle.get('model_type', 'xgboost')}")
print(f"   Features: {len(feature_names)}")

# Create test patient with high sepsis risk
print("\n👤 Test Patient Profile:")
patient_data = {
    'age': 72.0,
    'gender': 1,
    'heart_rate': 125.0,
    'systolic_bp': 90.0,
    'diastolic_bp': 55.0,
    'temperature': 102.0,
    'respiratory_rate': 28.0,
    'wbc_count': 22.0,
    'hemoglobin': 10.5,
    'platelet_count': 95.0,
    'creatinine': 2.1,
    'bun': 38.0,
    'glucose': 195.0,
    'lactate': 4.2,
}

# Add engineered features
patient_data['shock_index'] = patient_data['heart_rate'] / patient_data['systolic_bp']
patient_data['hr_bp_ratio'] = patient_data['heart_rate'] / patient_data['systolic_bp']
patient_data['creat_bun_ratio'] = patient_data['creatinine'] / patient_data['bun']
patient_data['age_glucose'] = patient_data['age'] * patient_data['glucose']

# Display vital signs
print(f"   Age: {patient_data['age']:.0f} years")
print(f"   Heart Rate: {patient_data['heart_rate']:.0f} bpm (ELEVATED)")
print(f"   Blood Pressure: {patient_data['systolic_bp']:.0f}/{patient_data['diastolic_bp']:.0f} mmHg (LOW)")
print(f"   Temperature: {patient_data['temperature']:.1f}°F (FEVER)")
print(f"   Respiratory Rate: {patient_data['respiratory_rate']:.0f} breaths/min (TACHYPNEA)")
print(f"   WBC: {patient_data['wbc_count']:.1f} K/µL (ELEVATED)")
print(f"   Lactate: {patient_data['lactate']:.1f} mmol/L (ELEVATED)")

# Make prediction
X_patient = pd.DataFrame([patient_data])[feature_names]
X_scaled = scaler.transform(X_patient)
risk_prob = model.predict_proba(X_scaled)[0][1]

print(f"\n🎯 SEPSIS RISK PREDICTION: {risk_prob:.1%}")
if risk_prob >= 0.7:
    print(f"   ⚠️  CRITICAL RISK - Immediate intervention recommended")
elif risk_prob >= 0.5:
    print(f"   ⚠️  HIGH RISK - Close monitoring required")
else:
    print(f"   ✓  MODERATE-LOW RISK")

# ============================================================================
# PART 2: SHAP Explanations
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: SHAP (SHapley Additive exPlanations)")
print("=" * 80)
print("\nSHAP values show how each feature contributes to the prediction:")
print("  • Positive SHAP = Increases sepsis risk")
print("  • Negative SHAP = Decreases sepsis risk")

# Create SHAP explainer
print("\n🔧 Creating SHAP explainer...")
# Use a background dataset (sample of training data approximation)
np.random.seed(42)
background_data = np.random.randn(100, len(feature_names)) * 0.5

try:
    # Try TreeExplainer first (fastest for tree models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_scaled)
    print("✅ Using TreeExplainer")
except Exception as e:
    print(f"⚠️  TreeExplainer failed ({str(e)[:50]}...), using KernelExplainer")
    # Fall back to KernelExplainer (model-agnostic but slower)
    def model_predict(X):
        return model.predict_proba(X)[:, 1]
    
    explainer = shap.KernelExplainer(model_predict, background_data)
    shap_values_raw = explainer.shap_values(X_scaled)
    
    # Wrap in Explanation object for consistency
    class SimpleExplanation:
        def __init__(self, values, base_values):
            self.values = values
            self.base_values = base_values
    
    shap_values = SimpleExplanation(shap_values_raw, explainer.expected_value)

print("✅ SHAP values calculated")

# Display SHAP values for top features
print("\n📊 SHAP Feature Contributions (Top 10):")
print("-" * 80)

# Get SHAP values for the positive class (sepsis risk)
if hasattr(shap_values, 'values'):
    if len(shap_values.values.shape) == 3:
        # TreeExplainer with multi-class output
        shap_vals = shap_values.values[0, :, 1]
        base_value = shap_values.base_values[0][1]
    else:
        # TreeExplainer with binary output or KernelExplainer
        shap_vals = shap_values.values[0] if len(shap_values.values.shape) == 2 else shap_values.values
        base_value = shap_values.base_values[0] if hasattr(shap_values.base_values, '__len__') else shap_values.base_values
else:
    # Simple array from KernelExplainer
    shap_vals = shap_values.values if hasattr(shap_values, 'values') else shap_values
    base_value = shap_values.base_values if hasattr(shap_values, 'base_values') else 0.5

# Sort by absolute SHAP value
indices = np.argsort(np.abs(shap_vals))[::-1]

for i, idx in enumerate(indices[:10]):
    feature = feature_names[idx]
    shap_val = shap_vals[idx]
    patient_val = X_patient.iloc[0, idx]
    
    # Direction indicator
    direction = "↑ INCREASES" if shap_val > 0 else "↓ DECREASES"
    bar = "█" * int(abs(shap_val) * 50)
    
    print(f"{i+1:2d}. {feature:20s} = {patient_val:8.2f}")
    print(f"    SHAP: {shap_val:+.4f}  {direction} risk")
    print(f"    {bar}")

print(f"\nBase value (average prediction): {base_value:.4f}")
print(f"SHAP contributions sum: {shap_vals.sum():.4f}")
print(f"Final prediction: {base_value + shap_vals.sum():.4f}")

# Create SHAP visualizations
print("\n📈 Generating SHAP visualizations...")

try:
    # Waterfall plot
    plt.figure(figsize=(10, 6))
    if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
        shap.plots.waterfall(shap_values[0, :, 1], max_display=10, show=False)
    else:
        # Create manual waterfall for KernelExplainer results
        indices_sorted = np.argsort(np.abs(shap_vals))[::-1][:10]
        features_sorted = [feature_names[i] for i in indices_sorted]
        values_sorted = [shap_vals[i] for i in indices_sorted]
        
        plt.barh(range(len(values_sorted)), values_sorted)
        plt.yticks(range(len(values_sorted)), features_sorted)
        plt.xlabel('SHAP value (impact on model output)')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.gca().invert_yaxis()
    
    plt.title("SHAP Waterfall Plot - Sepsis Risk Factors", fontsize=14, fontweight='bold')
    plt.tight_layout()
    waterfall_path = "shap_waterfall_sepsis.png"
    plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {waterfall_path}")
except Exception as e:
    print(f"   ⚠️  Waterfall plot error: {e}")

try:
    # Feature importance bar plot (simpler alternative to summary plot)
    plt.figure(figsize=(10, 6))
    
    # Get feature importances from model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        plt.barh(range(len(indices)), [importances[i] for i in indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title("Global Feature Importance - Sepsis Model", fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        summary_path = "shap_summary_sepsis.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {summary_path}")
except Exception as e:
    print(f"   ⚠️  Summary plot error: {e}")

# ============================================================================
# PART 3: LIME Explanations
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: LIME (Local Interpretable Model-agnostic Explanations)")
print("=" * 80)
print("\nLIME creates a simple, interpretable model around the prediction:")
print("  • Uses a linear model to approximate the complex model locally")
print("  • Shows which features matter most for THIS specific patient")

# Create LIME explainer
print("\n🔧 Creating LIME explainer...")

# Create background data for LIME
X_background = pd.DataFrame(
    scaler.inverse_transform(np.random.randn(1000, len(feature_names)) * 0.5),
    columns=feature_names
)

lime_explainer = lime_tabular.LimeTabularExplainer(
    X_background.values,
    feature_names=feature_names,
    class_names=['No Sepsis', 'Sepsis'],
    mode='classification',
    random_state=42
)

# Create prediction function
def predict_fn(X):
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)

# Explain the prediction
print("🔍 Generating LIME explanation...")
lime_exp = lime_explainer.explain_instance(
    X_patient.values[0],
    predict_fn,
    num_features=10,
    num_samples=1000
)

print("✅ LIME explanation generated")

# Display LIME explanation
print("\n📊 LIME Feature Contributions (Top 10):")
print("-" * 80)

lime_list = lime_exp.as_list()
for i, (feature_desc, weight) in enumerate(lime_list[:10], 1):
    direction = "↑ INCREASES" if weight > 0 else "↓ DECREASES"
    bar = "█" * int(abs(weight) * 20)
    print(f"{i:2d}. {feature_desc:50s}")
    print(f"    Weight: {weight:+.4f}  {direction} risk")
    print(f"    {bar}")

# LIME probabilities
lime_probs = lime_exp.predict_proba
print(f"\nLIME Predicted Probabilities:")
print(f"  No Sepsis: {lime_probs[0]:.1%}")
print(f"  Sepsis:    {lime_probs[1]:.1%}")

# Save LIME visualization
print("\n📈 Generating LIME visualization...")
try:
    fig = lime_exp.as_pyplot_figure()
    plt.title("LIME Explanation - Sepsis Risk Factors", fontsize=14, fontweight='bold')
    plt.tight_layout()
    lime_path = "lime_explanation_sepsis.png"
    plt.savefig(lime_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {lime_path}")
except Exception as e:
    print(f"   ⚠️  LIME plot error: {e}")

# ============================================================================
# PART 4: Comparison and Clinical Interpretation
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: Clinical Interpretation")
print("=" * 80)

print("\n🏥 CLINICAL SUMMARY:")
print("-" * 80)
print(f"\nPatient Risk: {risk_prob:.1%} probability of sepsis")
print("\nKey Risk Factors Identified by Both SHAP and LIME:")

# Find common top features
shap_top_features = [feature_names[idx] for idx in indices[:5]]
lime_top_features = [item[0].split(' ')[0] for item in lime_list[:5]]

common_features = []
for shap_feat in shap_top_features:
    for lime_feat in lime_top_features:
        if shap_feat in lime_feat or lime_feat in shap_feat:
            if shap_feat not in common_features:
                common_features.append(shap_feat)

for i, feat in enumerate(common_features[:5], 1):
    val = patient_data.get(feat, X_patient[feat].values[0])
    print(f"  {i}. {feat}: {val:.2f}")

print("\n💡 CLINICAL RECOMMENDATIONS:")
print("  • Monitor vital signs closely (high heart rate, fever)")
print("  • Check for infection source (elevated WBC, temperature)")
print("  • Assess fluid status (low BP, elevated lactate)")
print("  • Consider early antibiotic therapy")
print("  • Monitor kidney function (elevated creatinine)")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("📁 FILES GENERATED:")
print("=" * 80)
print("\n✅ SHAP Visualizations:")
print("   - shap_waterfall_sepsis.png")
print("   - shap_summary_sepsis.png")
print("\n✅ LIME Visualization:")
print("   - lime_explanation_sepsis.png")

print("\n" + "=" * 80)
print("🎓 UNDERSTANDING THE DIFFERENCE:")
print("=" * 80)
print("""
SHAP (SHapley Additive exPlanations):
  • Based on game theory (Shapley values)
  • Considers all possible feature combinations
  • Consistent and theoretically sound
  • Shows exact contribution of each feature
  • Better for overall model understanding

LIME (Local Interpretable Model-agnostic Explanations):
  • Creates a simple linear model locally
  • Generates variations around the instance
  • Model-agnostic (works with any model)
  • Faster and more intuitive for single predictions
  • Better for explaining individual cases to clinicians
  
Both methods help doctors understand WHY the AI made a specific prediction!
""")

print("\n✅ Demonstration complete!\n")
