"""
Verification Script for Trained Models
==========================================
Verifies all 4 disease-specific models can be loaded and used for inference.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def verify_models():
    """Load and verify all trained models."""
    
    models_dir = Path("trained_models")
    diseases = ["sepsis", "kidney_failure", "heart_disease", "diabetes", 
                "anemia", "thalassemia", "thrombocytopenia", "mortality"]
    
    print("=" * 60)
    print("Model Verification")
    print("=" * 60)
    
    # Create synthetic test patient
    test_patient = pd.DataFrame({
        'age': [65],
        'gender': [1],
        'heart_rate': [95],
        'systolic_bp': [140],
        'diastolic_bp': [85],
        'temperature': [37.8],
        'respiratory_rate': [20],
        'wbc_count': [12.5],
        'hemoglobin': [13.2],
        'platelet_count': [220],
        'creatinine': [1.3],
        'bun': [22],
        'glucose': [110],
        'lactate': [2.1]
    })
    
    results = {}
    
    for disease in diseases:
        model_path = models_dir / f"{disease}_xgboost_v1.0.0.pkl"
        
        print(f"\n{'=' * 60}")
        print(f"Testing {disease.upper()} Model")
        print(f"{'=' * 60}")
        
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            continue
            
        # Load model bundle
        try:
            bundle = joblib.load(model_path)
            print(f"✅ Loaded: {model_path}")
            print(f"   Size: {model_path.stat().st_size / 1024:.1f} KB")
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            continue
            
        # Verify bundle structure
        required_keys = ['model', 'scaler', 'feature_names', 'metrics', 'metadata']
        for key in required_keys:
            if key in bundle:
                print(f"✅ Contains '{key}'")
            else:
                print(f"❌ Missing '{key}'")
                
        # Test prediction
        try:
            # Transform features
            X_scaled = bundle['scaler'].transform(test_patient)
            
            # Make prediction
            pred_proba = bundle['model'].predict_proba(X_scaled)[0, 1]
            pred_class = bundle['model'].predict(X_scaled)[0]
            
            print(f"\n📊 Test Prediction:")
            print(f"   Risk Probability: {pred_proba:.3f}")
            print(f"   Predicted Class: {'POSITIVE' if pred_class == 1 else 'NEGATIVE'}")
            
            # Show metrics
            metrics = bundle['metrics']
            print(f"\n📈 Training Metrics:")
            print(f"   AUROC: {metrics['auroc']:.3f}")
            print(f"   Accuracy: {metrics['accuracy']:.3f}")
            print(f"   F1 Score: {metrics['f1_score']:.3f}")
            print(f"   Prevalence: {metrics['prevalence']:.1%}")
            
            results[disease] = {
                'status': 'SUCCESS',
                'prediction': pred_proba,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            results[disease] = {'status': 'FAILED', 'error': str(e)}
    
    # Summary
    print(f"\n{'=' * 60}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 60}")
    
    success_count = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
    total_count = len(diseases)
    
    print(f"\n✅ {success_count}/{total_count} models verified successfully")
    
    if success_count == total_count:
        print("\n🎉 All models are ready for deployment!")
        print("\n📝 Next Steps:")
        print("   1. Run: python enhanced_dashboard_with_whatif.py")
        print("   2. Open: http://localhost:8050")
        print("   3. Test predictions with SHAP explanations")
    else:
        print(f"\n⚠️  {total_count - success_count} model(s) failed verification")
    
    return results

if __name__ == "__main__":
    results = verify_models()
