"""
Diagnostic script to test why your model is giving constant predictions
Run this script to diagnose the issue with your ML model
"""

import pickle
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def diagnose_pkl_file():
    """Inspect the pickle file contents"""
    print("=" * 60)
    print("DIAGNOSING MODEL FILE")
    print("=" * 60)
    
    try:
        with open('partnership_success_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        print("\n1. File Contents:")
        print(f"   - Keys in pickle: {list(artifacts.keys())}")
        
        model = artifacts.get('model')
        scaler = artifacts.get('scaler')
        feature_names = artifacts.get('feature_names', [])
        
        print(f"\n2. Model Info:")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Has predict_proba: {hasattr(model, 'predict_proba')}")
        
        print(f"\n3. Scaler Info:")
        print(f"   - Scaler type: {type(scaler).__name__ if scaler else 'None'}")
        if scaler and hasattr(scaler, 'mean_'):
            print(f"   - Scaler mean values: {scaler.mean_[:5]}... (first 5)")
            print(f"   - Scaler scale values: {scaler.scale_[:5]}... (first 5)")
        
        print(f"\n4. Features Info:")
        print(f"   - Number of features: {len(feature_names)}")
        if feature_names:
            print(f"   - First 10 features: {feature_names[:10]}")
            print(f"   - Feature name examples:")
            for i, name in enumerate(feature_names[:5]):
                print(f"      {i+1}. '{name}'")
        
        return artifacts
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_predictions(artifacts):
    """Test model predictions with different inputs"""
    if not artifacts:
        return
    
    print("\n" + "=" * 60)
    print("TESTING PREDICTIONS")
    print("=" * 60)
    
    model = artifacts.get('model')
    scaler = artifacts.get('scaler')
    feature_names = artifacts.get('feature_names', [])
    
    if not feature_names:
        print("No feature names found! Using default 28 features.")
        feature_names = [f'feature_{i}' for i in range(28)]
    
    # Test Case 1: All zeros
    test1 = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)
    
    # Test Case 2: Random values
    np.random.seed(42)
    test2 = pd.DataFrame([np.random.randn(len(feature_names))], columns=feature_names)
    
    # Test Case 3: High values
    test3 = pd.DataFrame([[10] * len(feature_names)], columns=feature_names)
    
    # Test Case 4: Mixed values
    mixed_values = [i * 0.1 for i in range(len(feature_names))]
    test4 = pd.DataFrame([mixed_values], columns=feature_names)
    
    test_cases = [
        ("All zeros", test1),
        ("Random values", test2),
        ("High values", test3),
        ("Mixed incremental", test4)
    ]
    
    for name, test_data in test_cases:
        print(f"\nTest: {name}")
        print(f"  Input shape: {test_data.shape}")
        print(f"  Input mean: {test_data.mean().mean():.3f}")
        print(f"  Input std: {test_data.std().mean():.3f}")
        
        try:
            # Apply scaler if available
            if scaler:
                test_scaled = scaler.transform(test_data)
                print(f"  Scaled mean: {np.mean(test_scaled):.3f}")
                print(f"  Scaled std: {np.std(test_scaled):.3f}")
            else:
                test_scaled = test_data
            
            # Get prediction
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(test_scaled)[0]
                print(f"  Prediction: {pred[1]:.4f} (class probabilities: {pred})")
            else:
                pred = model.predict(test_scaled)[0]
                print(f"  Prediction: {pred:.4f}")
                
        except Exception as e:
            print(f"  Error: {e}")

def create_realistic_test():
    """Create a realistic test with actual partnership data"""
    print("\n" + "=" * 60)
    print("REALISTIC PARTNERSHIP TEST")
    print("=" * 60)
    
    # Two realistic partnership scenarios
    scenarios = [
        {
            'name': 'Large Tech Partnership',
            'primary_company_revenue': 5000000000,  # $5B
            'partner_company_revenue': 3000000000,   # $3B
            'primary_company_size': 5000,
            'partner_company_size': 3000,
            'primary_reputation_score': 0.85,
            'partner_reputation_score': 0.80,
            'primary_growth_rate': 0.25,
            'partner_growth_rate': 0.20,
            'estimated_value': 100000000,  # $100M
            'primary_sustainability_score': 8.0,
            'partner_sustainability_score': 7.5,
            'primary_industry': 'Technology',
            'partner_industry': 'Technology',
        },
        {
            'name': 'Small Startup Partnership',
            'primary_company_revenue': 5000000,  # $5M
            'partner_company_revenue': 3000000,  # $3M
            'primary_company_size': 50,
            'partner_company_size': 30,
            'primary_reputation_score': 0.60,
            'partner_reputation_score': 0.55,
            'primary_growth_rate': 0.50,
            'partner_growth_rate': 0.45,
            'estimated_value': 500000,  # $500K
            'primary_sustainability_score': 6.0,
            'partner_sustainability_score': 5.5,
            'primary_industry': 'Technology',
            'partner_industry': 'Healthcare',
        }
    ]
    
    # Load your actual model class
    from main import PartnershipPredictionModel  # Adjust import as needed
    
    model_wrapper = PartnershipPredictionModel()
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"  Revenue ratio: ${scenario['primary_company_revenue']/1e9:.1f}B vs ${scenario['partner_company_revenue']/1e9:.1f}B")
        print(f"  Industries: {scenario['primary_industry']} × {scenario['partner_industry']}")
        
        # Get prediction
        prob = model_wrapper.predict_success_probability(scenario)
        print(f"  Success Probability: {prob:.1%}")
        
        # Check if it's the constant 15.4%
        if abs(prob - 0.154) < 0.001:
            print("  ⚠️ WARNING: Got the constant 15.4% prediction!")

if __name__ == "__main__":
    # Run diagnostics
    artifacts = diagnose_pkl_file()
    
    if artifacts:
        test_predictions(artifacts)
        # Uncomment the next line if you have your main file available
        # create_realistic_test()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    print("\nPossible issues if getting constant predictions:")
    print("1. Feature names mismatch between training and prediction")
    print("2. Scaler expecting different feature ranges")
    print("3. Model overfitted to specific value during training")
    print("4. Missing or zero-valued features during prediction")
    print("\nRecommended fix:")
    print("- Retrain model with proper feature tracking")
    print("- Ensure feature names and order match exactly")
    print("- Check that input data normalization matches training")