#!/usr/bin/env python3
"""
Test Supabase Integration for Optuna

This script tests the Supabase database connection and Optuna integration.
Run this locally or in Kaggle to verify your Supabase setup works correctly.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_supabase_connection():
    """Test basic Supabase connection"""
    print("🧪 Testing Supabase Connection...")
    
    # Get credentials from environment
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_password = os.getenv('SUPABASE_PASSWORD')
    
    if not supabase_url or not supabase_password:
        print("❌ Missing Supabase credentials")
        print("   Set SUPABASE_URL and SUPABASE_PASSWORD environment variables")
        print(f"   SUPABASE_URL: {'✅' if supabase_url else '❌'}")
        print(f"   SUPABASE_PASSWORD: {'✅' if supabase_password else '❌'}")
        return False
    
    try:
        from supabase_integration import create_supabase_storage
        
        # Test connection
        storage = create_supabase_storage(
            supabase_url=supabase_url,
            supabase_password=supabase_password
        )
        
        print("✅ Supabase connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return False


def test_optuna_supabase_integration():
    """Test Optuna with Supabase storage"""
    print("\n🧪 Testing Optuna-Supabase Integration...")
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_password = os.getenv('SUPABASE_PASSWORD')
    
    if not supabase_url or not supabase_password:
        print("❌ Skipping - no Supabase credentials")
        return False
    
    try:
        import optuna
        from supabase_integration import create_supabase_storage, get_or_create_study
        
        # Create storage
        storage = create_supabase_storage(
            supabase_url=supabase_url,
            supabase_password=supabase_password
        )
        
        # Create or load study
        study = get_or_create_study(
            storage=storage,
            study_name="test_study",
            directions=["minimize", "maximize"]
        )
        
        print(f"✅ Study created/loaded: {study.study_name}")
        print(f"   Existing trials: {len(study.trials)}")
        
        # Test adding a trial
        def dummy_objective(trial):
            x = trial.suggest_float('x', -10, 10)
            y = trial.suggest_int('y', 0, 100)
            return x**2, y
        
        study.optimize(dummy_objective, n_trials=2)
        
        print(f"✅ Added test trials. Total trials: {len(study.trials)}")
        
        # Test loading the study again
        study2 = get_or_create_study(
            storage=storage,
            study_name="test_study",
            directions=["minimize", "maximize"]
        )
        
        print(f"✅ Study persistence verified: {len(study2.trials)} trials loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Optuna-Supabase integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_setup():
    """Test environment setup"""
    print("\n🧪 Testing Environment Setup...")
    
    try:
        from utils import setup_environment, log_system_info
        
        setup_environment()
        log_system_info()
        
        print("✅ Environment setup successful!")
        return True
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Supabase Integration Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test 1: Environment Setup
    results.append(test_environment_setup())
    
    # Test 2: Basic Supabase Connection
    results.append(test_supabase_connection())
    
    # Test 3: Optuna-Supabase Integration
    results.append(test_optuna_supabase_integration())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Environment Setup: {'✅' if results[0] else '❌'}")
    print(f"   Supabase Connection: {'✅' if results[1] else '❌'}")
    print(f"   Optuna Integration: {'✅' if results[2] else '❌'}")
    
    all_passed = all(results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if not all_passed:
        print("\n💡 Troubleshooting Tips:")
        if not results[0]:
            print("   - Check Python dependencies are installed")
        if not results[1]:
            print("   - Verify Supabase credentials are correct")
            print("   - Check Supabase project is active")
            print("   - Ensure database is accessible")
        if not results[2]:
            print("   - Verify Optuna and psycopg2-binary are installed")
            print("   - Check database permissions")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
