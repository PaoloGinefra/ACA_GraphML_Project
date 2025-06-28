#!/usr/bin/env python3
"""
Quick Test Configuration for Kaggle

This script provides a minimal configuration for testing the optimization pipeline
on Kaggle with a small number of trials and reduced complexity.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from optuna_multiobjective_optimization import OptimizationConfig, MultiObjectiveOptimizer


def create_test_config():
    """Create optimized configuration for quick testing."""
    return OptimizationConfig(
        study_name="kaggle_quick_test",
        n_trials=3,  # Very small for quick test
        max_epochs=5,  # Minimal epochs
        early_stopping_patience=2,
        wandb_project="gnn-quick-test",
        batch_size_options=[32, 64],  # Smaller batches
        memory_limit_gb=12.0,
        time_limit_hours=0.5,  # 30 minutes max
    )


def run_quick_test():
    """Run a quick test with minimal configuration."""
    print("🧪 Quick Test Configuration")
    print("=" * 40)
    
    # Create test config
    config = create_test_config()
    print(f"Trials: {config.n_trials}")
    print(f"Max Epochs: {config.max_epochs}")
    print(f"Time Limit: {config.time_limit_hours} hours")
    
    # Check for Supabase credentials
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_password = os.getenv('SUPABASE_PASSWORD')
    use_supabase = bool(supabase_url and supabase_password)
    
    supabase_config = {
        'url': supabase_url,
        'password': supabase_password
    } if use_supabase else None
    
    print(f"Supabase: {'✅ Enabled' if use_supabase else '❌ Disabled'}")
    
    # Initialize optimizer
    try:
        print("\n🚀 Starting quick test...")
        optimizer = MultiObjectiveOptimizer(
            config=config,
            use_supabase=use_supabase,
            supabase_config=supabase_config
        )
        
        # Run optimization
        optimizer.run_optimization()
        print("\n✅ Quick test completed successfully!")
        
        # Show basic results
        if os.path.exists("optimization_results.csv"):
            import pandas as pd
            df = pd.read_csv("optimization_results.csv")
            print(f"\n📊 Results: {len(df)} trials completed")
            if len(df) > 0:
                best_idx = df['val_mae'].idxmin()
                print(f"Best MAE: {df.loc[best_idx, 'val_mae']:.4f}")
                print(f"Best Model: {df.loc[best_idx, 'gnn_type']}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_quick_test()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)
