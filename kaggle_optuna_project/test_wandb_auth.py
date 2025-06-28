"""
Test W&B authentication and create a test project.
"""
import wandb
import os

def test_wandb_auth():
    """Test W&B authentication and create a test run."""
    try:
        print("🔑 Testing Weights & Biases authentication...")
        
        # Test login
        wandb.login()
        print("✅ W&B authentication successful!")
        
        # Create a test run
        run = wandb.init(
            project="gnn-hyperopt-test",
            name="authentication_test",
            config={
                "test_param": 123,
                "framework": "optuna",
                "dataset": "ZINC"
            },
            tags=["test", "setup"]
        )
        
        print(f"✅ Test project created: {run.project}")
        print(f"✅ Test run created: {run.name}")
        print(f"🌐 Run URL: {run.url}")
        
        # Log some test metrics
        wandb.log({
            "test_mae": 0.5,
            "test_memory_gb": 2.1,
            "test_throughput": 150.0
        })
        
        print("✅ Test metrics logged successfully!")
        
        # Finish the run
        wandb.finish()
        print("✅ W&B test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ W&B authentication failed: {e}")
        return False

if __name__ == "__main__":
    success = test_wandb_auth()
    if success:
        print("\n🎉 Weights & Biases is ready for your optimization!")
    else:
        print("\n❌ Please check your W&B setup.")
