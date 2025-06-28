"""
Optuna Supabase Database Integration

This module provides functionality to connect Optuna to a Supabase PostgreSQL database
for persistent study storage across multiple Kaggle runs.
"""

import optuna
from optuna.storages import RDBStorage
import os
from typing import Optional
import logging

def create_supabase_storage(
    supabase_url: str,
    supabase_password: str,
    database_name: str = "postgres",
    username: str = "postgres"
) -> RDBStorage:
    """
    Create an Optuna RDBStorage connected to Supabase PostgreSQL.
    
    Args:
        supabase_url: Your Supabase project URL (without https://)
        supabase_password: Your Supabase database password
        database_name: Database name (default: postgres)
        username: Database username (default: postgres)
    
    Returns:
        RDBStorage: Configured Optuna storage backend
    """
    
    # Construct PostgreSQL connection URL
    # Format: postgresql://username:password@host:port/database
    db_url = f"postgresql://{username}:{supabase_password}@db.{supabase_url}:5432/{database_name}"
    
    try:
        storage = RDBStorage(
            url=db_url,
            engine_kwargs={
                "pool_size": 20,
                "max_overflow": 0,
                "pool_pre_ping": True,
                "pool_recycle": 300,
                "connect_args": {
                    "sslmode": "require",  # Supabase requires SSL
                    "connect_timeout": 10,
                }
            }
        )
        
        # Test connection
        storage._engine.connect()
        print(f"✅ Successfully connected to Supabase database!")
        return storage
        
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        raise


def get_or_create_study(
    storage: RDBStorage,
    study_name: str,
    directions: list = None
) -> optuna.Study:
    """
    Get existing study or create new one with multi-objective optimization.
    
    Args:
        storage: Optuna storage backend
        study_name: Name of the study
        directions: List of optimization directions
    
    Returns:
        optuna.Study: The study object
    """
    
    if directions is None:
        directions = [
            "minimize",  # Validation MAE
            "minimize",  # Memory usage
            "maximize",  # Throughput
            "minimize",  # Latency
            "minimize"   # Training time
        ]
    
    try:
        # Try to load existing study
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        print(f"✅ Loaded existing study '{study_name}' with {len(study.trials)} trials")
        
    except KeyError:
        # Create new study if it doesn't exist
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=directions,
            load_if_exists=True
        )
        print(f"✅ Created new study '{study_name}'")
    
    return study


class SupabaseOptimizer:
    """
    Enhanced MultiObjectiveOptimizer with Supabase integration.
    """
    
    def __init__(self, config, supabase_config: dict):
        """
        Initialize optimizer with Supabase database connection.
        
        Args:
            config: OptimizationConfig object
            supabase_config: Dict with Supabase connection details
                {
                    'url': 'your-project.supabase.co',
                    'password': 'your-database-password'
                }
        """
        self.config = config
        self.supabase_config = supabase_config
        
        # Create Supabase storage
        self.storage = create_supabase_storage(
            supabase_url=supabase_config['url'],
            supabase_password=supabase_config['password']
        )
        
        # Create study with persistent storage
        self.study = get_or_create_study(
            storage=self.storage,
            study_name=config.study_name,
            directions=[
                "minimize",  # Validation MAE
                "minimize",  # Memory usage
                "maximize",  # Throughput (samples/sec)
                "minimize",  # Latency (inference time)
                "minimize"   # Training time
            ]
        )
        
        print(f"📊 Study statistics:")
        print(f"   - Total trials: {len(self.study.trials)}")
        print(f"   - Completed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"   - Failed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    
    def get_study_summary(self) -> dict:
        """Get comprehensive study summary."""
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]
        
        summary = {
            'total_trials': len(self.study.trials),
            'completed_trials': len(completed_trials),
            'failed_trials': len(failed_trials),
            'success_rate': len(completed_trials) / len(self.study.trials) if self.study.trials else 0,
        }
        
        if completed_trials:
            # Get best values for each objective
            best_mae = min(trial.values[0] for trial in completed_trials)
            best_memory = min(trial.values[1] for trial in completed_trials)
            best_throughput = max(trial.values[2] for trial in completed_trials)
            best_latency = min(trial.values[3] for trial in completed_trials)
            best_training_time = min(trial.values[4] for trial in completed_trials)
            
            summary.update({
                'best_mae': best_mae,
                'best_memory_gb': best_memory,
                'best_throughput_samples_sec': best_throughput,
                'best_latency_ms': best_latency,
                'best_training_time_min': best_training_time
            })
        
        return summary


def setup_supabase_from_secrets() -> Optional[dict]:
    """
    Setup Supabase configuration from Kaggle secrets.
    
    Expected secrets:
    - SUPABASE_URL: Your Supabase project URL
    - SUPABASE_PASSWORD: Your database password
    
    Returns:
        dict or None: Supabase configuration or None if secrets not available
    """
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        
        supabase_config = {
            'url': user_secrets.get_secret("SUPABASE_URL"),
            'password': user_secrets.get_secret("SUPABASE_PASSWORD")
        }
        
        print("✅ Supabase configuration loaded from Kaggle secrets")
        return supabase_config
        
    except Exception as e:
        print(f"⚠️  Could not load Supabase secrets: {e}")
        print("Make sure to add SUPABASE_URL and SUPABASE_PASSWORD to Kaggle secrets")
        return None


def create_supabase_tables_sql() -> str:
    """
    Generate SQL commands to create necessary tables in Supabase.
    
    Returns:
        str: SQL commands to execute in Supabase SQL editor
    """
    
    return """
-- Optuna tables for study persistence
-- Run this in your Supabase SQL editor

-- Enable the required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- The tables will be created automatically by Optuna when first connecting
-- But you can check the connection with:
SELECT current_database(), current_user, version();

-- Optional: Create a view for easy trial analysis
CREATE OR REPLACE VIEW trial_summary AS
SELECT 
    study_id,
    trial_id,
    trial_number,
    state,
    datetime_start,
    datetime_complete,
    (datetime_complete - datetime_start) as duration,
    value as objective_value
FROM trials 
WHERE state = 'COMPLETE'
ORDER BY trial_number DESC;
"""


# Example usage configuration
EXAMPLE_SUPABASE_CONFIG = {
    "url": "your-project-id.supabase.co",  # Replace with your Supabase project URL
    "password": "your-secure-password"      # Replace with your database password
}

# Instructions for setup
SETUP_INSTRUCTIONS = """
🔧 Supabase Setup Instructions:

1. Create a Supabase Project:
   - Go to https://supabase.com
   - Create a new project
   - Note your project URL: https://your-project.supabase.co

2. Get Database Password:
   - Go to Settings > Database
   - Note your database password

3. Add to Kaggle Secrets:
   - Go to Kaggle Account > Secrets
   - Add Secret: SUPABASE_URL = your-project.supabase.co
   - Add Secret: SUPABASE_PASSWORD = your-database-password

4. Test Connection:
   - Run: python -c "from supabase_integration import setup_supabase_from_secrets; setup_supabase_from_secrets()"

5. View Results:
   - All optimization results will be stored in your Supabase database
   - Access via Supabase dashboard or connect with any PostgreSQL client
"""
