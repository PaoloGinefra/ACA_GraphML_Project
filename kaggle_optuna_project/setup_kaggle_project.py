"""
Setup script to copy ACAgraphML package to the Kaggle project directory.

This ensures we have all necessary source code for deployment.
"""

import shutil
import os
from pathlib import Path


def copy_acagraphml_package():
    """
    Copy the ACAgraphML package to the Kaggle project src directory.
    """
    # Paths
    project_root = Path(__file__).parent.parent
    src_acagraphml = project_root / "src" / "ACAgraphML"
    kaggle_src = Path(__file__).parent / "src" / "ACAgraphML"
    
    # Remove existing if present
    if kaggle_src.exists():
        shutil.rmtree(kaggle_src)
    
    # Copy the entire ACAgraphML package
    shutil.copytree(src_acagraphml, kaggle_src)
    
    print(f"✅ Copied ACAgraphML package from {src_acagraphml} to {kaggle_src}")
    
    # Also copy the main package __init__.py files
    src_init = project_root / "src" / "__init__.py"
    if src_init.exists():
        kaggle_init = Path(__file__).parent / "src" / "__init__.py"
        shutil.copy2(src_init, kaggle_init)
    
    return kaggle_src


def copy_data_subset():
    """
    Copy a subset of ZINC data for testing (optional).
    """
    project_root = Path(__file__).parent.parent
    source_data = project_root / "data" / "ZINC"
    kaggle_data = Path(__file__).parent / "data" / "ZINC"
    
    if source_data.exists():
        if kaggle_data.exists():
            shutil.rmtree(kaggle_data)
        shutil.copytree(source_data, kaggle_data)
        print(f"✅ Copied ZINC data from {source_data} to {kaggle_data}")
    else:
        print(f"⚠️  ZINC data not found at {source_data}")
        # Create empty directory structure
        kaggle_data.mkdir(parents=True, exist_ok=True)
        (kaggle_data / "raw").mkdir(exist_ok=True)
        (kaggle_data / "processed").mkdir(exist_ok=True)
        print(f"📁 Created empty data directory structure at {kaggle_data}")


def update_imports_for_kaggle():
    """
    Update import statements in pipeline_components.py to work in Kaggle environment.
    """
    pipeline_components_file = Path(__file__).parent / "src" / "pipeline_components.py"
    
    # Read the current content
    with open(pipeline_components_file, 'r') as f:
        content = f.read()
    
    # Update the import section to work in both local and Kaggle environments
    updated_content = content.replace(
        '''# Add the source directory to path to import ACAgraphML
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))''',
        '''# Add the source directory to path to import ACAgraphML
# This works both locally and in Kaggle environment
current_dir = Path(__file__).parent
if (current_dir / "ACAgraphML").exists():
    # In Kaggle environment - ACAgraphML is in the same directory
    sys.path.insert(0, str(current_dir))
else:
    # In local environment - go up to find src
    project_root = current_dir.parent.parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))'''
    )
    
    # Write back the updated content
    with open(pipeline_components_file, 'w') as f:
        f.write(updated_content)
    
    print("✅ Updated import statements for Kaggle compatibility")


def create_kaggle_requirements():
    """
    Create a comprehensive requirements file for Kaggle.
    """
    requirements_content = """# PyTorch and related
torch>=2.0.0
torch-geometric>=2.3.0
pytorch-lightning>=2.0.0

# Hyperparameter optimization
optuna>=3.3.0

# Experiment tracking
wandb>=0.15.0

# Data processing and scientific computing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Performance monitoring
psutil>=5.8.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
tqdm>=4.62.0
pyyaml>=6.0

# Kaggle API
kaggle>=1.5.12

# Chemistry/molecules (if needed)
rdkit-pypi>=2022.3.5

# Additional dependencies that might be needed
networkx>=2.6
scipy>=1.7.0
"""
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    with open(requirements_file, 'w') as f:
        f.write(requirements_content)
    
    print(f"✅ Created comprehensive requirements file at {requirements_file}")


def verify_setup():
    """
    Verify that all necessary files are in place.
    """
    base_path = Path(__file__).parent
    
    required_files = [
        "optuna_multiobjective_optimization.py",
        "deploy_to_kaggle.py",
        "requirements.txt",
        "src/pipeline_components.py",
        "src/performance_monitor.py", 
        "src/utils.py",
        "src/ACAgraphML/__init__.py",
        "configs/optimization_config.yaml"
    ]
    
    print("\n📋 Verifying setup...")
    all_good = True
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_good = False
    
    if all_good:
        print("\n🎉 Setup verification complete! All files are ready for Kaggle deployment.")
    else:
        print("\n⚠️  Some files are missing. Please check the setup.")
    
    return all_good


def main():
    """
    Main setup function.
    """
    print("🚀 Setting up Kaggle optimization project...")
    print("=" * 60)
    
    # Step 1: Copy ACAgraphML package
    copy_acagraphml_package()
    
    # Step 2: Update imports for Kaggle compatibility
    update_imports_for_kaggle()
    
    # Step 3: Copy data (optional)
    copy_data_subset()
    
    # Step 4: Create comprehensive requirements
    create_kaggle_requirements()
    
    # Step 5: Verify setup
    setup_ok = verify_setup()
    
    if setup_ok:
        print("\n" + "=" * 60)
        print("🎯 Setup complete! Next steps:")
        print("1. Configure your Weights & Biases API key")
        print("2. Set up Kaggle API credentials")
        print("3. Run: python deploy_to_kaggle.py")
        print("=" * 60)
    
    return setup_ok


if __name__ == "__main__":
    main()
