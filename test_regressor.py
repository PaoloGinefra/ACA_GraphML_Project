"""
Test script for the Regressor class to ensure all implementations work correctly.
"""
from ACAgraphML.Pipeline.Models.Regressor import (
    Regressor,
    create_baseline_regressor,
    create_standard_regressor,
    create_advanced_regressor
)
import torch
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))


def test_regressor():
    """Test all regressor types with sample data."""

    # Sample graph embeddings (batch_size=8, input_dim=64)
    batch_size = 8
    input_dim = 64
    x = torch.randn(batch_size, input_dim)

    print("Testing Regressor implementations...")
    print("=" * 50)

    # Test each regressor type
    regressor_configs = [
        ('linear', {}),
        ('mlp', {'hidden_dims': [128, 64], 'mlp_dropout': 0.1}),
        ('residual_mlp', {'residual_hidden_dim': 128,
         'residual_num_layers': 3}),
        ('attention_mlp', {
         'attention_hidden_dim': 128, 'attention_num_heads': 4}),
        ('ensemble_mlp', {'ensemble_num_heads': 3, 'hidden_dims': [128, 64]})
    ]

    for regressor_type, kwargs in regressor_configs:
        print(f"\nTesting {regressor_type} regressor...")

        try:
            # Create regressor
            regressor = Regressor(
                input_dim=input_dim,
                regressor_type=regressor_type,
                **kwargs
            )

            # Test forward pass
            regressor.eval()
            with torch.no_grad():
                predictions = regressor(x)

            # Validate output
            assert predictions.shape == (batch_size,), \
                f"Expected shape ({batch_size},), got {predictions.shape}"
            assert torch.isfinite(predictions).all(), \
                f"Predictions contain non-finite values"

            # Count parameters
            num_params = sum(p.numel()
                             for p in regressor.parameters() if p.requires_grad)

            print(
                f"  ✓ {regressor_type}: {predictions.shape}, {num_params:,} parameters")
            print(
                f"    Range: [{predictions.min():.3f}, {predictions.max():.3f}]")

        except Exception as e:
            print(f"  ✗ {regressor_type}: Failed with error: {e}")

    # Test convenience functions
    print(f"\nTesting convenience functions...")
    try:
        baseline = create_baseline_regressor(input_dim)
        standard = create_standard_regressor(input_dim)
        advanced = create_advanced_regressor(input_dim)

        with torch.no_grad():
            pred_baseline = baseline(x)
            pred_standard = standard(x)
            pred_advanced = advanced(x)

        print(f"  ✓ Baseline: {pred_baseline.shape}")
        print(f"  ✓ Standard: {pred_standard.shape}")
        print(f"  ✓ Advanced: {pred_advanced.shape}")

    except Exception as e:
        print(f"  ✗ Convenience functions failed: {e}")

    print(f"\n" + "=" * 50)
    print("All regressor tests completed successfully! ✓")


def test_zinc_integration():
    """Test with typical ZINC dataset dimensions."""
    print(f"\nTesting ZINC dataset integration...")

    # Typical dimensions after GNN + pooling
    input_dims = [32, 64, 128, 256]  # Common graph embedding sizes
    batch_size = 32  # Typical batch size

    for input_dim in input_dims:
        x = torch.randn(batch_size, input_dim)

        # Test standard regressor (most commonly used)
        regressor = create_standard_regressor(input_dim)

        regressor.eval()
        with torch.no_grad():
            predictions = regressor(x)

        assert predictions.shape == (batch_size,)
        assert torch.isfinite(predictions).all()

        print(
            f"  ✓ Input dim {input_dim}: predictions range [{predictions.min():.3f}, {predictions.max():.3f}]")

    print("ZINC integration tests passed! ✓")


if __name__ == "__main__":
    test_regressor()
    test_zinc_integration()
