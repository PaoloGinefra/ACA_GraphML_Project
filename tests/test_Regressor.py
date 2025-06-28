"""
Unit tests for Regressor class implementations.
Tests all regressor architectures for molecular property prediction on ZINC dataset.
"""
import pytest
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from ACAgraphML.Dataset import ZINC_Dataset
from ACAgraphML.Transforms import OneHotEncodeFeat
from ACAgraphML.Pipeline.Models.Regressor import (
    Regressor,
    LinearRegressor,
    MLPRegressor,
    ResidualMLPRegressor,
    AttentionMLPRegressor,
    EnsembleMLPRegressor,
    create_baseline_regressor,
    create_standard_regressor,
    create_advanced_regressor
)


class TestRegressor:
    """Test class for Regressor implementations."""

    @pytest.fixture(scope="class")
    def sample_data(self):
        """Create sample graph embeddings for testing."""
        batch_size = 16
        input_dims = [32, 64, 128, 256]

        data = {}
        for input_dim in input_dims:
            data[input_dim] = torch.randn(batch_size, input_dim)

        data['batch_size'] = batch_size
        return data

    @pytest.fixture(scope="class")
    def zinc_data(self):
        """Load small ZINC dataset for integration testing."""
        NUM_NODE_FEATS = 28
        NUM_EDGE_FEATS = 4
        oneHotTransform = OneHotEncodeFeat(NUM_NODE_FEATS)

        def transform(data):
            data = oneHotTransform(data)
            data.x = data.x.float()
            data.edge_attr = torch.nn.functional.one_hot(
                data.edge_attr.long(), num_classes=NUM_EDGE_FEATS
            ).float()
            return data

        dataset = ZINC_Dataset.SMALL_TRAIN.load(transform=transform)
        sample_dataset = dataset[:20]  # Small sample for testing
        loader = DataLoader(sample_dataset, batch_size=8, shuffle=False)

        return {
            'loader': loader,
            'num_node_features': NUM_NODE_FEATS,
            'num_edge_features': NUM_EDGE_FEATS
        }

    @pytest.mark.parametrize("regressor_type", [
        "linear", "mlp", "residual_mlp", "attention_mlp", "ensemble_mlp"
    ])
    def test_regressor_types(self, sample_data, regressor_type):
        """Test all regressor types with different input dimensions."""
        for input_dim in [32, 64, 128]:
            x = sample_data[input_dim]
            batch_size = sample_data['batch_size']

            # Create regressor
            regressor = Regressor(
                input_dim=input_dim,
                regressor_type=regressor_type,
                hidden_dims=[64, 32] if regressor_type in [
                    'mlp', 'ensemble_mlp'] else [128, 64],
                mlp_dropout=0.1
            )

            # Test forward pass
            regressor.eval()
            with torch.no_grad():
                predictions = regressor(x)

            # Validate output
            assert predictions.shape == (batch_size,), \
                f"Expected shape ({batch_size},), got {predictions.shape} for {regressor_type}"
            assert torch.isfinite(predictions).all(), \
                f"Predictions contain non-finite values for {regressor_type}"

            # Test training mode
            regressor.train()
            predictions_train = regressor(x)
            assert predictions_train.shape == (batch_size,), \
                f"Training mode failed for {regressor_type}"

    def test_linear_regressor(self, sample_data):
        """Test LinearRegressor specifically."""
        x = sample_data[64]

        # Test without dropout
        regressor = LinearRegressor(input_dim=64, dropout=0.0)
        predictions = regressor(x)
        assert predictions.shape == (sample_data['batch_size'],)

        # Test with dropout
        regressor = LinearRegressor(input_dim=64, dropout=0.5)
        regressor.eval()
        with torch.no_grad():
            predictions = regressor(x)
        assert predictions.shape == (sample_data['batch_size'],)

    def test_mlp_regressor_configurations(self, sample_data):
        """Test MLPRegressor with different configurations."""
        x = sample_data[64]

        configs = [
            {"hidden_dims": [32], "normalization": "none"},
            {"hidden_dims": [128, 64], "normalization": "batch"},
            {"hidden_dims": [64, 32, 16], "normalization": "layer"},
            {"hidden_dims": [128], "activation": "gelu"},
            {"hidden_dims": [64], "activation": "leaky_relu"}
        ]

        for config in configs:
            regressor = MLPRegressor(input_dim=64, **config)
            regressor.eval()
            with torch.no_grad():
                predictions = regressor(x)
            assert predictions.shape == (sample_data['batch_size'],)

    def test_residual_mlp_regressor(self, sample_data):
        """Test ResidualMLPRegressor."""
        x = sample_data[128]

        regressor = ResidualMLPRegressor(
            input_dim=128,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1
        )

        regressor.eval()
        with torch.no_grad():
            predictions = regressor(x)

        assert predictions.shape == (sample_data['batch_size'],)

        # Test parameter count is reasonable
        num_params = sum(p.numel() for p in regressor.parameters())
        assert num_params > 10000, "ResidualMLP should have substantial parameters"

    def test_attention_mlp_regressor(self, sample_data):
        """Test AttentionMLPRegressor."""
        x = sample_data[64]

        regressor = AttentionMLPRegressor(
            input_dim=64,
            hidden_dim=32,
            num_heads=2,
            num_layers=2
        )

        regressor.eval()
        with torch.no_grad():
            predictions = regressor(x)

        assert predictions.shape == (sample_data['batch_size'],)

    def test_ensemble_mlp_regressor(self, sample_data):
        """Test EnsembleMLPRegressor."""
        x = sample_data[64]

        # Test mean aggregation
        regressor_mean = EnsembleMLPRegressor(
            input_dim=64,
            num_heads=3,
            hidden_dims=[32, 16],
            aggregation='mean'
        )

        regressor_mean.eval()
        with torch.no_grad():
            predictions_mean = regressor_mean(x)

        assert predictions_mean.shape == (sample_data['batch_size'],)

        # Test weighted aggregation
        regressor_weighted = EnsembleMLPRegressor(
            input_dim=64,
            num_heads=3,
            hidden_dims=[32, 16],
            aggregation='weighted'
        )

        regressor_weighted.eval()
        with torch.no_grad():
            predictions_weighted = regressor_weighted(x)

        assert predictions_weighted.shape == (sample_data['batch_size'],)

        # Test that weighted and mean give different results
        assert not torch.allclose(predictions_mean, predictions_weighted), \
            "Mean and weighted aggregation should give different results"

    def test_convenience_functions(self, sample_data):
        """Test convenience functions for quick model creation."""
        x = sample_data[64]

        # Test baseline regressor
        baseline = create_baseline_regressor(input_dim=64)
        baseline.eval()
        with torch.no_grad():
            pred_baseline = baseline(x)
        assert pred_baseline.shape == (sample_data['batch_size'],)

        # Test standard regressor
        standard = create_standard_regressor(input_dim=64)
        standard.eval()
        with torch.no_grad():
            pred_standard = standard(x)
        assert pred_standard.shape == (sample_data['batch_size'],)

        # Test advanced regressor
        advanced = create_advanced_regressor(input_dim=64)
        advanced.eval()
        with torch.no_grad():
            pred_advanced = advanced(x)
        assert pred_advanced.shape == (sample_data['batch_size'],)

    def test_parameter_counts(self, sample_data):
        """Test that parameter counts are reasonable for each regressor type."""
        input_dim = 64

        regressors = {
            "linear": Regressor(input_dim, "linear"),
            "mlp": Regressor(input_dim, "mlp", hidden_dims=[128, 64]),
            "residual_mlp": Regressor(input_dim, "residual_mlp", residual_hidden_dim=64),
            "attention_mlp": Regressor(input_dim, "attention_mlp", attention_hidden_dim=64),
            "ensemble_mlp": Regressor(input_dim, "ensemble_mlp", ensemble_num_heads=3)
        }

        param_counts = {}
        for name, regressor in regressors.items():
            param_counts[name] = sum(p.numel() for p in regressor.parameters())

        # Check ordering: linear < mlp < others
        assert param_counts["linear"] < param_counts["mlp"], \
            "Linear should have fewer parameters than MLP"

        # Check all models have reasonable parameter counts
        for name, count in param_counts.items():
            assert count > 0, f"{name} should have parameters"
            assert count < 1_000_000, f"{name} has too many parameters: {count:,}"

    def test_training_integration(self, sample_data):
        """Test that regressors can be trained with gradient descent."""
        x = sample_data[64]
        target = torch.randn(sample_data['batch_size'])  # Random targets

        regressor = create_standard_regressor(input_dim=64)
        optimizer = torch.optim.Adam(regressor.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Training loop
        initial_loss = None
        for epoch in range(5):
            regressor.train()
            optimizer.zero_grad()

            predictions = regressor(x)
            loss = criterion(predictions, target)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Check that loss changed (model is learning)
        assert abs(final_loss - initial_loss) > 1e-6, \
            "Loss should change during training"

    @pytest.mark.parametrize("pooling_output_dim", [32, 64, 128, 256])
    def test_zinc_compatibility(self, zinc_data, pooling_output_dim):
        """Test compatibility with typical ZINC dataset pooling outputs."""
        # Simulate graph embeddings after GNN + pooling
        batch = next(iter(zinc_data['loader']))
        batch_size = batch.batch.max().item() + 1
        graph_embeddings = torch.randn(batch_size, pooling_output_dim)

        # Test standard regressor
        regressor = create_standard_regressor(input_dim=pooling_output_dim)
        regressor.eval()

        with torch.no_grad():
            predictions = regressor(graph_embeddings)

        assert predictions.shape == (batch_size,), \
            f"Wrong prediction shape for batch size {batch_size}"
        assert torch.isfinite(predictions).all(), \
            "Predictions should be finite"

        # Test that predictions are in reasonable range for ZINC
        # ZINC targets are typically in range [-3, 5]
        assert predictions.abs().max() < 10, \
            "Predictions should be in reasonable range for molecular properties"

    def test_invalid_regressor_type(self, sample_data):
        """Test that invalid regressor types raise appropriate errors."""
        with pytest.raises(ValueError, match="Unknown regressor type"):
            Regressor(input_dim=64, regressor_type="invalid_type")

    def test_edge_cases(self, sample_data):
        """Test edge cases and boundary conditions."""
        # Test with minimum input dimension
        regressor = Regressor(input_dim=1, regressor_type="linear")
        x_small = torch.randn(4, 1)
        predictions = regressor(x_small)
        assert predictions.shape == (4,)

        # Test with single sample
        regressor = create_standard_regressor(input_dim=64)
        x_single = sample_data[64][:1]
        regressor.eval()
        with torch.no_grad():
            predictions = regressor(x_single)
        assert predictions.shape == (1,)

        # Test with zero dropout
        regressor = Regressor(
            input_dim=64,
            regressor_type="mlp",
            mlp_dropout=0.0
        )
        predictions = regressor(sample_data[64])
        assert predictions.shape == (sample_data['batch_size'],)


if __name__ == "__main__":
    # Run tests manually (simplified)
    print("Running Regressor tests manually...")

    # Create sample data manually
    batch_size = 16
    sample_data = {
        32: torch.randn(batch_size, 32),
        64: torch.randn(batch_size, 64),
        128: torch.randn(batch_size, 128),
        256: torch.randn(batch_size, 256),
        'batch_size': batch_size
    }

    test_instance = TestRegressor()

    try:
        test_instance.test_convenience_functions(sample_data)
        print("✓ Convenience functions test passed")

        test_instance.test_parameter_counts(sample_data)
        print("✓ Parameter counts test passed")

        test_instance.test_training_integration(sample_data)
        print("✓ Training integration test passed")

        test_instance.test_linear_regressor(sample_data)
        print("✓ Linear regressor test passed")

        test_instance.test_mlp_regressor_configurations(sample_data)
        print("✓ MLP regressor configurations test passed")

        print("All manual tests passed! Run with pytest for full suite.")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
