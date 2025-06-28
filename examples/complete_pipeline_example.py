"""
Complete pipeline example demonstrating GNN + Pooling + Regressor for ZINC dataset.
This shows how to combine all components for molecular property prediction.
"""
from ACAgraphML.Pipeline.Models.Regressor import Regressor, create_standard_regressor
from ACAgraphML.Pipeline.Models.Pooling import Pooling
from ACAgraphML.Pipeline.Models.GNNmodel import GNNModel
from ACAgraphML.Transforms import OneHotEncodeFeat
from ACAgraphML.Dataset import ZINC_Dataset
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))


class MolecularPropertyPredictor(nn.Module):
    """Complete molecular property prediction pipeline."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        gnn_hidden: int = 128,
        gnn_layers: int = 4,
        gnn_layer_type: str = "GINEConv",
        pooling_type: str = "attentional",
        regressor_type: str = "mlp",
        regressor_hidden_dims: list = None
    ):
        super().__init__()

        if regressor_hidden_dims is None:
            regressor_hidden_dims = [128, 64]

        # GNN for node embeddings
        self.gnn = GNNModel(
            c_in=node_features,
            c_hidden=gnn_hidden,
            c_out=gnn_hidden,  # Keep same size for pooling
            num_layers=gnn_layers,
            layer_name=gnn_layer_type,
            edge_dim=edge_features,
            dp_rate=0.1,
            use_residual=True,
            use_layer_norm=True
        )

        # Pooling for graph embeddings
        pool_output_dim = gnn_hidden
        if pooling_type == "set2set":
            pool_output_dim = gnn_hidden * 2  # Set2Set doubles the dimension

        self.pooling = Pooling(
            pooling_type=pooling_type,
            hidden_dim=gnn_hidden,
            processing_steps=3
        )

        # Regressor for final prediction
        self.regressor = Regressor(
            input_dim=pool_output_dim,
            regressor_type=regressor_type,
            hidden_dims=regressor_hidden_dims,
            mlp_dropout=0.15,
            normalization='batch',
            activation='relu'
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """Forward pass through complete pipeline."""
        # Node embeddings from GNN
        node_embeddings = self.gnn(x, edge_index, edge_attr)

        # Graph embeddings from pooling
        graph_embeddings = self.pooling(node_embeddings, batch)

        # Final predictions from regressor
        predictions = self.regressor(graph_embeddings)

        return predictions


def test_complete_pipeline():
    """Test the complete molecular property prediction pipeline."""
    print("Testing Complete Molecular Property Prediction Pipeline")
    print("=" * 60)

    # Load a small sample of ZINC data
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

    # Load small dataset for testing
    try:
        dataset = ZINC_Dataset.SMALL_TRAIN.load(transform=transform)
        sample_dataset = dataset[:50]  # Small sample for quick testing
        loader = DataLoader(sample_dataset, batch_size=16, shuffle=True)
        print(f"✓ Loaded dataset: {len(sample_dataset)} molecules")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # Test different configurations
    configs = [
        {
            "name": "Baseline (Linear)",
            "gnn_layer": "GCN",
            "pooling": "mean",
            "regressor": "linear",
            "regressor_hidden": [64]
        },
        {
            "name": "Standard (Recommended)",
            "gnn_layer": "GINEConv",
            "pooling": "attentional",
            "regressor": "mlp",
            "regressor_hidden": [128, 64]
        },
        {
            "name": "Advanced (High Performance)",
            "gnn_layer": "GINEConv",
            "pooling": "set2set",
            "regressor": "ensemble_mlp",
            "regressor_hidden": [256, 128, 64]
        }
    ]

    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 40)

        try:
            # Create model
            model = MolecularPropertyPredictor(
                node_features=NUM_NODE_FEATS,
                edge_features=NUM_EDGE_FEATS,
                gnn_hidden=64,  # Smaller for testing
                gnn_layers=3,
                gnn_layer_type=config["gnn_layer"],
                pooling_type=config["pooling"],
                regressor_type=config["regressor"],
                regressor_hidden_dims=config["regressor_hidden"]
            )

            # Count parameters
            num_params = sum(p.numel()
                             for p in model.parameters() if p.requires_grad)
            print(f"  Parameters: {num_params:,}")

            # Test forward pass
            model.eval()
            batch = next(iter(loader))

            with torch.no_grad():
                predictions = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch
                )

            # Validate output
            batch_size = batch.batch.max().item() + 1
            assert predictions.shape == (batch_size,), \
                f"Expected {batch_size} predictions, got {predictions.shape}"
            assert torch.isfinite(predictions).all(), \
                "Predictions contain non-finite values"

            print(f"  Output shape: {predictions.shape}")
            print(
                f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
            print(
                f"  Target range: [{batch.y.min():.3f}, {batch.y.max():.3f}]")

            # Quick training test (1 step)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            predictions_train = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(predictions_train, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"  Training loss: {loss.item():.4f}")
            print(f"  ✓ {config['name']} - All tests passed!")

        except Exception as e:
            print(f"  ✗ {config['name']} - Failed: {e}")

    print(f"\n" + "=" * 60)
    print("Pipeline testing completed!")


def show_regressor_options():
    """Display all available regressor options with justifications for ZINC dataset."""

    print("\nRegressor Options for ZINC Dataset Molecular Property Prediction")
    print("=" * 70)

    options = [
        {
            "name": "Linear Regressor",
            "complexity": "LOW",
            "params": "Minimal (~65 for 64D input)",
            "use_case": "Quick baseline, interpretable results",
            "pros": ["Fast training/inference", "No overfitting risk", "Interpretable"],
            "cons": ["Limited expressivity", "May underfit complex patterns"],
            "zinc_justification": "Good baseline to compare against. ZINC molecular properties might have simple linear relationships with graph embeddings."
        },
        {
            "name": "MLP Regressor",
            "complexity": "MEDIUM",
            "params": "Moderate (~17K for 64→128→64→1)",
            "use_case": "Standard choice, good performance-complexity balance",
            "pros": ["Good expressivity", "Configurable depth", "Batch/layer norm support"],
            "cons": ["More parameters than linear", "Needs hyperparameter tuning"],
            "zinc_justification": "RECOMMENDED: Best balance for ZINC. Molecular properties often have non-linear relationships that MLPs capture well."
        },
        {
            "name": "Residual MLP",
            "complexity": "MEDIUM-HIGH",
            "params": "High (~208K for 3 residual blocks)",
            "use_case": "Deeper networks without vanishing gradients",
            "pros": ["Handles deeper networks", "Stable gradients", "Good for complex patterns"],
            "cons": ["Many parameters", "Slower training", "May overfit"],
            "zinc_justification": "Use when molecular properties require complex feature interactions. Residual connections help with deeper understanding."
        },
        {
            "name": "Attention MLP",
            "complexity": "MEDIUM-HIGH",
            "params": "High (~99K with 4 heads)",
            "use_case": "When feature importance varies across molecules",
            "pros": ["Adaptive feature weighting", "Interpretable attention", "Modern architecture"],
            "cons": ["Complex", "May overfit", "Slower than standard MLP"],
            "zinc_justification": "Good for ZINC when different molecular features have varying importance across different molecules."
        },
        {
            "name": "Ensemble MLP",
            "complexity": "HIGH",
            "params": "Very High (~51K per head × num_heads)",
            "use_case": "Maximum accuracy, when computational resources allow",
            "pros": ["Best accuracy", "Robust predictions", "Handles uncertainty"],
            "cons": ["Most expensive", "Complex tuning", "Risk of overfitting"],
            "zinc_justification": "BEST PERFORMANCE: For ZINC competition or maximum accuracy. Ensemble reduces variance and improves molecular property predictions."
        }
    ]

    for option in options:
        print(f"\n{option['name']} ({option['complexity']} COMPLEXITY)")
        print("-" * 50)
        print(f"Parameters: {option['params']}")
        print(f"Use Case: {option['use_case']}")
        print(f"Pros: {', '.join(option['pros'])}")
        print(f"Cons: {', '.join(option['cons'])}")
        print(f"ZINC Justification: {option['zinc_justification']}")

    print(f"\n" + "=" * 70)
    print("RECOMMENDATIONS FOR ZINC DATASET:")
    print("🥇 FIRST CHOICE: MLP Regressor - Best balance of performance and complexity")
    print("🥈 SECOND CHOICE: Ensemble MLP - When you need maximum accuracy")
    print("🥉 THIRD CHOICE: Linear Regressor - For quick baseline comparison")
    print("\nStart with MLP, then try Ensemble if you need better performance!")


if __name__ == "__main__":
    show_regressor_options()
    test_complete_pipeline()
