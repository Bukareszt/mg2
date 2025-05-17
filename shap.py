import torch
import shap
import argparse
import numpy as np
from torch_geometric.loader import DataLoader
from model import GraphRegressor, LayerwiseGraphDataset  # Ensure these are importable
from train_script import load_dataset  # Your function
import matplotlib.pyplot as plt
import os
import wandb

def explain_layer_contributions(model, dataset, device, output_dir=None, use_wandb=False, num_samples=100):
    """
    Runs SHAP analysis to explain the contribution of each layer in the graph.

    Parameters:
        model: Trained GNN model
        dataset: PyG dataset of graphs
        device: 'cuda' or 'cpu'
        output_dir: Directory to save outputs
        use_wandb: Whether to log results to wandb
        num_samples: Number of graphs to sample for SHAP
    """
    model.eval()
    model.to(device)

    background = dataset[:num_samples]  # Select samples for background
    explainer_input = []
    edge_indices = []
    batch_indices = []

    for idx, data in enumerate(background):
        explainer_input.append(data.x)  # x: [num_layers, dim]
        edge_indices.append(data.edge_index)
        batch_indices.append(torch.full((data.x.size(0),), idx))  # Mark nodes for batch graph idx

    x_all = torch.cat(explainer_input, dim=0).to(device)
    edge_index_all = torch.cat([ei + i * ei.max() + 1 if i > 0 else ei for i, ei in enumerate(edge_indices)], dim=1).to(device)
    batch_all = torch.cat(batch_indices).to(device)

    def model_forward(x):
        # x: [total_nodes, dim]
        return model(x, edge_index_all, batch_all)

    # Choose a single sample graph to explain
    sample_graph = dataset[num_samples]
    sample_x = sample_graph.x.clone().detach().to(device).requires_grad_(True)
    sample_edge_index = sample_graph.edge_index.to(device)
    sample_batch = torch.zeros(sample_x.size(0), dtype=torch.long).to(device)

    def predict_fn(input_tensor):
        # input_tensor: shape [num_explanations, num_nodes * input_dim]
        outputs = []
        for row in input_tensor:
            reshaped_x = row.reshape(sample_x.shape)
            out = model(reshaped_x.to(device), sample_edge_index, sample_batch)
            outputs.append(out.item())
        return np.array(outputs)

    # SHAP KernelExplainer (model agnostic, slower but works for any black-box model)
    background_data = sample_x.cpu().numpy().reshape(1, -1)
    explainer = shap.KernelExplainer(predict_fn, background_data)

    shap_values = explainer.shap_values(sample_x.cpu().numpy().reshape(1, -1))
    shap_values = np.array(shap_values).reshape(sample_x.shape)

    # Plot SHAP values per layer
    layer_scores = np.linalg.norm(shap_values, axis=1)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(layer_scores)), layer_scores)
    plt.xlabel("Layer index (node in graph)")
    plt.ylabel("SHAP importance (L2 norm of vector)")
    plt.title("Layer-wise SHAP values")
    plt.tight_layout()
    
    # Determine output file path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "shap_layer_importance.png")
    else:
        output_file = "shap_layer_importance.png"
    
    plt.savefig(output_file)
    print(f"✅ SHAP layer importance plot saved as '{output_file}'.")
    
    # Log to wandb if enabled
    if use_wandb:
        # Log the figure
        wandb.log({"shap_layer_importance": wandb.Image(plt)})
        
        # Also log the raw layer scores as a bar chart for interactive viewing
        data = [[i, score] for i, score in enumerate(layer_scores)]
        table = wandb.Table(data=data, columns=["layer_index", "importance_score"])
        wandb.log({"layer_importance_scores": wandb.plot.bar(
            table, "layer_index", "importance_score", 
            title="Layer-wise SHAP Importance Scores")})
        
        # Log individual layer importances for easier comparison across runs
        for i, score in enumerate(layer_scores):
            wandb.log({f"layer_{i}_importance": score})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--edge_mode", type=str, default="sequential")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--length_threshold", type=int, default=0)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="embeddings-length-predictor", help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name")
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name)
        # Log model configuration
        config = {
            "model_path": args.model_path,
            "edge_mode": args.edge_mode,
            "hidden_dim": args.hidden_dim,
            "length_threshold": args.length_threshold
        }
        wandb.config.update(config)

    # Set output directory to model directory if not specified
    if not args.output_dir:
        args.output_dir = os.path.dirname(args.model_path)

    # Load dataset
    dataset = load_dataset(args.data_path, edge_mode=args.edge_mode, length_threshold=args.length_threshold)

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    model = GraphRegressor(input_dim=checkpoint['input_dim'], hidden_dim=checkpoint['hidden_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    explain_layer_contributions(model, dataset, device, output_dir=args.output_dir, use_wandb=args.use_wandb)

    # Close wandb run if enabled
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
