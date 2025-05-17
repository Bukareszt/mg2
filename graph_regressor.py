import torch
from scipy.stats import pearsonr
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import random
import os
import logging
import gc
import argparse
from tqdm import tqdm
from logger import Logger  # Import the Logger class
from torch_geometric.nn.models import GNNExplainer
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("graph_training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def free_gpu_memory():
    """
    Free up GPU memory by forcing garbage collection and clearing CUDA cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Cleared CUDA cache. Current memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

def set_seed(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class LayerwiseGraphDataset(InMemoryDataset):
    def __init__(self, embeddings_by_layer, labels, edge_mode="sequential"):
        super().__init__()
        self.layer_names = list(embeddings_by_layer.keys())
        self.data_list = self._build_graphs(embeddings_by_layer, labels, self.layer_names, edge_mode)

    def _build_graphs(self, embeddings_by_layer, labels, layer_order, edge_mode):
        graphs = []
        num_tokens = len(labels)

        for i in range(num_tokens):
            node_features = [embeddings_by_layer[layer][i] for layer in layer_order]
            x = torch.stack(node_features)  # [num_layers, hidden_dim]
            y = torch.tensor([labels[i]], dtype=torch.float)

            num_nodes = len(layer_order)
            if edge_mode == "sequential":
                edge_index = [[j, j + 1] for j in range(num_nodes - 1)]
            elif edge_mode == "fully_connected":
                edge_index = [[a, b] for a in range(num_nodes) for b in range(num_nodes) if a != b]
            else:
                raise ValueError(f"Unsupported edge_mode: {edge_mode}")

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            graphs.append(Data(x=x, edge_index=edge_index, y=y))

        return graphs

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]



class GraphRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.linear = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.linear(x).squeeze()


def train(model, loader, optimizer, loss_fn, device, scaler, use_amp=False, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    preds, labels_list = [], []

    # Progress bar
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=use_amp):
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            
        # Backward pass with gradient scaling for mixed precision
        if use_amp:
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
        # Collect predictions and labels
        preds.extend(out.detach().cpu().reshape(-1).numpy())
        labels_list.extend(batch.y.cpu().reshape(-1).numpy())
    
    # Calculate metrics
    metrics = compute_metrics(preds, labels_list)
    metrics["loss"] = total_loss / len(loader)
    
    return metrics


def evaluate(model, loader, loss_fn, device, use_amp=False):
    model.eval()
    total_loss = 0
    preds, labels_list = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation"):
            batch = batch.to(device)
            with autocast(enabled=use_amp):
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(out, batch.y)
            
            total_loss += loss.item()
            preds.extend(out.cpu().reshape(-1).numpy())
            labels_list.extend(batch.y.cpu().reshape(-1).numpy())
    
    
    # Calculate metrics
    metrics = compute_metrics(preds, labels_list)
    metrics["loss"] = total_loss / len(loader)
    
    return metrics


def compute_metrics(preds, labels):
    """Compute regression metrics and length-aware metrics."""
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)

    # Avoid division by zero
    nonzero_mask = labels != 0
    if np.any(nonzero_mask):
        norm_mae = np.mean(np.abs(preds[nonzero_mask] - labels[nonzero_mask]) / labels[nonzero_mask])
    else:
        norm_mae = float('nan')

    # Error vs prompt length correlation
    abs_errors = np.abs(np.array(preds) - np.array(labels))
    try:
        correlation_with_length, _ = pearsonr(abs_errors, labels)
    except Exception as e:
        correlation_with_length = float('nan')  # In case of constant inputs

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "normalized_mae": norm_mae,
        "error_prompt_length_corr": correlation_with_length
    }


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split a dataset into train, validation, and test sets.
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    train_end = int(train_ratio * len(indices))
    val_end = train_end + int(val_ratio * len(indices))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    logger.info(f"Dataset split into {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test examples")
    
    return train_dataset, val_dataset, test_dataset


def load_dataset(path, edge_mode="sequential", length_threshold=None):
    """
    Load dataset without length filtering (moved to separate function)
    """
    logger.info(f"🔄 Loading data from {path}...")
    data = torch.load(path)

    embeddings_by_layer = {k: v for k, v in data.items() if k.startswith("layer_")}
    labels = data["labels"]
    
    # Length filtering was here but is now removed from initial loading

    logger.info(f"Using all {len(embeddings_by_layer)} layers: {list(embeddings_by_layer.keys())}")
    dataset = LayerwiseGraphDataset(embeddings_by_layer, labels, edge_mode=edge_mode)
    logger.info(f"✅ Created dataset with {len(dataset)} graphs (1 per token)")

    return dataset


def filter_dataset_by_length(dataset, length_threshold, upper_threshold=None):
    """
    Filter a dataset to only include tokens with lengths in the specified range
    
    Args:
        dataset: The dataset to filter
        length_threshold: Minimum token length to include
        upper_threshold: Maximum token length to include (if None, no upper limit)
    """
    if length_threshold <= 0 and upper_threshold is None:
        return dataset
    
    filtered_data = []
    
    for data in dataset:
        y_value = data.y.item()
        # Apply lower threshold
        if y_value < length_threshold:
            continue
        
        # Apply upper threshold if specified
        if upper_threshold is not None and y_value > upper_threshold:
            continue
            
        filtered_data.append(data)
    
    if upper_threshold is not None:
        logger.info(f"Filtered dataset by length range [{length_threshold}, {upper_threshold}]: {len(dataset)} → {len(filtered_data)} tokens")
    else:
        logger.info(f"Filtered dataset by length threshold {length_threshold}: {len(dataset)} → {len(filtered_data)} tokens")
    
    return filtered_data


def train_model(args):
    """
    Training function for the graph regressor model.
    """
    logger.info("Starting training with clean memory state")
    free_gpu_memory()
    
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb logger with model name that includes dataset info
    config = vars(args)
    config['loss_type'] = "L1Loss"  # We're using L1Loss
    
    model_name = f"graph-regressor-{args.edge_mode}"
    
    wandb_logger = Logger(
        config=config,
        model_name=model_name,
        project_name=args.wandb_project,
        enable_logging=args.use_wandb,
        log_model=args.log_model
    )
    
    # Load dataset without filtering
    dataset = load_dataset(args.data_path, args.edge_mode)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, 
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed
    )
    
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Determine input dimension
    sample_graph = train_dataset[0]
    input_dim = sample_graph.x.shape[1]
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = GraphRegressor(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize optimizer, loss function, and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(args.num_epochs):
        # Train
        train_metrics = train(
            model, 
            train_loader, 
            optimizer, 
            loss_fn, 
            device,
            scaler,
            use_amp=args.use_amp,
            max_grad_norm=args.max_grad_norm
        )
        
        # Validate
        val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            use_amp=args.use_amp
        )
        
        # Update learning rate
        scheduler.step(val_metrics["loss"])
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Train MAE: {train_metrics['mae']:.4f}")
        logger.info(f"  Train RMSE: {train_metrics['rmse']:.4f}")
        logger.info(f"  Train R²: {train_metrics['r2']:.4f}")
        logger.info(f"  Train Norm. MAE: {train_metrics['normalized_mae']:.4f}")
        logger.info(f"  Train Error-Length Corr: {train_metrics['error_prompt_length_corr']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  Val MAE: {val_metrics['mae']:.4f}")
        logger.info(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        logger.info(f"  Val R²: {val_metrics['r2']:.4f}")
        logger.info(f"  Val Norm. MAE: {val_metrics['normalized_mae']:.4f}")
        logger.info(f"  Val Error-Length Corr: {val_metrics['error_prompt_length_corr']:.4f}")

        # Log metrics to wandb
        if args.use_wandb:
            wandb_metrics = {
                "train/loss": train_metrics['loss'],
                "train/mae": train_metrics['mae'],
                "train/rmse": train_metrics['rmse'],
                "train/r2": train_metrics['r2'],
                "train/normalized_mae": train_metrics['normalized_mae'],
                "train/error_prompt_length_corr": train_metrics['error_prompt_length_corr'],
                "val/loss": val_metrics['loss'],
                "val/mae": val_metrics['mae'],
                "val/rmse": val_metrics['rmse'],
                "val/r2": val_metrics['r2'],
                "lr": optimizer.param_groups[0]['lr'],
                "val/normalized_mae": val_metrics['normalized_mae'],
                "val/error_prompt_length_corr": val_metrics['error_prompt_length_corr'],
            }
            wandb_logger.log_metrics(wandb_metrics, step=epoch)
        
        # Check for improvement and save model
        if val_metrics["loss"] + args.min_loss_improvement < best_val_loss:
            logger.info(f"Validation loss decreased from {best_val_loss:.4f} to {val_metrics['loss']:.4f}. Saving model...")
            best_val_loss = val_metrics["loss"]
            early_stop_counter = 0
            
            # Save the model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics["loss"],
                'input_dim': input_dim,
                'hidden_dim': args.hidden_dim
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
            
            # Log model to wandb
            if args.use_wandb and args.log_model:
                wandb_logger.log_model_checkpoint(
                    model, 
                    os.path.join(args.output_dir, "best_model.pt"),
                    f"best_model_epoch_{epoch}"
                )
        else:
            early_stop_counter += 1
            logger.info(f"Validation loss did not decrease significantly. Early stopping counter: {early_stop_counter}/{args.early_stopping_patience}")
            
            if early_stop_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save the final model
    checkpoint = {
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics["loss"],
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim
    }
    torch.save(checkpoint, os.path.join(args.output_dir, "final_model.pt"))
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Finish wandb logging
    if args.use_wandb:
        wandb_logger.finish()
    
    # Clean up memory
    del model, optimizer, train_dataset, val_dataset, train_loader, val_loader
    free_gpu_memory()
    
    return best_val_loss


def explain_model_predictions(args, model, dataset, device, wandb_logger=None):
    """
    Run GNNExplainer on the trained model to explain predictions.
    """
    logger.info("Running GNNExplainer to interpret model predictions...")
    model.eval()
    
    # Create output directory for explanations
    explanation_dir = os.path.join(args.output_dir, "explanations")
    os.makedirs(explanation_dir, exist_ok=True)
    
    # Initialize explainer
    explainer = GNNExplainer(
        model=model,
        epochs=100,
        return_type='regression'
    )
    
    # Select a subset of examples to explain
    num_examples = min(10, len(dataset))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_indices = indices[:num_examples]
    
    # Track average feature importance across examples
    layer_importance = []
    
    for i, idx in enumerate(selected_indices):
        logger.info(f"Explaining example {i+1}/{num_examples} (dataset index {idx})...")
        
        # Get data sample
        data = dataset[idx].to(device)
        
        # Run explainer
        node_feat_mask, edge_mask = explainer.explain_graph(
            x=data.x, 
            edge_index=data.edge_index,
            batch=None  # Single graph, no batch
        )
        
        # Store node feature importance for later analysis
        layer_importance.append(node_feat_mask.detach().cpu().numpy())
        
        # Log feature and edge importance
        logger.info(f"Example {idx} (true token length: {data.y.item():.1f}):")
        logger.info(f"  Node feature importance: {node_feat_mask.tolist()}")
        logger.info(f"  Edge importance: {edge_mask.tolist()}")
        
        # Visualize and save explanation
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot node feature importance
            plt.subplot(2, 1, 1)
            plt.bar(range(len(node_feat_mask)), node_feat_mask.detach().cpu().numpy())
            plt.title(f"Layer feature importance for example {idx}")
            plt.xlabel("Feature dimension")
            plt.ylabel("Importance")
            
            # Plot edge importance
            plt.subplot(2, 1, 2)
            plt.bar(range(len(edge_mask)), edge_mask.detach().cpu().numpy())
            plt.title(f"Edge importance for example {idx}")
            plt.xlabel("Edge index")
            plt.ylabel("Importance")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(explanation_dir, f"explanation_{idx}.png"))
            
            # Log to wandb
            if args.use_wandb and wandb_logger:
                wandb_logger.log_artifact(
                    os.path.join(explanation_dir, f"explanation_{idx}.png"),
                    f"explanation_{idx}"
                )
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing explanation: {e}")
    
    # Calculate and log average layer importance
    if layer_importance:
        avg_layer_importance = np.mean(layer_importance, axis=0)
        logger.info(f"Average feature importance across {num_examples} examples: {avg_layer_importance.tolist()}")
        
        # Visualize average importance
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(avg_layer_importance)), avg_layer_importance)
        plt.title("Average feature importance across examples")
        plt.xlabel("Feature dimension")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(explanation_dir, "average_importance.png"))
        
        # Log to wandb
        if args.use_wandb and wandb_logger:
            wandb_logger.log_artifact(
                os.path.join(explanation_dir, "average_importance.png"),
                "average_feature_importance"
            )
            
            # Also log as a bar chart in wandb
            if hasattr(wandb_logger, 'run') and wandb_logger.run:
                import wandb
                wandb_logger.run.log({
                    "feature_importance": wandb.Image(
                        os.path.join(explanation_dir, "average_importance.png")
                    )
                })
        
        plt.close()

def evaluate_model(args):
    """
    Evaluation function for the graph regressor model.
    """
    logger.info("Starting evaluation with clean memory state")
    free_gpu_memory()
    
    set_seed(args.seed)
    
    # Initialize wandb logger for evaluation
    wandb_logger = None
    if args.use_wandb:
        config = vars(args)
        config['phase'] = 'evaluation'
        
        model_name = f"eval-graph-regressor-{args.edge_mode}"
        
        wandb_logger = Logger(
            config=config,
            model_name=model_name,
            project_name=args.wandb_project,
            enable_logging=args.use_wandb,
            log_model=False
        )
    
    # Load dataset without filtering
    dataset = load_dataset(args.data_path, args.edge_mode)
    
    # Split dataset
    _, _, test_dataset = split_dataset(
        dataset, 
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed
    )
    
    # Apply length threshold only to test dataset
    if args.length_threshold > 0 or args.length_upper_threshold is not None:
        original_test_size = len(test_dataset)
        test_dataset = filter_dataset_by_length(
            test_dataset, 
            args.length_threshold,
            args.length_upper_threshold
        )
        logger.info(f"Applied length thresholds to test set only: {original_test_size} → {len(test_dataset)} examples")
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the best model
    try:
        checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
        input_dim = checkpoint.get('input_dim')
        hidden_dim = checkpoint.get('hidden_dim', args.hidden_dim)
        
        # If input_dim is not stored in the checkpoint, get it from the dataset
        if input_dim is None:
            sample_graph = test_dataset[0]
            input_dim = sample_graph.x.shape[1]
            
        model = GraphRegressor(input_dim=input_dim, hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {os.path.join(args.output_dir, 'best_model.pt')}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Evaluate
    loss_fn = nn.L1Loss()
    test_metrics = evaluate(model, test_loader, loss_fn, device, use_amp=args.use_amp)
    
    # Log metrics
    logger.info("Test Metrics:")
    logger.info(f"  Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  MAE: {test_metrics['mae']:.4f}")
    logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"  R²: {test_metrics['r2']:.4f}")
    logger.info(f"  Norm. MAE: {test_metrics['normalized_mae']:.4f}")
    logger.info(f"  Error-Length Corr: {test_metrics['error_prompt_length_corr']:.4f}")

    # Log test metrics to wandb
    if args.use_wandb:
        test_metrics_wandb = {
            "test/loss": test_metrics['loss'],
            "test/mae": test_metrics['mae'],
            "test/mse": test_metrics['mse'],
            "test/rmse": test_metrics['rmse'],
            "test/r2": test_metrics['r2'],
            "test/normalized_mae": test_metrics['normalized_mae'],
            "test/error_prompt_length_corr": test_metrics['error_prompt_length_corr'],
        }
        wandb_logger.log_metrics(test_metrics_wandb)
    
    # Run GNNExplainer if requested
    if args.explain:
        explain_model_predictions(args, model, test_dataset, device, wandb_logger)
        
    # Finish wandb logging
    if args.use_wandb and wandb_logger:
        wandb_logger.finish()
    
    # Clean up memory
    del model, test_dataset, test_loader
    free_gpu_memory()
    
    return test_metrics


def main():
    """
    Main function to parse arguments and run training/evaluation.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a graph-based token length predictor")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="trail_dataset_all_layers.pt",
                        help="Path to the pre-extracted embeddings dataset (.pt file)")
    parser.add_argument("--length_threshold", type=int, default=0,
                        help="Minimum token length to include in training (tokens with fewer remaining tokens are filtered out)")
    parser.add_argument("--length_upper_threshold", type=int, default=None,
                        help="Maximum token length to include in evaluation (creates a sliding window with length_threshold)")
    
    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Dimension of hidden layers in the GCN model")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save model and results")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for regularization")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--min_loss_improvement", type=float, default=0.001,
                        help="Minimum validation loss improvement to consider as significant")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # DataLoader optimization arguments
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for data loading")
    
    # Mode arguments
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run evaluation on test set")
    
    # Precision arguments
    parser.add_argument("--use_amp", action="store_true",
                        help="Whether to use automatic mixed precision for training and inference")
    
    parser.add_argument("--edge_mode", type=str, choices=["sequential", "fully_connected"], default="sequential",
                        help="How to connect nodes (layers) in the token graph")
                        
    # Wandb logging arguments
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="graph-length-predictor",
                        help="Weights & Biases project name")
    parser.add_argument("--log_model", action="store_true",
                        help="Whether to log model checkpoints to W&B")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="Weights & Biases group name for experiment comparison")
    parser.add_argument("--explain", action="store_true",
                    help="Run GNNExplainer on the trained model")

    args = parser.parse_args()
    
    if args.do_train:
        train_model(args)
    
    if args.do_eval:
        evaluate_model(args)
    
    logger.info("Script execution completed")


if __name__ == "__main__":
    main()
