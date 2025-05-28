import pandas as pd
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
from torch_geometric.explain import GNNExplainer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import random
import os
import logging
import gc
import argparse
import csv  # Add CSV module import
from tqdm import tqdm
from logger import Logger  # Import the Logger class
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("graph_training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def run_explainer(model, dataset, device, output_dir, wandb_logger=None, num_examples=1):
    """
    Run GNNExplainer on the trained model and log results to wandb if available.
    """
    model.eval()
    explainer = GNNExplainer(model, return_type='regression')

    for i in range(num_examples):
        data = dataset[i].to(device)
        try:
            # Use the correct method name for newer PyTorch Geometric versions
            explanation = explainer(data.x, data.edge_index, epochs=30)
            logger.info(f"Explanation for example {i} obtained.")

            # Save explanation masks
            node_feat_mask = explanation.node_feat_mask.cpu().detach().numpy()
            edge_mask = explanation.edge_mask.cpu().detach().numpy()

            # Save to files
            node_mask_path = os.path.join(output_dir, f'node_feat_mask_{i}.npy')
            edge_mask_path = os.path.join(output_dir, f'edge_mask_{i}.npy')
            
            np.save(node_mask_path, node_feat_mask)
            np.save(edge_mask_path, edge_mask)

            # Create and save visualization
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.bar(range(len(node_feat_mask)), node_feat_mask)
            plt.title(f"Node Feature Importance (Example {i})")
            plt.xlabel("Feature Index")
            plt.ylabel("Importance")
            
            plt.subplot(1, 2, 2)
            plt.bar(range(len(edge_mask)), edge_mask)
            plt.title(f"Edge Importance (Example {i})")
            plt.xlabel("Edge Index")
            plt.ylabel("Importance")
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"explanation_plot_{i}.png")
            plt.savefig(plot_path)
            plt.close()  # Close the figure to free memory
            
            logger.info(f"Saved explanation visualization for example {i}.")
            
            # Log to wandb if available
            if wandb_logger:
                # Log the explanation masks as artifacts
                wandb_logger.log_artifact(node_mask_path, f"node_feat_mask_example_{i}", "explanation")
                wandb_logger.log_artifact(edge_mask_path, f"edge_mask_example_{i}", "explanation")
                wandb_logger.log_artifact(plot_path, f"explanation_plot_example_{i}", "explanation")
                
                # Log summary statistics
                explanation_metrics = {
                    f"explanation/example_{i}/node_feat_importance_mean": float(np.mean(node_feat_mask)),
                    f"explanation/example_{i}/node_feat_importance_std": float(np.std(node_feat_mask)),
                    f"explanation/example_{i}/node_feat_importance_max": float(np.max(node_feat_mask)),
                    f"explanation/example_{i}/edge_importance_mean": float(np.mean(edge_mask)),
                    f"explanation/example_{i}/edge_importance_std": float(np.std(edge_mask)),
                    f"explanation/example_{i}/edge_importance_max": float(np.max(edge_mask)),
                    f"explanation/example_{i}/true_length": float(data.y.item()),
                }
                wandb_logger.log_metrics(explanation_metrics, step=i)

        except Exception as e:
            logger.error(f"Failed to explain example {i}: {e}")
            # Log the error details for debugging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")


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

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        if batch is not None:
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
    
    # Return metrics and the raw predictions/labels for detailed analysis
    return metrics, preds, labels_list


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


def load_dataset(path, edge_mode="sequential", length_threshold=None, layer_names=None):
    """
    Load dataset without length filtering (moved to separate function)
    """
    logger.info(f"🔄 Loading data from {path}...")
    data = torch.load(path)

    embeddings_by_layer = {k: v for k, v in data.items() if k.startswith("layer_")}
    
    # Filter layers based on layer_names argument
    if layer_names:
        filtered_embeddings = {}
        for layer_name in layer_names:
            if layer_name in embeddings_by_layer:
                filtered_embeddings[layer_name] = embeddings_by_layer[layer_name]
            else:
                logger.warning(f"Layer {layer_name} not found in dataset. Available layers: {list(embeddings_by_layer.keys())}")
        embeddings_by_layer = filtered_embeddings
        logger.info(f"Using selected {len(embeddings_by_layer)} layers: {list(embeddings_by_layer.keys())}")
    else:
        logger.info(f"Using all {len(embeddings_by_layer)} layers: {list(embeddings_by_layer.keys())}")
    
    labels = data["labels"]
    
    # Length filtering was here but is now removed from initial loading

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
    
    # Create model name that includes layer information
    if args.layer_names:
        layer_suffix = "_".join(args.layer_names)
        model_name = f"graph-regressor-{args.edge_mode}-layers-{layer_suffix}"
    else:
        model_name = f"graph-regressor-{args.edge_mode}-all-layers"
    
    wandb_logger = Logger(
        config=config,
        model_name=model_name,
        project_name=args.wandb_project,
        enable_logging=args.use_wandb,
        log_model=args.log_model
    )
    
    # Load dataset without filtering, but with layer selection
    dataset = load_dataset(args.data_path, args.edge_mode, layer_names=args.layer_names)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, 
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed
    )
    
    # Apply length thresholds to all datasets
    if args.length_threshold > 0 or args.length_upper_threshold is not None:
        original_train_size = len(train_dataset)
        original_val_size = len(val_dataset)
        original_test_size = len(test_dataset)
        
        train_dataset = filter_dataset_by_length(
            train_dataset, 
            args.length_threshold,
            args.length_upper_threshold
        )
        
        val_dataset = filter_dataset_by_length(
            val_dataset, 
            args.length_threshold,
            args.length_upper_threshold
        )
        
        test_dataset = filter_dataset_by_length(
            test_dataset, 
            args.length_threshold,
            args.length_upper_threshold
        )
        
        logger.info(f"Applied length thresholds to all datasets:")
        logger.info(f"  Training: {original_train_size} → {len(train_dataset)} examples")
        logger.info(f"  Validation: {original_val_size} → {len(val_dataset)} examples")
        logger.info(f"  Test: {original_test_size} → {len(test_dataset)} examples")
    
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
        val_results = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            use_amp=args.use_amp
        )
        val_metrics, val_preds, val_labels = val_results
        
        # Save validation predictions to CSV
        if epoch == 0 or val_metrics["loss"] + args.min_loss_improvement < best_val_loss:
            # Save token predictions to CSV
            preds_csv_path = os.path.join(args.output_dir, f"val_predictions_epoch_{epoch+1}.csv")
            with open(preds_csv_path, 'w', newline='') as csvfile:
                fieldnames = ['true_length', 'predicted_length']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for true_len, pred_len in zip(val_labels, val_preds):
                    writer.writerow({
                        'true_length': f"{true_len:.1f}",
                        'predicted_length': f"{pred_len:.1f}"
                    })
            
            # Log predictions CSV to wandb
            if args.use_wandb:
                wandb_logger.log_artifact(preds_csv_path, f"val_predictions_epoch_{epoch+1}", "predictions")
        
        # Update learning rate
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]['lr']
        
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
                "lr": current_lr,
                "val/normalized_mae": val_metrics['normalized_mae'],
                "val/error_prompt_length_corr": val_metrics['error_prompt_length_corr'],
            }
            wandb_logger.log_metrics(wandb_metrics, step=epoch)
            
            # Create validation results CSV path
            csv_path = os.path.join(args.output_dir, "validation_results.csv")
            
            # Save validation metrics to CSV for this epoch
            with open(csv_path, 'a', newline='') as csvfile:
                fieldnames = ['epoch', 'loss', 'mae', 'mse', 'rmse', 'r2', 'normalized_mae', 
                             'error_prompt_length_corr', 'lr']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header if the file is new
                if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
                    writer.writeheader()
                    
                writer.writerow({
                    'epoch': epoch + 1,
                    'loss': f"{val_metrics['loss']:.6f}",
                    'mae': f"{val_metrics['mae']:.6f}",
                    'mse': f"{val_metrics['mse']:.6f}",
                    'rmse': f"{val_metrics['rmse']:.6f}",
                    'r2': f"{val_metrics['r2']:.6f}",
                    'normalized_mae': f"{val_metrics['normalized_mae']:.6f}",
                    'error_prompt_length_corr': f"{val_metrics['error_prompt_length_corr']:.6f}",
                    'lr': f"{current_lr:.8f}"
                })
            
            # Log CSV as an artifact to wandb
            wandb_logger.log_artifact(csv_path, f"validation_results_epoch_{epoch+1}", "metrics")
        
        # Check for improvement and save model
        if val_metrics["loss"] + args.min_loss_improvement < best_val_loss:
            logger.info(f"Validation loss decreased from {best_val_loss:.4f} to {val_metrics['loss']:.4f}. Saving model...")
            best_val_loss = val_metrics["loss"]
            early_stop_counter = 0
            
            # Save the model with layer information
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics["loss"],
                'input_dim': input_dim,
                'hidden_dim': args.hidden_dim,
                'layer_names': args.layer_names  # Save layer names used
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
    
    # Save the final model with layer information
    checkpoint = {
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics["loss"],
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'layer_names': args.layer_names  # Save layer names used
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

def plot_mae_vs_distance_from_end(preds, labels, output_dir=None, title="MAE vs Distance from End", window=3, min_count=5):
    """
    Plots MAE as a function of the distance from the end of the generation (i.e., true token length),
    including standard deviation and smoothing.
    
    Args:
        preds: List or array of predicted token lengths
        labels: List or array of true token lengths
        output_dir: If provided, saves the plot as a PNG in this directory
        title: Plot title
        window: Size of smoothing window (rolling average)
        min_count: Minimum number of samples per bin to include in the plot
    """
    preds = np.array(preds)
    labels = np.array(labels)

    # Create DataFrame and compute absolute error
    df = pd.DataFrame({'true_length': labels, 'pred': preds})
    df['error'] = np.abs(df['true_length'] - df['pred'])

    # Group by true length
    grouped = df.groupby('true_length').agg(
        mae=('error', 'mean'),
        std=('error', 'std'),
        count=('error', 'count')
    ).reset_index()

    # Filter out low-count bins
    grouped = grouped[grouped['count'] >= min_count]

    # Apply smoothing using rolling window
    grouped['mae_smooth'] = grouped['mae'].rolling(window=window, center=True).mean()
    grouped['std_smooth'] = grouped['std'].rolling(window=window, center=True).mean()

    # Plot with shaded standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(grouped['true_length'], grouped['mae_smooth'], label='MAE (smoothed)', color='blue')
    plt.fill_between(
        grouped['true_length'],
        grouped['mae_smooth'] - grouped['std_smooth'],
        grouped['mae_smooth'] + grouped['std_smooth'],
        color='blue',
        alpha=0.2,
        label='±1 STD'
    )
    
    plt.xlabel("Distance from End (True Token Length)")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if output_dir:
        plot_path = os.path.join(output_dir, "mae_vs_distance_from_end.png")
        plt.savefig(plot_path)
        logger.info(f"Saved MAE vs Distance plot to {plot_path}")
    else:
        plt.show()


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
        
        # Create model name that includes layer information
        if args.layer_names:
            layer_suffix = "_".join(args.layer_names)
            model_name = f"eval-graph-regressor-{args.edge_mode}-layers-{layer_suffix}"
        else:
            model_name = f"eval-graph-regressor-{args.edge_mode}-all-layers"
        
        wandb_logger = Logger(
            config=config,
            model_name=model_name,
            project_name=args.wandb_project,
            enable_logging=args.use_wandb,
            log_model=False
        )
    
    # Load dataset without filtering, but with layer selection
    dataset = load_dataset(args.data_path, args.edge_mode, layer_names=args.layer_names)
    
    # Split dataset
    _, _, test_dataset = split_dataset(
        dataset, 
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed
    )
    
    # Apply length threshold to test dataset
    if args.length_threshold > 0 or args.length_upper_threshold is not None:
        original_test_size = len(test_dataset)
        test_dataset = filter_dataset_by_length(
            test_dataset, 
            args.length_threshold,
            args.length_upper_threshold
        )
        logger.info(f"Applied length thresholds to test set: {original_test_size} → {len(test_dataset)} examples")
    
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
        saved_layer_names = checkpoint.get('layer_names')
        
        # Verify layer names match if both are specified
        if args.layer_names and saved_layer_names:
            if set(args.layer_names) != set(saved_layer_names):
                logger.warning(f"Layer names mismatch! Model trained on {saved_layer_names}, but evaluation requested {args.layer_names}")
        
        # If input_dim is not stored in the checkpoint, get it from the dataset
        if input_dim is None:
            sample_graph = test_dataset[0]
            input_dim = sample_graph.x.shape[1]
            
        model = GraphRegressor(input_dim=input_dim, hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {os.path.join(args.output_dir, 'best_model.pt')}")
        if saved_layer_names:
            logger.info(f"Model was trained on layers: {saved_layer_names}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Evaluate
    loss_fn = nn.L1Loss()
    test_results = evaluate(model, test_loader, loss_fn, device, use_amp=args.use_amp)
    test_metrics, test_preds, test_labels = test_results
    
    # Log metrics
    logger.info("Test Metrics:")
    logger.info(f"  Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  MAE: {test_metrics['mae']:.4f}")
    logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"  R²: {test_metrics['r2']:.4f}")
    logger.info(f"  Norm. MAE: {test_metrics['normalized_mae']:.4f}")
    logger.info(f"  Error-Length Corr: {test_metrics['error_prompt_length_corr']:.4f}")

    # Log test metrics to wandb
    if args.use_wandb and wandb_logger:
        test_wandb_metrics = {
            "test/loss": test_metrics['loss'],
            "test/mae": test_metrics['mae'],
            "test/rmse": test_metrics['rmse'],
            "test/r2": test_metrics['r2'],
            "test/normalized_mae": test_metrics['normalized_mae'],
            "test/error_prompt_length_corr": test_metrics['error_prompt_length_corr'],
        }
        wandb_logger.log_metrics(test_wandb_metrics, step=0)

    # Save test results to CSV
    csv_path = os.path.join(args.output_dir, "test_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['loss', 'mae', 'mse', 'rmse', 'r2', 'normalized_mae', 'error_prompt_length_corr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'loss': f"{test_metrics['loss']:.6f}",
            'mae': f"{test_metrics['mae']:.6f}",
            'mse': f"{test_metrics['mse']:.6f}",
            'rmse': f"{test_metrics['rmse']:.6f}",
            'r2': f"{test_metrics['r2']:.6f}",
            'normalized_mae': f"{test_metrics['normalized_mae']:.6f}",
            'error_prompt_length_corr': f"{test_metrics['error_prompt_length_corr']:.6f}"
        })

    # Create and save MAE vs Distance plot
    plot_mae_vs_distance_from_end(
        test_preds, 
        test_labels, 
        output_dir=args.output_dir,
        title="MAE vs Distance from End (Test Set)"
    )
    
    # Save token predictions to CSV
    preds_csv_path = os.path.join(args.output_dir, "test_predictions.csv")
    with open(preds_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['true_length', 'predicted_length', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for true_len, pred_len in zip(test_labels, test_preds):
            error = abs(true_len - pred_len)
            writer.writerow({
                'true_length': f"{true_len:.1f}",
                'predicted_length': f"{pred_len:.1f}",
                'error': f"{error:.1f}"
            })
    
    # Log predictions CSV and plot to wandb
    if args.use_wandb:
        wandb_logger.log_artifact(preds_csv_path, "test_predictions", "predictions")
        # Log the plot to wandb
        plot_path = os.path.join(args.output_dir, "mae_vs_distance_from_end.png")
        wandb_logger.log_artifact(plot_path, "mae_vs_distance_plot", "plot")
    
    # Run explainer if requested (before cleaning up model and dataset)
    if args.explain:
        logger.info("🔍 Running GNNExplainer...")
        try:
            # Create explanation output directory
            explanation_dir = os.path.join(args.output_dir, "explanations")
            os.makedirs(explanation_dir, exist_ok=True)
            
            # Run explainer on a subset of test examples
            num_examples = min(5, len(test_dataset))  # Explain up to 5 examples
            run_explainer(
                model, 
                test_dataset, 
                device, 
                explanation_dir, 
                wandb_logger=wandb_logger if args.use_wandb else None,
                num_examples=num_examples
            )
            logger.info(f"✅ Completed explanations for {num_examples} examples")
        except Exception as e:
            logger.error(f"❌ Failed to run explainer: {e}")
    
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
                        help="Run GNNExplainer on the trained model (only works with --do_eval)")
    parser.add_argument('--layer_names', nargs='+', default=None, 
                        help="List of layer names to use (e.g., layer_8 layer_16). If not specified, all layers will be used.")

    args = parser.parse_args()
    
    if args.do_train:
        train_model(args)
    
    if args.do_eval:
        evaluate_model(args)
    elif args.explain:
        logger.warning("⚠️  --explain flag requires --do_eval to be set. Explainer runs during evaluation.")
    
    logger.info("Script execution completed")


if __name__ == "__main__":
    main()
