import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import random
import os
import logging
import gc
import argparse
from tqdm import tqdm

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


class HiddenStatesGraphDataset(InMemoryDataset):
    def __init__(self, embeddings, lengths, threshold=0.9, max_seq_len=None):
        super().__init__()
        self.data_list = self._build_graphs(embeddings, lengths, threshold, max_seq_len)

    def _build_graphs(self, embeddings, lengths, threshold, max_seq_len):
        graphs = []
        i = 0
        while i < len(lengths):
            seq = []
            original_length = 0
            while i < len(lengths) and lengths[i] != 0:
                seq.append(embeddings[i])
                original_length += 1
                i += 1
            if i < len(lengths):
                seq.append(embeddings[i])
                original_length += 1
                i += 1

            if len(seq) < 2:
                continue

            # Cut sequence to max_seq_len if set
            if max_seq_len is not None:
                seq = seq[:max_seq_len]

            x = torch.stack(seq)
            y = torch.tensor([original_length], dtype=torch.float)  # target is full sequence length

            sim = cosine_similarity(x.numpy())
            edge_index = []
            for a in range(len(sim)):
                for b in range(len(sim)):
                    if a != b and sim[a, b] > threshold:
                        edge_index.append([a, b])
            if not edge_index:
                edge_index = [[i, i + 1] for i in range(len(x) - 1)]

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
        preds.extend(out.detach().cpu().numpy())
        labels_list.extend(batch.y.cpu().numpy())
    
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
            preds.extend(out.cpu().numpy())
            labels_list.extend(batch.y.cpu().numpy())
    
    # Calculate metrics
    metrics = compute_metrics(preds, labels_list)
    metrics["loss"] = total_loss / len(loader)
    
    return metrics


def compute_metrics(preds, labels):
    """Compute regression metrics."""
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
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


def load_dataset(path, layer, threshold=0.9, max_seq_len=None):
    logger.info(f"🔄 Loading data from {path}...")
    data = torch.load(path)

    if layer not in data:
        available_layers = [key for key in data.keys() if key != "labels"]
        raise ValueError(f"Layer '{layer}' not found in the dataset. Available layers: {available_layers}")

    embeddings = data[layer]
    labels = data["labels"]

    logger.info(f"Creating graph dataset with threshold {threshold} and max_seq_len={max_seq_len}...")
    dataset = HiddenStatesGraphDataset(embeddings, labels, threshold, max_seq_len)
    logger.info(f"Created dataset with {len(dataset)} graphs")
    return dataset



def train_model(args):
    """
    Training function for the graph regressor model.
    """
    logger.info("Starting training with clean memory state")
    free_gpu_memory()
    
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset and create graph dataset
    dataset = load_dataset(args.data_path, args.layer_name, args.threshold, args.max_seq_len)
    
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
    loss_fn = nn.MSELoss()
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
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  Val MAE: {val_metrics['mae']:.4f}")
        logger.info(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        logger.info(f"  Val R²: {val_metrics['r2']:.4f}")
        
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
    
    # Clean up memory
    del model, optimizer, train_dataset, val_dataset, train_loader, val_loader
    free_gpu_memory()
    
    return best_val_loss


def evaluate_model(args):
    """
    Evaluation function for the graph regressor model.
    """
    logger.info("Starting evaluation with clean memory state")
    free_gpu_memory()
    
    set_seed(args.seed)
    
    # Load dataset and create graph dataset
    dataset = load_dataset(args.data_path, args.layer_name, args.threshold, args.max_seq_len)
    
    # Split dataset
    _, _, test_dataset = split_dataset(
        dataset, 
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed
    )
    
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
    loss_fn = nn.MSELoss()
    test_metrics = evaluate(model, test_loader, loss_fn, device, use_amp=args.use_amp)
    
    # Log metrics
    logger.info("Test Metrics:")
    logger.info(f"  Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  MAE: {test_metrics['mae']:.4f}")
    logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"  R²: {test_metrics['r2']:.4f}")
    
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
    parser.add_argument("--layer_name", type=str, default="layer_13",
                        help="Name of the layer to use for embeddings (e.g., layer_13)")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Cosine similarity threshold for creating graph edges")
    
    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Dimension of hidden layers in the GCN model")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save model and results")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
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
    
    parser.add_argument("--max_seq_len", type=int, default=5,
                    help="Maximum number of tokens used per input sequence")
    
    args = parser.parse_args()
    
    if args.do_train:
        train_model(args)
    
    if args.do_eval:
        evaluate_model(args)
    
    logger.info("Script execution completed")


if __name__ == "__main__":
    main()
