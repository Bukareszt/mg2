import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import logging
import gc
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
from logger import Logger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
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

class TokenLengthPredictor(nn.Module):
    """
    A simple MLP regressor that predicts the number of remaining tokens in an LLM output sequence.
    Input: Embedding vector from a transformer layer
    Output: A single scalar value representing the predicted token length
    """
    def __init__(self, input_dim, hidden_dim=512):
        super(TokenLengthPredictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, 1)  # Output a single value for regression
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x.squeeze(-1)  # Remove last dimension to get [batch_size] instead of [batch_size, 1]

def set_seed(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class PreExtractedEmbeddingDataset(Dataset):
    """
    Dataset class for pre-extracted embeddings and their corresponding labels.
    """
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx]
        }

def custom_collate_fn(batch):
    """
    Custom collate function to handle the dataset.
    """
    embeddings = torch.stack([item["embedding"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)
    
    return {"embeddings": embeddings, "labels": labels}

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

def load_and_split_dataset(data_path, layer_name, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Load the dataset from a .pt file and split it into train, validation, and test sets.
    
    Args:
        data_path: Path to the .pt file containing the dataset
        layer_name: Name of the layer to use for embeddings (e.g., "layer_8")
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    logger.info(f"Loading dataset from {data_path}...")
    
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load the dataset
    try:
        data = torch.load(data_path)
        
        # Check if the specified layer exists in the dataset
        if layer_name not in data:
            available_layers = [key for key in data.keys() if key != "labels"]
            raise ValueError(f"Layer '{layer_name}' not found in the dataset. Available layers: {available_layers}")
            
        # Get embeddings and labels
        embeddings = data[layer_name]
        labels = data["labels"].float()  # Convert labels to float
        
        logger.info(f"Dataset loaded with {len(labels)} examples")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Create indices for splitting
        indices = np.random.permutation(len(labels))
        train_end = int(train_ratio * len(labels))
        val_end = train_end + int(val_ratio * len(labels))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create datasets
        train_dataset = PreExtractedEmbeddingDataset(
            embeddings[train_indices],
            labels[train_indices]
        )
        
        val_dataset = PreExtractedEmbeddingDataset(
            embeddings[val_indices],
            labels[val_indices]
        )
        
        test_dataset = PreExtractedEmbeddingDataset(
            embeddings[test_indices],
            labels[test_indices]
        )
        
        logger.info(f"Dataset split into {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test examples")
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def train_model(args):
    """Training function."""
    logger.info("Starting training with clean memory state")
    
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb logger with model name that includes dataset info
    config = vars(args)
    config['loss_type'] = "L1Loss"  # We're using L1Loss
    config['layer_name'] = args.layer_name
    
    wandb_logger = Logger(
        config=config,
        model_name=f"embedings_prediction-layer{args.layer_name}",
        project_name=args.wandb_project,
        enable_logging=args.use_wandb,
        log_model=args.log_model
    )
    
    # Load and split dataset
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(
        args.data_path, 
        args.layer_name,
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15, 
        seed=args.seed
    )
    
    # Get embedding dimension from the dataset
    sample_item = train_dataset[0]
    input_dim = sample_item["embedding"].shape[0]
    logger.info(f"Input embedding dimension: {input_dim}")
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=custom_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=custom_collate_fn
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model, loss function, and optimizer
    model = TokenLengthPredictor(input_dim=input_dim, hidden_dim=args.hidden_dim)
    model.to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()
    
    # Initialize learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = args.early_stopping_patience
    
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Prediction and label collections for metric calculation
        train_preds = []
        train_labels = []
        
        for batch in train_pbar:
            embeddings = batch["embeddings"].to(device)
            labels = batch["labels"].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=args.use_amp):
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling for mixed precision
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update weights with gradient scaling for mixed precision
            scaler.step(optimizer)
            scaler.update()
            
            # Update training loss and progress bar
            train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})
            
            # Collect predictions and labels for metric calculation
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate average training loss and metrics
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        train_metrics = compute_metrics(train_preds, train_labels)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                embeddings = batch["embeddings"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass with mixed precision
                with autocast(enabled=args.use_amp):
                    outputs = model(embeddings)
                
                # Calculate loss
                # print 10 random values of outputs and labels
                print(outputs[0:10])
                print(labels[0:10])
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate average validation loss and metrics
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_metrics = compute_metrics(val_preds, val_labels)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Train MAE: {train_metrics['mae']:.4f}")
        logger.info(f"  Train RMSE: {train_metrics['rmse']:.4f}")
        logger.info(f"  Train R²: {train_metrics['r2']:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Val MAE: {val_metrics['mae']:.4f}")
        logger.info(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        logger.info(f"  Val R²: {val_metrics['r2']:.4f}")
        
        # Log metrics to wandb
        if args.use_wandb:
            wandb_metrics = {
                "train/l1_loss": avg_train_loss,
                "train/mae": train_metrics['mae'],
                "train/rmse": train_metrics['rmse'],
                "train/r2": train_metrics['r2'],
                "val/l1_loss": avg_val_loss,
                "val/mae": val_metrics['mae'],
                "val/rmse": val_metrics['rmse'],
                "val/r2": val_metrics['r2'],
                "lr": optimizer.param_groups[0]['lr']
            }
            wandb_logger.log_metrics(wandb_metrics, step=epoch)
        
        # Update learning rate based on validation loss
        lr_scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss + args.min_loss_improvement < best_val_loss:
            logger.info(f"Validation loss decreased from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            
            # Save the model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
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
            logger.info(f"Validation loss did not decrease significantly. Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            
            if early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Clean up at the end of training
    logger.info("Training finished, cleaning up memory")
    
    # Save the final model regardless of performance
    checkpoint = {
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim
    }
    torch.save(checkpoint, os.path.join(args.output_dir, "final_model.pt"))
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Finish wandb logging
    if args.use_wandb:
        wandb_logger.finish()
    
    # Final memory cleanup
    del model, optimizer, train_dataset, val_dataset, train_dataloader, val_dataloader
    
    return best_val_loss

def evaluate(args):
    """Evaluation function on the test set."""
    logger.info("Starting evaluation with clean memory state")
    
    set_seed(args.seed)
    

    # Initialize wandb logger for evaluation
    if args.use_wandb:
        config = vars(args)
        config['phase'] = 'evaluation'
        config['layer_name'] = args.layer_name
        
        wandb_logger = Logger(
            config=config,
            model_name=f"eval-embed-predictor-layer{args.layer_name}",
            project_name=args.wandb_project,
            enable_logging=args.use_wandb,
            log_model=False
        )
    
    # Load and split dataset
    _, _, test_dataset = load_and_split_dataset(
        args.data_path, 
        args.layer_name,
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15, 
        seed=args.seed
    )
    
    # Create test data loader
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=custom_collate_fn
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the best model
    try:
        checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
        input_dim = checkpoint.get('input_dim')
        hidden_dim = checkpoint.get('hidden_dim', args.hidden_dim)
        
        # If input_dim is not stored in the checkpoint, get it from the dataset
        if input_dim is None:
            sample_item = test_dataset[0]
            input_dim = sample_item["embedding"].shape[0]
            
        model = TokenLengthPredictor(input_dim=input_dim, hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {os.path.join(args.output_dir, 'best_model.pt')}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Evaluation
    all_preds = []
    all_labels = []
    evaluation_results = []
    row_counter = 0
    
    # Also calculate L1 loss explicitly
    criterion = nn.L1Loss()
    l1_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            embeddings = batch["embeddings"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            with autocast(enabled=args.use_amp):
                outputs = model(embeddings)
            
            # Loss and predictions
            batch_loss = criterion(outputs.float(), labels.float()).item()
            l1_loss += batch_loss
            
            batch_preds = outputs.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            # Record detailed results for each example
            for i in range(len(batch_preds)):
                evaluation_results.append({
                    'row': row_counter,
                    'actual_length': float(batch_labels[i]),
                    'predicted_length': float(batch_preds[i])
                })
                row_counter += 1
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels)
    avg_l1_loss = l1_loss / len(test_dataloader)
    metrics['l1_loss'] = avg_l1_loss
    
    logger.info("Test Metrics:")
    logger.info(f"  L1 Loss: {avg_l1_loss:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    
    # Log test metrics to wandb
    if args.use_wandb:
        test_metrics = {
            "test/l1_loss": avg_l1_loss,
            "test/mae": metrics['mae'],
            "test/mse": metrics['mse'],
            "test/rmse": metrics['rmse'],
            "test/r2": metrics['r2']
        }
        wandb_logger.log_metrics(test_metrics)
        
        wandb_logger.finish()
    
    # Clean up memory
    logger.info("Evaluation finished, cleaning up memory")
    del model, test_dataset, test_dataloader
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a token length predictor with pre-extracted embeddings")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the pre-extracted embeddings dataset (.pt file)")
    parser.add_argument("--layer_name", type=str, default="layer_8",
                        help="Name of the layer to use for embeddings (e.g., layer_8)")
    
    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Dimension of hidden layer in the predictor model")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save model and results")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
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
    
    # Wandb logging arguments
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="embeddings-length-predictor",
                        help="Weights & Biases project name")
    parser.add_argument("--log_model", action="store_true",
                        help="Whether to log model checkpoints to W&B")
    
    # DataLoader optimization arguments
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for data loading")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Number of batches loaded in advance by each worker")
    
    # Mode arguments
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run evaluation on test set")
    
    # Precision arguments
    parser.add_argument("--use_amp", action="store_true",
                        help="Whether to use automatic mixed precision for training and inference")
    
    args = parser.parse_args()
    
    # Clean CUDA memory at start
    free_gpu_memory()
    
    if args.do_train:
        train_model(args)
    
    if args.do_eval:
        evaluate(args)
    
    logger.info("Script execution completed")
