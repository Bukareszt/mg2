import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler  # Add imports for mixed precision
import numpy as np
import os
import logging
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TokenLengthPredictor(nn.Module):
    """
    A simple MLP regressor that predicts the number of remaining tokens in an LLM output sequence.
    Input: Embedding vector of shape [4096] from a transformer layer
    Output: A single scalar value representing the predicted token length
    """
    def __init__(self, input_dim=4096, hidden_dim=512):
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

def custom_collate_fn(batch):
    """
    Custom collate function to handle varying tensor sizes in the dataset.
    """
    # Sort batch by label length (descending order)
    batch_by_keys = {
        key: [d[key] for d in batch] for key in batch[0].keys()
    }
    
    # For labels, we need to ensure they're all the same shape
    if 'labels' in batch_by_keys:
        # Convert to list if it's a tensor
        labels = [label.item() if isinstance(label, torch.Tensor) and label.numel() == 1 
                  else label for label in batch_by_keys['labels']]
        batch_by_keys['labels'] = torch.tensor(labels, dtype=torch.float)
    
    # For raw prompts, we just need a list of strings
    if 'prompt' in batch_by_keys:
        # No processing needed for string prompts
        pass
        
    return batch_by_keys

def format_vicuna_prompt(text):
    """Format a prompt for Vicuna models"""
    # Vicuna uses a different format than LLaMA-3
    # For most Vicuna models, the format is:
    # "USER: {user_message}\nASSISTANT:"
    return f"USER: {text}\nASSISTANT:"

def extract_vicuna_embeddings(model, tokenizer, text, layer_idx=-1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Extract embeddings from a specific layer of a Vicuna model for a single text input.
    
    Args:
        model: The Vicuna model
        tokenizer: The Vicuna tokenizer
        text: The input text to process
        layer_idx: The index of the layer to extract embeddings from (-1 for last layer)
        device: Device to run inference on
        
    Returns:
        Tensor of embeddings of shape [1, hidden_size]
    """
    # Format the prompt for Vicuna
    formatted_text = format_vicuna_prompt(text)
    
    # Move model to the specified device if not already there
    model = model.to(device)
    
    # Tokenize input text
    inputs = tokenizer(formatted_text, return_tensors="pt").to(device)
    
    # Run forward pass with output_hidden_states=True to get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get hidden states from all layers
    hidden_states = outputs.hidden_states
    
    # Extract embeddings from the specified layer
    # If layer_idx is -1, we get the last layer
    layer_output = hidden_states[layer_idx]
    
    # We typically use the embedding of the last token for classification
    # Shape: [batch_size, sequence_length, hidden_size]
    last_token_embedding = layer_output[0, -1, :]
    
    return last_token_embedding.unsqueeze(0)  # Shape: [1, hidden_size]

def extract_batched_embeddings(model, tokenizer, prompts, layer_idx=-1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Extract embeddings for a batch of prompts in a single forward pass from Vicuna model.
    
    Args:
        model: The Vicuna model
        tokenizer: The Vicuna tokenizer
        prompts: List of prompt strings
        layer_idx: The index of the layer to extract embeddings from (-1 for last layer)
        device: Device to run inference on
    
    Returns:
        Batch of embeddings [batch_size, hidden_size]
    """
    # Format prompts for Vicuna
    formatted_prompts = [format_vicuna_prompt(prompt) for prompt in prompts]
    
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    layer_output = outputs.hidden_states[layer_idx]  # [batch_size, seq_len, hidden_dim]
    mean_embeddings = layer_output.mean(dim=1)       # Mean over tokens
    return mean_embeddings

def predict_remaining_tokens(model, vicuna_model, tokenizer, text, layer_idx=-1):
    """
    Predict the number of remaining tokens in an LLM output sequence.
    
    Args:
        model: The TokenLengthPredictor model
        vicuna_model: The Vicuna model
        tokenizer: The Vicuna tokenizer
        text: The input text to process
        layer_idx: The index of the layer to extract embeddings from
        
    Returns:
        Predicted number of remaining tokens
    """
    # Set models to evaluation mode
    model.eval()
    vicuna_model.eval()
    
    # Extract embeddings from Vicuna model
    device = next(model.parameters()).device
    embeddings = extract_vicuna_embeddings(vicuna_model, tokenizer, text, layer_idx, device)
    
    # Make prediction with the model
    with torch.no_grad():
        prediction = model(embeddings)
    
    return prediction.item()

def load_vicuna_model(model_name="lmsys/vicuna-13b-v1.3", use_auth_token=None, precision="float16"):
    """
    Load a Vicuna model and tokenizer.
    
    Args:
        model_name: The name of the model to load (default: lmsys/vicuna-13b-v1.3)
        use_auth_token: HuggingFace token for accessing gated models
        precision: Model precision - "float16", "bfloat16", or "float32"
        
    Returns:
        model, tokenizer
    """
    logger.info(f"Loading Vicuna model: {model_name} with precision {precision}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token, legacy=False)
    
    # Determine torch dtype based on precision argument
    if precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        logger.info("Using BFloat16 precision")
    elif precision == "float16":
        torch_dtype = torch.float16
        logger.info("Using Float16 precision")
    else:
        torch_dtype = torch.float32
        logger.info("Using Float32 precision")
    
    # Load model with ability to output hidden states
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=use_auth_token,
        output_hidden_states=True,
        torch_dtype=torch_dtype,  # Use specified precision
        device_map="auto",        # Automatically distribute across available GPUs
    )
    
    return model, tokenizer

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

def extract_dataset_info(data_dir):
    """
    Extract dataset name and configuration from data directory path.
    If the name contains 'preview' (like preview5/preview10), extract that info.
    Otherwise, label it as 'normal'.
    """
    # Extract the base name without the path and suffixes
    base_name = os.path.basename(data_dir)
    
    # Handle different path formats
    if '/' in data_dir:
        parts = data_dir.split('/')
        if len(parts) >= 2:
            base_name = parts[-1]  # Get the last part of the path
    
    # Check if "preview" is in the name and extract that info
    if "preview" in base_name:
        # Find the preview pattern (preview followed by numbers)
        import re
        preview_match = re.search(r'preview(\d+)', base_name)
        if preview_match:
            preview_count = preview_match.group(1)
            return f"preview{preview_count}"
    
    # If no preview found, return "normal"
    return "normal"

def train_model(args):
    """Training function."""
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading train dataset from {args.data_dir}_train")
    train_dataset = load_from_disk(f"{args.data_dir}_train")
    
    logger.info(f"Loading validation dataset from {args.data_dir}_val")
    val_dataset = load_from_disk(f"{args.data_dir}_val")
    
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
    
    # Load Vicuna model for embedding extraction
    logger.info("Loading Vicuna model for training")
    vicuna_model, tokenizer = load_vicuna_model(args.vicuna_model_name, args.hf_token, args.precision)
    vicuna_model.eval()  # Set model to evaluation mode for embedding extraction
    
    # Get a sample batch to determine embedding size
    sample_batch = next(iter(train_dataloader))
    sample_prompts = sample_batch['prompt'][:1]  # Just use the first prompt
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Extract sample embedding to determine input dimension
    with torch.no_grad():
        sample_embedding = extract_batched_embeddings(
            vicuna_model, tokenizer, sample_prompts, args.layer_idx, device
        )
        input_dim = sample_embedding.shape[1]
        logger.info(f"Detected embedding dimension: {input_dim}")
    
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
    
    # Extract dataset info for logging
    dataset_info = extract_dataset_info(args.data_dir)
    logger.info(f"Training on dataset: {dataset_info}")
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Prediction and label collections for metric calculation
        train_preds = []
        train_labels = []
        
        for batch in train_pbar:
            prompts = batch['prompt']
            labels = batch['labels'].float().to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Extract embeddings from Vicuna model using raw prompts
            with autocast(enabled=args.use_amp):
                with torch.no_grad():
                    embeddings = extract_batched_embeddings(
                        vicuna_model, tokenizer, prompts, args.layer_idx, device
                    )
                
                # Ensure embeddings are the same type as model parameters
                if next(model.parameters()).dtype != embeddings.dtype:
                    embeddings = embeddings.to(next(model.parameters()).dtype)
                
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
                prompts = batch['prompt']
                labels = batch['labels'].float().to(device)
                
                # Extract embeddings from Vicuna model using raw prompts
                with autocast(enabled=args.use_amp):
                    embeddings = extract_batched_embeddings(
                        vicuna_model, tokenizer, prompts, args.layer_idx, device
                    )
                    
                    # Ensure embeddings are the same type as model parameters
                    if next(model.parameters()).dtype != embeddings.dtype:
                        embeddings = embeddings.to(next(model.parameters()).dtype)
                    
                    outputs = model(embeddings)
                
                # Convert outputs to float32 for loss calculation
                if outputs.dtype != torch.float32:
                    outputs = outputs.float()
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_metrics = compute_metrics(val_preds, val_labels)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}:")
        logger.info(f"  Train Loss (L1): {avg_train_loss:.4f}")
        logger.info(f"  Train MAE: {train_metrics['mae']:.4f}")
        logger.info(f"  Train RMSE: {train_metrics['rmse']:.4f}")
        logger.info(f"  Train R²: {train_metrics['r2']:.4f}")
        logger.info(f"  Val Loss (L1): {avg_val_loss:.4f}")
        logger.info(f"  Val MAE: {val_metrics['mae']:.4f}")
        logger.info(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        logger.info(f"  Val R²: {val_metrics['r2']:.4f}")
        
        # Update learning rate based on validation loss
        lr_scheduler.step(avg_val_loss)
        
        # Check if this is the best model so far
        if avg_val_loss < best_val_loss - args.min_loss_improvement:
            logger.info(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}")
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            
            # Save the model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_metrics': val_metrics,
                'input_dim': input_dim,
                'hidden_dim': args.hidden_dim
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
            logger.info(f"Saved new best model to {os.path.join(args.output_dir, 'best_model.pt')}")
        else:
            early_stop_counter += 1
            logger.info(f"Validation loss did not improve. Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            
            if early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model for final evaluation
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, best_val_loss, checkpoint['val_metrics']

def evaluate(args):
    """Evaluation function."""
    set_seed(args.seed)
    
    # Load test dataset
    logger.info(f"Loading test dataset from {args.data_dir}_test")
    test_dataset = load_from_disk(f"{args.data_dir}_test")
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else False,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=custom_collate_fn
    )
    
    # Load Vicuna model for embedding extraction
    logger.info("Loading Vicuna model for evaluation")
    vicuna_model, tokenizer = load_vicuna_model(args.vicuna_model_name, args.hf_token, args.precision)
    vicuna_model.eval()
    
    # Load the best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get a sample to determine embedding size
    sample_batch = next(iter(test_dataloader))
    sample_prompts = sample_batch['prompt'][:1]  # Just use the first prompt
    
    with torch.no_grad():
        sample_embedding = extract_batched_embeddings(
            vicuna_model, tokenizer, sample_prompts, args.layer_idx, device
        )
        input_dim = sample_embedding.shape[1]
        logger.info(f"Detected embedding dimension: {input_dim}")
    
    model = TokenLengthPredictor(input_dim=input_dim, hidden_dim=args.hidden_dim)
    
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Evaluation
    all_preds = []
    all_labels = []
    
    # Also calculate L1 loss explicitly
    criterion = nn.L1Loss()
    l1_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            prompts = batch['prompt']
            labels = batch['labels'].float().to(device)
            
            # Extract embeddings from Vicuna model using raw prompts
            with autocast(enabled=args.use_amp):
                embeddings = extract_batched_embeddings(
                    vicuna_model, tokenizer, prompts, args.layer_idx, device
                )
                
                outputs = model(embeddings)
            
            # Loss and predictions
            batch_loss = criterion(outputs.float(), labels.float()).item()
            l1_loss += batch_loss
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels)
    avg_l1_loss = l1_loss / len(test_dataloader)
    metrics['l1_loss'] = avg_l1_loss
    
    logger.info("Test Metrics:")
    logger.info(f"  L1 Loss: {avg_l1_loss:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    
    return metrics

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to the processed dataset (without _train, _val, _test suffix)")
    
    # Model arguments
    parser.add_argument("--input_dim", type=int, default=4096,
                        help="Dimension of input embeddings (detected automatically)")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Dimension of hidden layer")
    parser.add_argument("--vicuna_model_name", type=str, default="lmsys/vicuna-13b-v1.3",
                        help="Name of Vicuna model to use for embedding extraction")
    parser.add_argument("--layer_idx", type=int, default=-1,
                        help="Index of Vicuna layer to extract embeddings from (-1 for last layer)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for accessing gated models")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save model and results")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for regularization")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of warmup steps for learning rate scheduler")
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
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Number of batches loaded in advance by each worker")
    
    # Mode arguments
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run evaluation on test set")
    
    # Precision arguments
    parser.add_argument("--precision", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Precision for Vicuna model")
    parser.add_argument("--use_amp", action="store_true",
                        help="Whether to use automatic mixed precision for training and inference")
    
    args = parser.parse_args()
    
    # Check if using BFloat16 on a supported device
    if args.precision == "bfloat16" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        logger.warning("BFloat16 not supported on this device, falling back to Float16")
        args.precision = "float16"
    
    if args.do_train:
        train_model(args)
    
    if args.do_eval:
        evaluate(args)
