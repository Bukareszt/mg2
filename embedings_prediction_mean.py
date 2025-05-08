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
import time
import csv
from logger import Logger  # Import the existing Logger class

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
    def __init__(self, input_dim=4096, hidden_dim=512, aggregation=None, num_layers=None):
        super(TokenLengthPredictor, self).__init__()
        
        # For learned weighted sum, create layer weights
        self.aggregation = aggregation
        if aggregation == "learned_weighted_sum" and num_layers is not None:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
            self.softmax = nn.Softmax(dim=0)
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, 1)  # Output a single value for regression
        
    def forward(self, x):
        # x is either a single embedding tensor or a list of tensors (for learned_weighted_sum)
        if self.aggregation == "learned_weighted_sum" and isinstance(x, list):
            # Apply softmax to ensure weights sum to 1
            weights = self.softmax(self.layer_weights)
            
            # Apply weights to each layer embedding and sum
            weighted_sum = torch.zeros_like(x[0])
            for i, embedding in enumerate(x):
                weighted_sum += weights[i] * embedding
            
            x = weighted_sum
        
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

def extract_vicuna_embeddings(model, tokenizer, text, layer_indices=None, device='cuda' if torch.cuda.is_available() else 'cpu', aggregation=None):
    """
    Extract embeddings from specific layers of a Vicuna model for a single text input.
    
    Args:
        model: The Vicuna model
        tokenizer: The Vicuna tokenizer
        text: The input text to process
        layer_indices: List of layer indices to extract embeddings from (e.g., [1,2,3,4])
        device: Device to run inference on
        aggregation: Method to aggregate across specified layers ("concat", "mean", "learned_weighted_sum")
        
    Returns:
        Tensor of embeddings or list of embeddings (for learned_weighted_sum)
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
    
    # Use specified layer indices or default to last layer
    if layer_indices is None:
        # Default to last layer if no indices provided
        layer_indices = [-1]
    
    # Extract embeddings from specified layers
    if aggregation == "learned_weighted_sum":
        # For learned_weighted_sum, return a list of embeddings for each layer
        layer_embeddings = [hidden_states[idx][0, -1, :].unsqueeze(0) for idx in layer_indices]
        return layer_embeddings
    elif aggregation == "concat":
        # Concatenate embeddings from specified layers
        layer_embeddings = [hidden_states[idx][0, -1, :] for idx in layer_indices]
        concat_embedding = torch.cat(layer_embeddings, dim=0)
        return concat_embedding.unsqueeze(0)  # Add batch dimension
    elif aggregation == "mean":
        # Mean pool embeddings from specified layers
        layer_embeddings = [hidden_states[idx][0, -1, :] for idx in layer_indices]
        mean_embedding = torch.stack(layer_embeddings).mean(dim=0)
        return mean_embedding.unsqueeze(0)  # Add batch dimension
    else:
        # Default: use the first specified layer index
        layer_output = hidden_states[layer_indices[0]]
        return layer_output[0, -1, :].unsqueeze(0)  # Shape: [1, hidden_size]

def extract_batched_embeddings(model, tokenizer, prompts, layer_indices=None, device='cuda' if torch.cuda.is_available() else 'cpu', aggregation=None):
    """
    Extract embeddings for a batch of prompts in a single forward pass from Vicuna model.
    
    Args:
        model: The Vicuna model
        tokenizer: The Vicuna tokenizer
        prompts: List of prompt strings
        layer_indices: List of layer indices to extract embeddings from (e.g., [1,2,3,4])
        device: Device to run inference on
        aggregation: Method to aggregate across specified layers ("concat", "mean", "learned_weighted_sum")
    
    Returns:
        Batch of embeddings or list of batched embeddings (for learned_weighted_sum)
    """
    # Format prompts for Vicuna
    formatted_prompts = [format_vicuna_prompt(prompt) for prompt in prompts]
    
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get hidden states from all layers
    hidden_states = outputs.hidden_states
    
    # Use specified layer indices or default to last layer
    if layer_indices is None:
        # Default to last layer if no indices provided
        layer_indices = [-1]
    
    # Extract embeddings from specified layers
    batch_size = inputs.input_ids.shape[0]
    
    if aggregation == "learned_weighted_sum":
        # For learned_weighted_sum, return a list of batched embeddings for each layer
        layer_embeddings = [hidden_states[idx].mean(dim=1) for idx in layer_indices]
        return layer_embeddings
    elif aggregation == "concat":
        # Concatenate embeddings from specified layers
        layer_embeddings = [hidden_states[idx].mean(dim=1) for idx in layer_indices]
        concat_embedding = torch.cat(layer_embeddings, dim=1)
        return concat_embedding
    elif aggregation == "mean":
        # Mean pool embeddings from specified layers
        layer_embeddings = [hidden_states[idx].mean(dim=1) for idx in layer_indices]
        mean_embedding = torch.stack(layer_embeddings).mean(dim=0)
        return mean_embedding
    else:
        # Default: use the first specified layer index
        return hidden_states[layer_indices[0]].mean(dim=1)

def predict_remaining_tokens(model, vicuna_model, tokenizer, text, layer_indices=None, device=None, aggregation=None):
    """
    Predict the number of remaining tokens in an LLM output sequence.
    
    Args:
        model: The TokenLengthPredictor model
        vicuna_model: The Vicuna model
        tokenizer: The Vicuna tokenizer
        text: The input text to process
        layer_indices: List of layer indices to extract embeddings from (e.g., [1,2,3,4])
        device: Device to run on (if None, will use the device of the model)
        aggregation: Method to aggregate across specified layers ("concat", "mean", "learned_weighted_sum")
        
    Returns:
        Predicted number of remaining tokens
    """
    # Set models to evaluation mode
    model.eval()
    vicuna_model.eval()
    
    # Determine device if not provided
    if device is None:
        device = next(model.parameters()).device
    
    # Extract embeddings from Vicuna model
    embeddings = extract_vicuna_embeddings(vicuna_model, tokenizer, text, layer_indices, device, aggregation)
    
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
    Extract dataset information from the data directory path.
    """
    try:
        # Extract the last part of the path
        parts = data_dir.rstrip('/').split('/')
        info = parts[-1]
        
        # Remove any leading "lmsys_" prefix
        if info.startswith('lmsys_'):
            info = info[len('lmsys_'):]
            
        return info
    except:
        return "unknown_dataset"

def train_model(args):
    """Training function."""
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract dataset info from data directory path
    dataset_info = extract_dataset_info(args.data_dir)
    
    # Initialize wandb logger with model name that includes dataset info
    config = vars(args)
    # Add dataset info to config for better tracking
    config['dataset_info'] = dataset_info
    config['loss_type'] = "L1Loss"  # We're using L1Loss
    
    wandb_logger = Logger(
        config=config,
        model_name=f"embed-length-predictor-{args.vicuna_model_name.split('/')[-1]}-{dataset_info}",
        project_name=args.wandb_project,
        enable_logging=args.use_wandb
    )
    
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
        # Convert layer_indices from string to list of integers if provided
        layer_indices = None
        if args.layer_indices:
            layer_indices = [int(idx) for idx in args.layer_indices.strip('[]').split(',')]
            logger.info(f"Using layer indices: {layer_indices}")
            
        sample_embedding = extract_batched_embeddings(
            vicuna_model, tokenizer, sample_prompts, layer_indices, device, args.aggregation
        )
        
        # Calculate input dimension based on aggregation method
        if args.aggregation == "learned_weighted_sum":
            # For learned_weighted_sum, get dimension from first layer embedding
            input_dim = sample_embedding[0].shape[1]
            num_layers = len(sample_embedding)
        elif args.aggregation == "concat" and layer_indices:
            # For concat, dimension is sum of dimensions from all layers
            input_dim = sample_embedding.shape[1]
            num_layers = len(layer_indices) if layer_indices else 1
        else:
            # For mean or single layer, dimension is the embedding dimension
            input_dim = sample_embedding.shape[1]
            num_layers = len(layer_indices) if layer_indices else 1
            
        logger.info(f"Detected embedding dimension: {input_dim}")
        logger.info(f"Number of layers used: {num_layers}")
    
    # Initialize model with appropriate parameters
    model = TokenLengthPredictor(
        input_dim=input_dim, 
        hidden_dim=args.hidden_dim,
        aggregation=args.aggregation if args.aggregation == "learned_weighted_sum" else None,
        num_layers=num_layers if args.aggregation == "learned_weighted_sum" else None
    )
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
            prompts = batch['prompt']
            labels = batch['labels'].float().to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Extract embeddings from Vicuna model using raw prompts
            with autocast(enabled=args.use_amp):
                with torch.no_grad():
                    embeddings = extract_batched_embeddings(
                        vicuna_model, tokenizer, prompts, layer_indices, device, args.aggregation
                    )
                
                # Ensure embeddings are the same type as model parameters
                if args.aggregation == "learned_weighted_sum":
                    # For learned_weighted_sum, convert each embedding in the list
                    embeddings = [emb.to(next(model.parameters()).dtype) for emb in embeddings]
                else:
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
                        vicuna_model, tokenizer, prompts, layer_indices, device, args.aggregation
                    )
                    
                    # Ensure embeddings are the same type as model parameters
                    if args.aggregation == "learned_weighted_sum":
                        # For learned_weighted_sum, convert each embedding in the list
                        embeddings = [emb.to(next(model.parameters()).dtype) for emb in embeddings]
                    else:
                        if next(model.parameters()).dtype != embeddings.dtype:
                            embeddings = embeddings.to(next(model.parameters()).dtype)
                    
                    outputs = model(embeddings)
                
                # Calculate loss
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
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
            
        else:
            early_stop_counter += 1
            logger.info(f"Validation loss did not decrease significantly. Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            
            if early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save the final model regardless of performance
    checkpoint = {
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
    }
    torch.save(checkpoint, os.path.join(args.output_dir, "final_model.pt"))
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Finish wandb logging
    if args.use_wandb:
        wandb_logger.finish()
    
    return best_val_loss

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
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=custom_collate_fn
    )
    
    # Extract dataset info for logging
    dataset_info = extract_dataset_info(args.data_dir)
    
    # Initialize wandb logger for evaluation
    if args.use_wandb:
        config = vars(args)
        config['dataset_info'] = dataset_info
        config['phase'] = 'evaluation'
        
        wandb_logger = Logger(
            config=config,
            model_name=f"eval-embed-predictor-{args.vicuna_model_name.split('/')[-1]}-{dataset_info}",
            project_name=args.wandb_project,
            enable_logging=args.use_wandb,
            log_model=False
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
    
    # Convert layer_indices from string to list of integers if provided
    layer_indices = None
    if args.layer_indices:
        layer_indices = [int(idx) for idx in args.layer_indices.strip('[]').split(',')]
        logger.info(f"Using layer indices: {layer_indices}")
    
    with torch.no_grad():
        sample_embedding = extract_batched_embeddings(
            vicuna_model, tokenizer, sample_prompts, layer_indices, device, args.aggregation
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
    evaluation_results = []
    row_counter = 0
    
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
                    vicuna_model, tokenizer, prompts, layer_indices, device, args.aggregation
                )
                
                # Ensure embeddings are the same type as model parameters
                if next(model.parameters()).dtype != embeddings.dtype:
                    embeddings = embeddings.to(next(model.parameters()).dtype)
                
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
    
    return metrics

if __name__ == '__main__':
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
    
    # Deprecated: use layer_indices instead
    parser.add_argument("--layer_idx", type=int, default=-1,
                        help="DEPRECATED: Use --layer_indices instead. Index of Vicuna layer to extract embeddings from.")
    
    # New parameter for multiple layer indices
    parser.add_argument("--layer_indices", type=str, default=None,
                        help="Comma-separated list of layer indices to use, e.g., '[1,2,3,4]'")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save model and results")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
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
    parser.add_argument("--min_loss_improvement", type=float, default=0.01,
                        help="Minimum validation loss improvement to consider as significant")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Wandb logging arguments
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="embeddings-length-predictor",
                        help="Weights & Biases project name")
    
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
    
    # Add aggregation argument
    parser.add_argument("--aggregation", type=str, default="mean", 
                        choices=["mean", "concat", "learned_weighted_sum"],
                        help="Method to aggregate embeddings across specified layers")
    
    args = parser.parse_args()
    
    # Check if using BFloat16 on a supported device
    if args.precision == "bfloat16" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        logger.warning("BFloat16 not supported on this device, falling back to Float16")
        args.precision = "float16"
    
    if args.do_train:
        train_model(args)
    
    if args.do_eval:
        evaluate(args)
