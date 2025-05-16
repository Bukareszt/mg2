import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import load_from_disk
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from logger import Logger
import csv
import random
from transformers import BertModel

FLAG_BERT_TUNING = False
FLAG_VICUNA_DATA_ONLY = False

class BasicBertForRegression(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=128):
        super(BasicBertForRegression, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        

        self.cls = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        x = self.relu(self.cls(cls_output))

        x = self.relu(self.fc1(x))
        prediction = self.fc2(x).squeeze(-1)
        return prediction


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add a new dataset class for metadata format
class MetadataDataset(Dataset):
    def __init__(self, prompts, lengths, tokenizer, max_length=512):
        self.prompts = prompts
        self.lengths = lengths
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        length = self.lengths[idx]
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(length, dtype=torch.float)
        }

# Add function to load and process metadata
def process_metadata(metadata_path, tokenizer, max_samples=None, train_ratio=0.7, val_ratio=0.15):
    logger.info(f"Loading metadata from {metadata_path}")
    metadata = torch.load(metadata_path)
    
    prompts = metadata["queries"]
    
    # If metadata contains precomputed lengths, use those - check multiple possible field names
    if "lengths" in metadata:
        lengths = metadata["lengths"]
    elif "generated_lengths" in metadata:
        lengths = metadata["generated_lengths"]
    elif "responses" in metadata:
        # Compute lengths from responses using the same method as in pia.py
        responses = metadata["responses"]
        # Use the same tokenization approach as in pia.py
        lengths = []
        for response in responses:
            # Calculate token length the same way as in pia.py
            response_tokens = tokenizer(response, return_tensors="pt", truncation=True).input_ids.shape[1]
            lengths.append(response_tokens)
    else:
        raise ValueError("Metadata must contain either 'lengths', 'generated_lengths' or 'responses'")
    
    # Limit sample size if specified
    if max_samples and max_samples < len(prompts):
        indices = random.sample(range(len(prompts)), max_samples)
        prompts = [prompts[i] for i in indices]
        lengths = [lengths[i] for i in indices]
    
    # Create dataset splits
    data_size = len(prompts)
    indices = list(range(data_size))
    random.shuffle(indices)
    
    train_end = int(train_ratio * data_size)
    val_end = train_end + int(val_ratio * data_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_prompts = [prompts[i] for i in train_indices]
    train_lengths = [lengths[i] for i in train_indices]
    
    val_prompts = [prompts[i] for i in val_indices]
    val_lengths = [lengths[i] for i in val_indices]
    
    test_prompts = [prompts[i] for i in test_indices]
    test_lengths = [lengths[i] for i in test_indices]
    
    return (
        (train_prompts, train_lengths),
        (val_prompts, val_lengths),
        (test_prompts, test_lengths)
    )

def set_seed(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def get_model(args):
    """Initialize model based on arguments."""
    return BasicBertForRegression(model_name=args.model_name)

def compute_metrics(preds, labels):
    """Compute regression metrics."""
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)
    
    # Add PiA-style metrics
    error = np.abs(np.array(preds) - np.array(labels))
    acc_50 = (error < 50).mean()
    acc_100 = (error < 100).mean()
    
    # Calculate normalized MAE
    nonzero_mask = np.array(labels) != 0
    norm_mae = np.mean(np.abs(np.array(preds)[nonzero_mask] - np.array(labels)[nonzero_mask]) / np.array(labels)[nonzero_mask])
    
    return {
        "mae": mae,
        "norm_mae": norm_mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
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
    
    # Let PyTorch handle the padding for input_ids and attention_mask
    if 'input_ids' in batch_by_keys:
        batch_by_keys['input_ids'] = torch.nn.utils.rnn.pad_sequence(
            batch_by_keys['input_ids'], batch_first=True, padding_value=0)
    
    if 'attention_mask' in batch_by_keys:
        batch_by_keys['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
            batch_by_keys['attention_mask'], batch_first=True, padding_value=0)
    
    return batch_by_keys

def train(args):
    """Training function."""
    set_seed(args.seed)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize wandb logger
    config = vars(args)
    config['loss_type'] = "L1Loss"  # Always use L1Loss
    
    wandb_logger = Logger(
        config=config,
        model_name=f"bert-length-predictor",
        project_name=args.wandb_project,
        enable_logging=args.use_wandb,
        log_model=args.log_model
    )
    
    # Load and process data
    if args.metadata_path:
        # Process metadata
        (train_prompts, train_lengths), (val_prompts, val_lengths), _ = process_metadata(
            args.metadata_path, tokenizer, args.max_samples, 
            train_ratio=0.7, val_ratio=0.15
        )
        
        # Create datasets
        train_dataset = MetadataDataset(train_prompts, train_lengths, tokenizer)
        val_dataset = MetadataDataset(val_prompts, val_lengths, tokenizer)
        
        logger.info(f"Created train dataset with {len(train_dataset)} samples")
        logger.info(f"Created validation dataset with {len(val_dataset)} samples")
    else:
        # Load datasets from disk (original behavior)
        logger.info(f"Loading datasets from {args.data_dir}")
        train_dataset = load_from_disk(f"{args.data_dir}_train")
        val_dataset = load_from_disk(f"{args.data_dir}_val")
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=custom_collate_fn
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = get_model(args)
    model.to(device)
    
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Use L1 loss only
    criterion = nn.L1Loss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Early stopping variables
    early_stop_counter = 0
    early_stop_patience = args.early_stopping_patience
    
    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"training_loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].float().to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                all_preds.extend(outputs.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        # Compute metrics
        metrics = compute_metrics(all_preds, all_labels)
        metrics['l1_loss'] = avg_val_loss  # Add L1 loss to metrics
        
        # Log metrics to console
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}:")
        logger.info(f"  Train Loss (L1): {avg_train_loss:.4f}")
        logger.info(f"  Val Loss (L1): {avg_val_loss:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        
        # Log metrics to wandb
        wandb_metrics = {
            "train/l1_loss": avg_train_loss,
            "val/l1_loss": avg_val_loss,
            "val/mae": metrics['mae'],
            "val/rmse": metrics['rmse'],
            "val/r2": metrics['r2'],
            "lr": optimizer.param_groups[0]['lr']
        }
        wandb_logger.log_metrics(wandb_metrics, step=epoch)
        
        lr_scheduler.step(avg_val_loss)
        
        # Handle early stopping and model saving
        if avg_val_loss < best_val_loss:
            improvement = best_val_loss - avg_val_loss
            if improvement >= args.min_loss_improvement:
                logger.info(f"Validation loss decreased from {best_val_loss:.4f} to {avg_val_loss:.4f} (improvement: {improvement:.4f})")
                best_val_loss = avg_val_loss
                early_stop_counter = 0  # Reset counter when validation loss improves significantly
                
                # Save the best model
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                
                model_path = os.path.join(args.output_dir, "best_model.pt")
                logger.info(f"Saving best model to {model_path}")
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'metrics': metrics
                }, model_path)
                
                # Log model checkpoint to wandb
                wandb_logger.log_model_checkpoint(
                    model=model, 
                    path=model_path, 
                    name=f"best_model_epoch_{epoch+1}"
                )
            else:
                logger.info(f"Validation loss decreased from {best_val_loss:.4f} to {avg_val_loss:.4f}, but improvement ({improvement:.4f}) below threshold ({args.min_loss_improvement:.4f})")
                early_stop_counter += 1
                logger.info(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")
                
                # Add check for early stopping here
                if early_stop_counter >= early_stop_patience:
                    logger.info(f"Early stopping triggered after {early_stop_counter} epochs without significant improvement")
                    break
        else:
            # Increment early stopping counter when validation loss doesn't improve
            early_stop_counter += 1
            logger.info(f"No improvement in validation loss. Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            
            # Check if we should stop training
            if early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {early_stop_counter} epochs without improvement")
                break
    
    # Finish wandb logging
    wandb_logger.finish()
    
    return model, best_val_loss, metrics

def evaluate(args):
    """Evaluation function."""
    set_seed(args.seed)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize wandb logger for test evaluation if requested
    if args.use_wandb:
        config = vars(args)
        config['mode'] = 'evaluation'
        config['loss_type'] = "L1Loss"  # Always use L1Loss
        
        wandb_logger = Logger(
            config=config,
            model_name=f"bert-length-predictor-eval-{args.model_name.split('/')[-1]}",
            project_name=args.wandb_project,
            enable_logging=args.use_wandb,
            log_model=False
        )
    
    # Load test dataset
    if args.metadata_path:
        # Process metadata
        _, _, (test_prompts, test_lengths) = process_metadata(
            args.metadata_path, tokenizer, args.max_samples, 
            train_ratio=0.7, val_ratio=0.15
        )
        
        # Create test dataset
        test_dataset = MetadataDataset(test_prompts, test_lengths, tokenizer)
        logger.info(f"Created test dataset with {len(test_dataset)} samples")
    else:
        # Load datasets from disk (original behavior)
        logger.info(f"Loading test dataset from {args.data_dir}_test")
        test_dataset = load_from_disk(f"{args.data_dir}_test")
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=custom_collate_fn
    )
    
    # Load the best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args)
    
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Evaluation
    all_preds = []
    all_labels = []
    evaluation_results = []  # Store results for output file
    
    # Also calculate L1 loss explicitly
    criterion = nn.L1Loss()
    l1_loss = 0.0
    
    row_counter = 0  # Initialize row counter
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate L1 loss
            batch_loss = criterion(outputs, labels).item()
            l1_loss += batch_loss
            
            # Get predictions and labels as numpy arrays
            batch_preds = outputs.view(-1).cpu().numpy()
            batch_labels = labels.view(-1).cpu().numpy()
            
            # Record results with row number, actual length, predicted length
            for i in range(len(batch_preds)):
                evaluation_results.append({
                    'row': row_counter,
                    'actual_length': float(batch_labels[i]),
                    'predicted_length': float(batch_preds[i])
                })
                row_counter += 1
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
    
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
            "test/r2": metrics['r2'],
            "test/norm_mae": metrics['norm_mae'],
        }
        wandb_logger.log_metrics(test_metrics)
        
        # Save evaluation results to file and log as artifact
        output_path = os.path.join(args.output_dir, "evaluation_results.csv")
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row', 'actual_length', 'predicted_length'])
            for result in evaluation_results:
                writer.writerow([
                    result['row'],
                    result['actual_length'],
                    result['predicted_length']
                ])
        
        # Use the logger to log the artifact instead of direct wandb API
        wandb_logger.log_artifact(
            file_path=output_path, 
            name="evaluation_results", 
            artifact_type="eval_results"
        )
        
        logger.info(f"Saved evaluation results to {output_path} and uploaded to wandb as artifact")
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the processed dataset (without _train, _val, _test suffix)")
    parser.add_argument("--metadata_path", type=str, default=None,
                        help="Path to metadata file used in PiA evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use from metadata")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                       help="BERT model name to use")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Directory to save model and results")
    parser.add_argument("--num_epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=1e-5 ,
                       help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for regularization")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Ratio of warmup steps for learning rate scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                       help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Wandb logging arguments
    parser.add_argument("--use_wandb", action="store_true",
                       help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="output-length-prediction",
                       help="Weights & Biases project name")
    parser.add_argument("--log_model", action="store_true",
                       help="Whether to log model checkpoints to W&B")
    parser.add_argument("--plot_every", type=int, default=1,
                       help="Plot predictions every N epochs")
    
    # Mode arguments
    parser.add_argument("--do_train", action="store_true",
                       help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                       help="Whether to run evaluation on test set")
    
    # Add dataloader optimization arguments
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of worker processes for data loading")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                       help="Number of batches loaded in advance by each worker")
    
    # Add minimal loss improvement threshold argument
    parser.add_argument("--min_loss_improvement", type=float, default=1,
                       help="Minimum validation loss improvement to consider as significant")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.data_dir and not args.metadata_path:
        raise ValueError("Either --data_dir or --metadata_path must be provided")
    
    # Update the config with the loss type
    config = vars(args)
    config['loss_type'] = "L1Loss"  # Always use L1Loss
    
    if args.do_train:
        train(args)
    
    if args.do_eval:
        evaluate(args)

if __name__ == "__main__":
    main()