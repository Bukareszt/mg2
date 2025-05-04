import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import load_from_disk
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM
from models.BasicBert import BasicBertForRegression
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from logger import Logger
import time  # Add import for timing
import csv
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def get_model(args):
    """Initialize model based on arguments."""
    return BasicBertForRegression(model_name=args.model_name)

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
    
    # Extract dataset info from data directory path
    dataset_info = extract_dataset_info(args.data_dir)
    
    # Initialize wandb logger with model name that includes dataset info
    config = vars(args)
    # Add dataset info to config for better tracking in wandb
    config['dataset_info'] = dataset_info
    config['loss_type'] = "L1Loss"  # Always use L1Loss
    
    wandb_logger = Logger(
        config=config,
        model_name=f"bert-length-predictor-{args.model_name.split('/')[-1]}-{dataset_info}",
        project_name=args.wandb_project,
        enable_logging=args.use_wandb,
        log_model=args.log_model
    )
    
    # Load datasets
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
            rounded_outputs = torch.round(outputs)
            loss = criterion(rounded_outputs, torch.round(labels))
            
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
                rounded_outputs = torch.round(outputs)
                loss = criterion(rounded_outputs, torch.round(labels))
                
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

def generate_step_by_step_batch(model, tokenizer, prompts: list[str], steps: int) -> list[str]:
    """
    Generates tokens step-by-step for a batch of prompts.
    Returns list of final texts (prompt + generated tokens).
    """
    device = next(model.parameters()).device
    generated_texts = prompts[:]
    
    for _ in range(steps):
        inputs = tokenizer(generated_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_ids = torch.argmax(next_token_logits, dim=-1)
        new_tokens = tokenizer.batch_decode(next_token_ids, skip_special_tokens=True)

        # Append new tokens to current texts
        generated_texts = [txt + token for txt, token in zip(generated_texts, new_tokens)]

    return generated_texts


def evaluate(args):
    """Evaluation function."""
    set_seed(args.seed)
    
    # Extract dataset info from data directory path
    dataset_info = extract_dataset_info(args.data_dir)
    
    # Initialize wandb logger for test evaluation if requested
    if args.use_wandb:
        config = vars(args)
        config['mode'] = 'evaluation'
        # Add dataset info to config for better tracking in wandb
        config['dataset_info'] = dataset_info
        config['loss_type'] = "L1Loss"  # Always use L1Loss
        
        wandb_logger = Logger(
            config=config,
            model_name=f"bert-length-predictor-eval-{args.model_name.split('/')[-1]}-{dataset_info}",
            project_name=args.wandb_project,
            enable_logging=args.use_wandb,
            log_model=False
        )
    
    # Load test dataset
    logger.info(f"Loading test dataset from {args.data_dir}_test")
    test_dataset = load_from_disk(f"{args.data_dir}_test")
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,  # Use the same batch size as training
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
    
    # Comment out Vicuna model loading
    # vicuna_model_name = "lmsys/vicuna-13b-v1.3"
    # bert_tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    # Load tokenizer and model
    # vicuna_tokenizer = AutoTokenizer.from_pretrained(vicuna_model_name, use_fast=False)
    # vicuna_model = AutoModelForCausalLM.from_pretrained(
    #        vicuna_model_name,
    #        device_map="auto",  # Automatically distributes across available GPUs
    #        load_in_4bit=True,  # If using bitsandbytes (low memory footprint)
    #        torch_dtype=torch.float16,  # Optional depending on GPU
    # )
    # vicuna_model.eval()
    
    # Evaluation
    all_preds = []
    all_labels = []
    latencies = []  # Track latency for each batch
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

            # Comment out step-by-step token generation
            # prompts = batch['prompt']  # List[str], one prompt per sample
            # Step-by-step token generation for full batch
            # extended_prompts = generate_step_by_step_batch(
            #     model=vicuna_model,
            #     tokenizer=vicuna_tokenizer,
            #     prompts=prompts,
            #     steps=args.max_gen_tokens
            # )

            # Tokenize all new prompts at once with BERT tokenizer
            # bert_tokenized = bert_tokenizer(
            #     extended_prompts,
            #     return_tensors="pt",
            #     padding="max_length",
            #     truncation=True,
            #     max_length=512
            # ).to(device)

            start_time = time.time()
            # Use the input_ids and attention_mask directly from the batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            end_time = time.time()
            batch_latency = end_time - start_time

            # Round outputs for integer predictions
            rounded_outputs = torch.round(outputs)
            
            # Loss and predictions
            batch_loss = criterion(rounded_outputs, torch.round(labels)).item()
            l1_loss += batch_loss

            batch_preds = rounded_outputs.view(-1).cpu().numpy()
            batch_labels = labels.view(-1).cpu().numpy()

            for i in range(len(batch_preds)):
                evaluation_results.append({
                    'row': row_counter,
                    'actual_length': float(batch_labels[i]),
                    'predicted_length': float(batch_preds[i]),
                    'latency': float(batch_latency / len(batch_preds))
                })
                row_counter += 1

            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            latencies.append(batch_latency)
    
    metrics = compute_metrics(all_preds, all_labels)
    avg_l1_loss = l1_loss / len(test_dataloader)
    metrics['l1_loss'] = avg_l1_loss
    
    # Calculate average latency
    avg_latency = sum(latencies) / len(latencies)
    metrics['avg_latency'] = avg_latency
    
    logger.info("Test Metrics:")
    logger.info(f"  L1 Loss: {avg_l1_loss:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"  Avg Latency: {avg_latency:.4f} seconds")
    
    # Log test metrics to wandb
    if args.use_wandb:
        test_metrics = {
            "test/l1_loss": avg_l1_loss,
            "test/mae": metrics['mae'],
            "test/mse": metrics['mse'],
            "test/rmse": metrics['rmse'],
            "test/r2": metrics['r2'],
            "performance/avg_latency": avg_latency
        }
        wandb_logger.log_metrics(test_metrics)
        
        # Save evaluation results to file and log as artifact
        output_path = os.path.join(args.output_dir, "evaluation_results.csv")
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row', 'actual_length', 'predicted_length', 'latency'])
            for result in evaluation_results:
                writer.writerow([
                    result['row'],
                    result['actual_length'],
                    result['predicted_length'],
                    result['latency']
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
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to the processed dataset (without _train, _val, _test suffix)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                       help="BERT model name to use")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Directory to save model and results")
    parser.add_argument("--num_epochs", type=int, default=10,
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

    parser.add_argument('--max_gen_tokens', type=int, default=5)
    
    args = parser.parse_args()
    
    # Update the config with the loss type
    config = vars(args)
    config['loss_type'] = "L1Loss"  # Always use L1Loss
    
    if args.do_train:
        train(args)
    
    if args.do_eval:
        evaluate(args)

if __name__ == "__main__":
    main()
