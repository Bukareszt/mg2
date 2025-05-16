import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import mean_absolute_error
import logging
from logger import Logger  # Import the Logger class

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PiA-Evaluator")

def estimate_length_with_pia(model, tokenizer, prompt, device, max_new_tokens=5):
    pia_prompt = (
        f"{prompt.strip()}\n\n"
        "Before responding to the above instruction, estimate the length of your response in tokens. "
        "Print the estimated number in the first line. Then go to a new line and write the response."
    )

    inputs = tokenizer(pia_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens + 507, do_sample=False)

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    lines = decoded.strip().split("\n")

    try:
        estimated = int(lines[0].strip().split()[0])
    except Exception:
        return -1, 0

    response = "\n".join(lines[1:])
    response_tokens = tokenizer(response, return_tensors="pt", truncation=True).input_ids.shape[1]

    return estimated, response_tokens

def estimate_length_with_pia_batch(model, tokenizer, prompts, device, max_new_tokens=5, batch_size=8):
    """Process multiple prompts in batches for improved performance"""
    all_estimated = []
    all_actual = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        pia_prompts = []
        for prompt in batch_prompts:
            pia_prompt = (
                f"{prompt.strip()}\n\n"
                "Before responding to the above instruction, estimate the length of your response in tokens. "
                "Print the estimated number in the first line. Then go to a new line and write the response."
            )
            pia_prompts.append(pia_prompt)
        
        # Tokenize the batch
        inputs = tokenizer(pia_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Generate outputs for the entire batch
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens + 507, do_sample=False)
        
        # Process each output in the batch
        for j, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            lines = decoded.strip().split("\n")
            
            try:
                estimated = int(lines[0].strip().split()[0])
            except Exception:
                estimated = -1
                
            response = "\n".join(lines[1:])
            response_tokens = tokenizer(response, return_tensors="pt", truncation=True).input_ids.shape[1]
            print(f"Estimated: {estimated}, Actual: {response_tokens}")
            
            all_estimated.append(estimated)
            all_actual.append(response_tokens)
    
    return all_estimated, all_actual

def evaluate_pia(model_id, prompts, max_samples=1000, max_new_tokens=5, 
                 use_wandb=False, wandb_project="pia-evaluator", wandb_name=None,
                 batch_size=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize wandb logger if enabled
    if use_wandb:
        config = {
            "model_id": model_id,
            "max_samples": max_samples,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size
        }
        model_name = wandb_name or f"pia-{model_id.split('/')[-1]}"
        wandb_logger = Logger(
            config=config,
            model_name=model_name,
            project_name=wandb_project,
            enable_logging=use_wandb,
            log_model=False
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()

    if max_samples:
        prompts = prompts[:max_samples]

    logger.info(f"Processing {len(prompts)} prompts with batch size {batch_size}")
    
    if batch_size > 1:
        predictions, true_lengths = estimate_length_with_pia_batch(
            model, tokenizer, prompts, device, max_new_tokens, batch_size
        )
    else:
        # Fallback to non-batched processing if batch_size is 1
        predictions = []
        true_lengths = []
        for prompt in tqdm(prompts, desc="Predicting with PiA"):
            estimated, actual = estimate_length_with_pia(model, tokenizer, prompt, device, max_new_tokens)
            predictions.append(estimated)
            true_lengths.append(actual)

    # Convert to numpy arrays
    true_lengths = np.array(true_lengths)
    predictions = np.array(predictions)
    
    # Count failures
    failed = np.sum(predictions == -1)

    # Metrics
    error = np.abs(true_lengths - predictions)
    mae = error.mean()
    
    # Calculate normalized MAE (similar to bines.py)
    nonzero_mask = true_lengths != 0
    norm_mae = np.mean(np.abs(predictions[nonzero_mask] - true_lengths[nonzero_mask]) / true_lengths[nonzero_mask])
    
    acc_50 = (error < 50).mean()
    acc_100 = (error < 100).mean()

    logger.info(f"Total samples: {len(true_lengths)}")
    logger.info(f"Failures: {failed}")
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"Normalized MAE: {norm_mae:.4f}")
    logger.info(f"Acc@50: {acc_50*100:.2f}%")
    logger.info(f"Acc@100: {acc_100*100:.2f}%")

    # Log metrics to wandb if enabled
    if use_wandb:
        metrics = {
            "mae": mae,
            "normalized_mae": norm_mae,
            "acc_50": acc_50*100,
            "acc_100": acc_100*100,
            "failures": failed,
            "total_samples": len(true_lengths)
        }
        wandb_logger.log_metrics(metrics)
        wandb_logger.finish()

    return mae, norm_mae, acc_50, acc_100, failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    
    # Add wandb arguments similar to bines.py
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="pia-evaluator", help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Custom run name for the wandb experiment")
    
    # Add batch size parameter
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing prompts")

    args = parser.parse_args()

    logger.info(f"Loading prompts from metadata: {args.metadata_path}")
    metadata = torch.load(args.metadata_path)
    prompts = metadata["queries"]

    evaluate_pia(
        model_id=args.model_id,
        prompts=prompts,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        batch_size=args.batch_size
    )
