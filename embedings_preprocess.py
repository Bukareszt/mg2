from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse
import os
import random
import numpy as np
import torch
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("preprocessing.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def format_prompt(text):
    """Format the prompt for LLaMA 3 instruction model"""
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

def generate_response_and_count_tokens(prompt, model, tokenizer, device, max_new_tokens=512):
    """
    Generate a response using the LLaMA model and count its tokens
    
    Args:
        prompt: User prompt text
        model: LLaMA model
        tokenizer: LLaMA tokenizer
        device: Device to run inference on
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        The number of tokens in the generated response
    """
    # Format the prompt for LLaMA
    formatted_prompt = format_prompt(prompt)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    # Count only the new tokens that were generated (excluding the prompt)
    response_length = outputs.shape[1] - prompt_length
    return response_length

def process_batch(examples, model, tokenizer, device):
    """Process a batch of examples using the LLaMA model"""
    prompts = examples['prompt']
    labels = []
    
    for prompt in prompts:
        # Generate response and count tokens
        response_length = generate_response_and_count_tokens(
            prompt, model, tokenizer, device)
        labels.append(response_length)
    
    examples['labels'] = labels
    return examples

def preprocess_dataset(dataset, model, tokenizer, batch_size=8):
    """Preprocess the dataset for embedding-based prediction with LLaMA inference"""
    # Remove unnecessary columns
    dataset = dataset.remove_columns(['openai_moderation', 'redacted', 'language', 'conversation_id', 'turn', 'model'])
    
    # Determine device - use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Make sure model is on the correct device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Convert conversation to prompts and generate responses
    logger.info("Extracting user prompts...")
    # First extract just the prompts from the conversations
    def extract_user_prompt(example):
        conversation = example['conversation']
        user_content = ''
        
        # Extract user prompt
        for i, sentence in enumerate(conversation):
            if sentence['role'] == 'user':
                if i > 0:
                    user_content += '\n'
                user_content += sentence['content']
            else:
                break
                
        example['prompt'] = user_content
        return example
    
    # Extract prompts only
    dataset = dataset.map(extract_user_prompt, remove_columns=['conversation'])
    
    # Initialize labels column
    dataset = dataset.add_column('labels', [0] * len(dataset))
    
    # Generate responses and compute lengths in batches
    logger.info("Generating LLaMA responses and computing token counts...")
    dataset = dataset.map(
        lambda examples: process_batch(examples, model, tokenizer, device),
        batched=True,
        batch_size=batch_size,
        desc="Generating responses",
    )
    
    logger.info(f'Num samples before filtering: {len(dataset)}')
    dataset = dataset.filter(lambda example: example["labels"] > 1 and example["labels"] <= 512)
    logger.info(f'Num samples after filtering: {len(dataset)}')
    
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, help='Size of the dataset to use (in thousands)', default=1000)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./data', help='Directory to save the processed datasets')
    parser.add_argument('--llama_model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct', 
                        help='Name of the LLaMA model to use for generating responses')
    parser.add_argument('--hf_token', type=str, default='None',
                        help='HuggingFace token for accessing gated models')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing examples with inference')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum new tokens to generate with LLaMA')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize the LLaMA model and tokenizer
    logger.info(f"Loading LLaMA model and tokenizer: {args.llama_model_name}")
    
    # Load tokenizer first
    llama_tokenizer = AutoTokenizer.from_pretrained(
        args.llama_model_name, 
        use_auth_token=args.hf_token
    )
    
    # Then load model with low precision to save memory
    llama_model = AutoModelForCausalLM.from_pretrained(
        args.llama_model_name,
        use_auth_token=args.hf_token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Dataset parameters
    dataset_name = 'lmsys/lmsys-chat-1m'
    selected_data_size = 1000 * args.data_size

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the dataset once
    logger.info(f"Loading {dataset_name} dataset")
    dataset = load_dataset(dataset_name, split='train')
    dataset = dataset.select(range(selected_data_size))
    
    # Filter to only include vicuna-13b samples
    logger.info("Filtering for vicuna-13b samples")
    dataset = dataset.filter(lambda example: example["model"] == "vicuna-13b")
    dataset = dataset.shuffle(seed=args.seed)

    # Split the dataset BEFORE preprocessing to prevent data leakage
    logger.info("Splitting dataset into train/validation/test")
    train_test_split = dataset.train_test_split(test_size=0.2, seed=args.seed)
    train_raw = train_test_split['train']
    temp_eval_raw = train_test_split['test']

    # Further split the test set into validation and test
    val_test_split = temp_eval_raw.train_test_split(test_size=0.5, seed=args.seed)
    val_raw = val_test_split['train']
    test_raw = val_test_split['test']

    logger.info(f"Raw dataset split sizes: Train={len(train_raw)}, Validation={len(val_raw)}, Test={len(test_raw)}")

    # Create dataset path with model name
    model_short_name = args.llama_model_name.split('/')[-1].lower()
    dataset_path = f'{args.output_dir}/lmsys_{model_short_name}_gen_{int(selected_data_size / 1000)}K'

    # Process each split with LLaMA model inference
    logger.info("Processing training data with LLaMA inference...")
    train_dataset = preprocess_dataset(train_raw, llama_model, llama_tokenizer, args.batch_size)
    train_dataset.set_format("torch")

    logger.info("Processing validation data with LLaMA inference...")
    val_dataset = preprocess_dataset(val_raw, llama_model, llama_tokenizer, args.batch_size)
    val_dataset.set_format("torch")

    logger.info("Processing test data with LLaMA inference...")
    test_dataset = preprocess_dataset(test_raw, llama_model, llama_tokenizer, args.batch_size)
    test_dataset.set_format("torch")

    logger.info(f"Processed dataset split sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    base_path = dataset_path.rstrip('/')

    # Save the datasets to disk
    train_dataset.save_to_disk(f"{base_path}_train")
    val_dataset.save_to_disk(f"{base_path}_val")
    test_dataset.save_to_disk(f"{base_path}_test")

    logger.info(f'Saved train dataset to {base_path}_train')
    logger.info(f'Saved validation dataset to {base_path}_val')
    logger.info(f'Saved test dataset to {base_path}_test')
