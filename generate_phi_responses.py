from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse
import os
import random
import numpy as np
import torch
import tqdm
import gc


def extract_user_questions(example):
    """Extract the first user message from a conversation."""
    conversation = example['conversation']
    user_content = ""
    
    # Find the first user message
    for sentence in conversation:
        if sentence['role'] == 'user':
            user_content = sentence['content']
            break
    
    return user_content


def generate_pythia_response(user_question, model, tokenizer, max_length=512, device="cuda"):
    """Generate a response using the Pythia model."""
    prompt = f"[USER]: {user_question}\n[ASSISTANT]:"

    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id  # Suppresses warning if pad_token_id is not set
        )

    # Decode the full output
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant's reply
    if "[ASSISTANT]:" in full_response:
        assistant_response = full_response.split("[ASSISTANT]:", 1)[-1].strip()
    else:
        assistant_response = full_response.strip()

    return assistant_response


def process_dataset_with_pythia(dataset, model, tokenizer, batch_size=32, max_samples=None):
    """Process dataset examples with Pythia model and return a new dataset with responses."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Extract all user questions first
    user_questions = []
    for example in tqdm.tqdm(dataset, desc="Extracting user questions"):
        user_questions.append(extract_user_questions(example))
    
    # Limit samples if specified
    if max_samples is not None:
        user_questions = user_questions[:max_samples]
    
    # Generate responses with pythia model
    pythia_responses = []
    for i in tqdm.tqdm(range(0, len(user_questions), batch_size), desc="Generating pythia responses"):
        batch = user_questions[i:i+batch_size]
        batch_responses = []
        
        for question in batch:
            response = generate_pythia_response(question, model, tokenizer, device=device)
            batch_responses.append(response)
        
        pythia_responses.extend(batch_responses)
        
        # Optional: Clear CUDA cache to avoid OOM
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    
    # Create a new dataset with user questions and pythia responses
    new_data = {
        "user_question": user_questions,
        "pythia_response": pythia_responses
    }
    
    return Dataset.from_dict(new_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, help='Size of the dataset to use (in thousands)', default=10)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./data', help='Directory to save the dataset')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum length of generated responses')
    parser.add_argument('--model_revision', type=str, default='step143000', help='Revision/checkpoint of the model to use')
    parser.add_argument('--cache_dir', type=str, default='./pythia-1b', help='Cache directory for model files')
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load Pythia 1B model and tokenizer
    model_name = "EleutherAI/pythia-1b"
    print(f"Loading {model_name} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        revision=args.model_revision,
        cache_dir=f"{args.cache_dir}/{args.model_revision}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=args.model_revision,
        torch_dtype=torch.float32,
        cache_dir=f"{args.cache_dir}/{args.model_revision}"
    )

    # Load the dataset
    dataset_name = 'lmsys/lmsys-chat-1m'
    selected_data_size = 1000 * args.data_size
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name, split='train')
    dataset = dataset.select(range(selected_data_size))
    
    # Filter to only include vicuna-13b samples
    dataset = dataset.filter(lambda example: example["model"] == "vicuna-13b")
    dataset = dataset.shuffle(seed=args.seed)

    # Split the dataset
    train_test_split = dataset.train_test_split(test_size=0.2, seed=args.seed)
    train_raw = train_test_split['train']
    temp_eval_raw = train_test_split['test']

    # Further split the test set into validation and test
    val_test_split = temp_eval_raw.train_test_split(test_size=0.5, seed=args.seed)
    val_raw = val_test_split['train']
    test_raw = val_test_split['test']

    print(f"Raw dataset split sizes: Train={len(train_raw)}, Validation={len(val_raw)}, Test={len(test_raw)}")

    # Create directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each split with pythia
    print("Processing training data...")
    train_pythia_dataset = process_dataset_with_pythia(train_raw, model, tokenizer, 
                                                batch_size=args.batch_size, 
                                                max_samples=len(train_raw))
    
    print("Processing validation data...")
    val_pythia_dataset = process_dataset_with_pythia(val_raw, model, tokenizer, 
                                              batch_size=args.batch_size,
                                              max_samples=len(val_raw))
    
    print("Processing test data...")
    test_pythia_dataset = process_dataset_with_pythia(test_raw, model, tokenizer, 
                                               batch_size=args.batch_size,
                                               max_samples=len(test_raw))

    # Create paths for saving
    base_path = f"{args.output_dir}/lmsys_pythia-1b_{int(selected_data_size / 1000)}K"

    # Save the datasets to disk
    train_pythia_dataset.save_to_disk(f"{base_path}_train")
    val_pythia_dataset.save_to_disk(f"{base_path}_val")
    test_pythia_dataset.save_to_disk(f"{base_path}_test")

    print(f'Saved train dataset to {base_path}_train')
    print(f'Saved validation dataset to {base_path}_val')
    print(f'Saved test dataset to {base_path}_test') 