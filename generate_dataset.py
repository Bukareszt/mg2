import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
from collections import defaultdict
import logging
import os
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TRAIL-Dataset")

class EmbeddingExtractor:
    def __init__(
        self, 
        model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
        max_length=512, 
        num_bins=10, 
        device=None,
        layer_range=(8, 16)
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.num_bins = num_bins
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_range = layer_range
        
        logger.info(f"Initializing EmbeddingExtractor with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Layer range: {self.layer_range}, Max length: {self.max_length}, Num bins: {self.num_bins}")
        
        # Load model and tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        
        logger.info(f"Loading model to {self.device}...")
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.model.eval()
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")

        # Bin edges for remaining length
        self.bin_edges = torch.linspace(0, self.max_length, self.num_bins + 1)
        logger.info(f"Bin edges created: {self.bin_edges}")

    def bin_remaining_length(self, length):
        return torch.bucketize(torch.tensor([length]), self.bin_edges)[0] - 1

    def extract_embeddings_and_labels(self, prompt_text):
        logger.debug(f"Processing prompt (first 50 chars): {prompt_text[:50]}...")
        inputs = self.tokenizer(
            prompt_text, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        output_length = self.max_length - input_ids.shape[1]

        if output_length <= 0:
            logger.warning(f"Prompt too long ({input_ids.shape[1]} tokens), skipping.")
            raise ValueError(f"Prompt too long ({input_ids.shape[1]} tokens), skipping.")

        logger.debug(f"Generating with max_new_tokens={output_length}")
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=output_length,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

            full_input = generated.sequences  # [1, prompt + gen]
            full_attention_mask = (full_input != self.tokenizer.pad_token_id).long()

            # Run full forward pass to get hidden states
            logger.debug("Running forward pass to extract hidden states")
            outputs = self.model(
                input_ids=full_input,
                attention_mask=full_attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states  # tuple of (num_layers+1) x [1, seq_len, hidden_size]

        layer_embeddings = defaultdict(list)
        labels = []

        gen_len = full_input.shape[1] - input_ids.shape[1]
        logger.debug(f"Generated sequence length: {gen_len}")

        for i in range(gen_len):
            remaining = gen_len - i - 1
            bin_id = self.bin_remaining_length(remaining)
            labels.append(bin_id)

            token_pos = input_ids.shape[1] + i
            for layer_idx in range(*self.layer_range):
                if token_pos < hidden_states[layer_idx].shape[1]:
                    embedding = hidden_states[layer_idx][0][token_pos]
                    layer_embeddings[f"layer_{layer_idx}"].append(embedding.detach().cpu())
                else:
                    logger.warning(f"Skipped token position {token_pos} for layer {layer_idx} (out of bounds)")

        logger.debug(f"Extracted {len(labels)} tokens with embeddings")
        return layer_embeddings, labels


    def process_dataset(self, dataset_name, split="train[:1000]", output_file="trail_dataset_all_layers.pt", batch_size=4):
        logger.info(f"Loading dataset: {dataset_name}, split: {split}")
        ds = load_dataset(dataset_name, split=split)
        logger.info(f"Dataset loaded with {len(ds)} examples")
        
        all_layer_embeddings = defaultdict(list)
        all_labels = []

        batch = []
        total_processed = 0
        skipped = 0
        
        logger.info(f"Processing dataset with batch size {batch_size}")
        progress_bar = tqdm(total=len(ds), desc="Processing examples")
        
        for idx, example in enumerate(ds):
            prompt = example["text"]
            batch.append(prompt)

            if len(batch) == batch_size or idx == len(ds) - 1:
                logger.debug(f"Processing batch {total_processed//batch_size + 1}, examples {total_processed}-{total_processed+len(batch)-1}")
                for prompt_text in batch:
                    try:
                        layer_embs, labels = self.extract_embeddings_and_labels(prompt_text)
                        for layer_key, embs in layer_embs.items():
                            all_layer_embeddings[layer_key].extend(embs)
                        all_labels.extend(labels)
                        total_processed += 1
                    except Exception as e:
                        logger.warning(f"Skipped example: {e}")
                        skipped += 1
                    finally:
                        progress_bar.update(1)
                batch = []
                
                # Log progress periodically
                if total_processed % 100 == 0 and total_processed > 0:
                    logger.info(f"Progress: {total_processed}/{len(ds)} examples processed ({skipped} skipped)")

        progress_bar.close()
        logger.info(f"Dataset processing complete. Total processed: {total_processed}, skipped: {skipped}")

        # Save the dataset
        logger.info(f"Saving dataset to {output_file}")
        saved_data = {key: torch.stack(val) for key, val in all_layer_embeddings.items()}
        saved_data["labels"] = torch.tensor(all_labels)
        torch.save(saved_data, output_file)

        logger.info(f"\n✅ Dataset saved to {output_file}")
        logger.info(f"🔢 Total examples: {len(all_labels)}")
        logger.info(f"🧠 Layers saved: {list(all_layer_embeddings.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset with all LLM layer embeddings for TRAIL")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins for length classification")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help="Dataset name")
    parser.add_argument("--split", type=str, default="train[:1000]", help="Dataset split")
    parser.add_argument("--output", type=str, default="trail_dataset_all_layers.pt", help="Output file path")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log_file", type=str, default="", help="Log file path (if empty, logs to console only)")

    args = parser.parse_args()
    
    # Configure logging based on arguments
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    
    # Add file handler if log file is specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
    logger.info(f"Starting TRAIL dataset generation with args: {args}")

    extractor = EmbeddingExtractor(
        model_name=args.model,
        max_length=args.max_length,
        num_bins=args.num_bins
    )

    extractor.process_dataset(
        dataset_name=args.dataset,
        split=args.split,
        output_file=args.output
    )
    
    logger.info("Dataset generation completed successfully")


if __name__ == "__main__":
    main()