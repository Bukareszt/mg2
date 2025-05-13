import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
from collections import defaultdict
import logging
import traceback
import os
from tqdm import tqdm
import gc
from pathlib import Path

def setup_logger(log_file="embedding_extraction.log", level=logging.INFO):
    """Set up logger with file and console handlers."""
    # Create logger
    logger = logging.getLogger("embedding_extractor")
    logger.setLevel(level)
    logger.handlers = []  # Clear any existing handlers
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class EmbeddingExtractor:
    def __init__(
        self, 
        model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
        max_length=512, 
        num_bins=10, 
        device=None,
        layer_range=(8, 9),
        logger=None,
        batch_size=10,
        save_interval=100,
        use_half_precision=False
    ):
        self.logger = logger or logging.getLogger("embedding_extractor")
        self.model_name = model_name
        self.max_length = max_length
        self.num_bins = num_bins
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_range = layer_range
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.use_half_precision = use_half_precision
        
        self.logger.info(f"Initializing EmbeddingExtractor with model: {model_name}")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Batch size: {batch_size}, Save interval: {save_interval}")
        if use_half_precision:
            self.logger.info("Using half precision (FP16) for model")
        
        # Load model and tokenizer
        try:
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            self.logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                output_hidden_states=True
            ).to(self.device)
            self.model.eval()
            self.logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            self.logger.critical(f"Failed to load model or tokenizer: {e}")
            self.logger.debug(traceback.format_exc())
            raise

        # Bin edges for remaining length
        self.bin_edges = torch.linspace(0, self.max_length, self.num_bins + 1)
        self.logger.info(f"Created {self.num_bins} bins for remaining length")

    def bin_remaining_length(self, length):
        return torch.bucketize(torch.tensor([length]), self.bin_edges)[0] - 1

    def extract_embeddings_and_labels(self, prompt_text):
        try:
            self.logger.debug(f"Tokenizing prompt (first 30 chars): {prompt_text[:30]}...")
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            output_length = self.max_length - input_ids.shape[1]

            if output_length <= 0:
                self.logger.warning(f"Prompt too long ({input_ids.shape[1]} tokens), skipping.")
                raise ValueError(f"Prompt too long ({input_ids.shape[1]} tokens), skipping.")

            self.logger.debug(f"Generating tokens with max new tokens: {output_length}")
            
            with torch.no_grad():  # Disable gradient calculation to save memory
                generated = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=output_length,
                    return_dict_in_generate=True,
                    output_hidden_states=True
                )

            tokens_generated = generated.sequences[0][input_ids.shape[1]:]
            self.logger.debug(f"Generated {len(tokens_generated)} tokens")
            hidden_states = generated.hidden_states  # Tuple: [num_layers+1][batch, seq_len, dim]

            layer_embeddings = defaultdict(list)
            labels = []

            for i in range(len(tokens_generated)):
                remaining = len(tokens_generated) - i - 1
                bin_id = self.bin_remaining_length(remaining)
                labels.append(bin_id)

                for layer_idx in range(*self.layer_range):  # Only selected layers
                    # Explicitly detach and move to CPU to free up GPU memory
                    embedding = hidden_states[layer_idx][0][-len(tokens_generated) + i].detach().cpu()
                    layer_embeddings[f"layer_{layer_idx}"].append(embedding)
            
            # Clear references to large objects
            del generated, hidden_states, inputs
            
            # Explicitly run garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.debug(f"Extracted embeddings for {len(tokens_generated)} tokens across layers {self.layer_range}")
            return layer_embeddings, labels
            
        except Exception as e:
            self.logger.error(f"Error during embedding extraction: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    def save_partial_dataset(self, layer_embeddings, labels, output_dir, part_num):
        """Save a chunk of processed data"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"part_{part_num}.pt")
            
            saved_data = {key: torch.stack(val) for key, val in layer_embeddings.items()}
            saved_data["labels"] = torch.tensor(labels)
            
            torch.save(saved_data, output_file)
            self.logger.info(f"Saved part {part_num} with {len(labels)} examples to {output_file}")
            
            # Clear memory after saving
            del saved_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return output_file
        except Exception as e:
            self.logger.error(f"Error saving partial dataset: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    def merge_saved_parts(self, part_files, output_file):
        """Merge all saved parts into a final dataset file"""
        self.logger.info(f"Merging {len(part_files)} partial datasets...")
        
        all_layer_embeddings = defaultdict(list)
        all_labels = []
        
        for part_file in part_files:
            try:
                part_data = torch.load(part_file)
                
                for key, val in part_data.items():
                    if key == "labels":
                        all_labels.append(val)
                    else:
                        all_layer_embeddings[key].append(val)
                        
                # Remove the part file after loading to free disk space
                os.remove(part_file)
                self.logger.debug(f"Processed and removed part file: {part_file}")
                
            except Exception as e:
                self.logger.error(f"Error processing part file {part_file}: {e}")
                self.logger.debug(traceback.format_exc())
        
        # Concatenate all tensors
        final_data = {}
        for key, val in all_layer_embeddings.items():
            final_data[key] = torch.cat(val, dim=0)
        
        final_data["labels"] = torch.cat(all_labels, dim=0)
        
        # Save final merged dataset
        torch.save(final_data, output_file)
        self.logger.info(f"✅ Final merged dataset saved to {output_file}")
        self.logger.info(f"🔢 Total examples: {len(final_data['labels'])}")
        self.logger.info(f"🧠 Layers saved: {list(all_layer_embeddings.keys())}")
        
        return final_data

    def process_dataset(self, dataset_name, split="train[:1000]", output_file="trail_dataset_all_layers.pt"):
        self.logger.info(f"Loading dataset: {dataset_name}, split: {split}")
        output_dir = os.path.dirname(output_file) or "."
        temp_dir = os.path.join(output_dir, f"temp_{Path(output_file).stem}")
        
        try:
            # Use streaming for memory efficiency when possible
            try:
                ds = load_dataset(dataset_name, split=split, streaming=True)
                self.logger.info(f"Dataset loaded in streaming mode")
                dataset_size = "unknown (streaming)"
            except:
                ds = load_dataset(dataset_name, split=split)
                dataset_size = len(ds)
                self.logger.info(f"Dataset loaded with {dataset_size} examples")
        except Exception as e:
            self.logger.critical(f"Failed to load dataset: {e}")
            self.logger.debug(traceback.format_exc())
            raise

        # Create temp directory for partial saves
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process examples with progress bar if possible
        self.logger.info("Processing dataset examples...")
        
        all_part_files = []
        current_part = 1
        successful_examples = 0
        all_layer_embeddings = defaultdict(list)
        all_labels = []
        
        # Create iterator with progress bar if dataset size is known
        if isinstance(dataset_size, int):
            dataset_iter = tqdm(enumerate(ds), total=dataset_size, desc="Processing examples")
        else:
            # For streaming datasets with unknown size
            dataset_iter = enumerate(ds)
        
        for i, example in dataset_iter:
            try:
                prompt = example["text"]
                self.logger.debug(f"Processing example {i+1}")
                layer_embs, labels = self.extract_embeddings_and_labels(prompt)

                for layer_key, embs in layer_embs.items():
                    all_layer_embeddings[layer_key].extend(embs)

                all_labels.extend(labels)
                successful_examples += 1
                
                # Save partial results at intervals to free memory
                if successful_examples % self.save_interval == 0:
                    part_file = self.save_partial_dataset(all_layer_embeddings, all_labels, temp_dir, current_part)
                    all_part_files.append(part_file)
                    current_part += 1
                    
                    # Clear memory after saving
                    all_layer_embeddings = defaultdict(list)
                    all_labels = []
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Log progress periodically 
                if successful_examples % 10 == 0:
                    self.logger.info(f"Successfully processed {successful_examples} examples")

            except Exception as e:
                self.logger.warning(f"Skipped example {i+1}: {e}")
                if "CUDA out of memory" in str(e):
                    self.logger.error("CUDA out of memory error - consider reducing batch size or model")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # Try to recover memory
                    
        # Save any remaining data
        if all_labels:
            part_file = self.save_partial_dataset(all_layer_embeddings, all_labels, temp_dir, current_part)
            all_part_files.append(part_file)
                    
        # Check if we have any data to save
        if not all_part_files:
            self.logger.critical("No examples were successfully processed. Cannot save dataset.")
            raise ValueError("No examples were successfully processed")

        # Merge all parts for final dataset
        try:
            self.merge_saved_parts(all_part_files, output_file)
            
            # Clean up temp directory if it's empty
            if len(os.listdir(temp_dir)) == 0:
                os.rmdir(temp_dir)
                
        except Exception as e:
            self.logger.critical(f"Failed to merge and save dataset: {e}")
            self.logger.debug(traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description="Generate dataset with all LLM layer embeddings for TRAIL")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins for length classification")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help="Dataset name")
    parser.add_argument("--split", type=str, default="train[:100]", help="Dataset split")
    parser.add_argument("--output", type=str, default="trail_dataset_all_layers.pt", help="Output file path")
    parser.add_argument("--log_file", type=str, default="embedding_extraction.log", help="Log file path")
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--save_interval", type=int, default=100, 
                       help="Save partial results after this many successful examples")
    parser.add_argument("--half_precision", action="store_true", 
                       help="Use half precision (FP16) to reduce memory usage")

    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(args.log_file, log_level)
    
    logger.info("Starting embedding extraction process")
    logger.info(f"Arguments: {args}")
    
    try:
        extractor = EmbeddingExtractor(
            model_name=args.model,
            max_length=args.max_length,
            num_bins=args.num_bins,
            logger=logger,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            use_half_precision=args.half_precision
        )

        extractor.process_dataset(
            dataset_name=args.dataset,
            split=args.split,
            output_file=args.output
        )
        logger.info("Process completed successfully")
    except Exception as e:
        logger.critical(f"Process failed with error: {e}")
        logger.critical(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
