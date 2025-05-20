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
        device=None,
        layer_range=(8, 16)
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_range = layer_range
        
        logger.info(f"Initializing EmbeddingExtractor with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Layer range: {self.layer_range}, Max length: {self.max_length}")
        
        # Load model and tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        
        logger.info(f"Loading model to {self.device}...")
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.model.eval()
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")

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
            labels.append(remaining)  # Use the actual remaining length as label

            token_pos = input_ids.shape[1] + i
            for layer_idx in range(*self.layer_range):
                if token_pos < hidden_states[layer_idx].shape[1]:
                    embedding = hidden_states[layer_idx][0][token_pos]
                    layer_embeddings[f"layer_{layer_idx}"].append(embedding.detach().cpu())
                else:
                    logger.warning(f"Skipped token position {token_pos} for layer {layer_idx} (out of bounds)")

        logger.debug(f"Extracted {len(labels)} tokens with embeddings")
        return layer_embeddings, labels, prompt_text, gen_len

    def extract_hidden_state_sequences_all_layers_to_single_tensor(
            self,
            dataset_name,
            split="train[:2000]",
            output_file="/lustre/pd01/hpc-tomasznaskret-1742832160/hidden_state_sequences_all_layers_entropy.pt"
    ):
        logger.info(f"Loading dataset: {dataset_name}, split: {split}")
        ds = load_dataset(dataset_name, split=split)

        layer_count = self.layer_range[1] - self.layer_range[0]
        all_layers_sequences = [[] for _ in range(layer_count)]
        collected = 0

        for example in tqdm(ds, desc="Collecting hidden states"):
            prompt = example["text"]

            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_attention_mask=True
                ).to(self.device)

                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                output_length = self.max_length - input_ids.shape[1]

                if output_length <= 0:
                    logger.warning(f"Prompt too long ({input_ids.shape[1]} tokens), skipping.")
                    continue

                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=output_length,
                        return_dict_in_generate=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    full_input = generated.sequences
                    full_attention_mask = (full_input != self.tokenizer.pad_token_id).long()

                    outputs = self.model(
                        input_ids=full_input,
                        attention_mask=full_attention_mask,
                        output_hidden_states=True
                    )

                all_hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors
                prompt_len = input_ids.shape[1]
                gen_len = full_input.shape[1] - prompt_len

                if gen_len < 2:
                    continue

                for i, layer_idx in enumerate(range(*self.layer_range)):
                    layer_states = all_hidden_states[layer_idx][0]  # [seq_len, hidden_size]
                    generated_states = layer_states[prompt_len:]  # [gen_len, hidden_size]
                    all_layers_sequences[i].append(generated_states.cpu())

                collected += 1

            except Exception as e:
                logger.warning(f"Failed to process prompt: {e}")
                continue

        logger.info(f"Collected {collected} sequences for each layer")

        # Determine max sequence length
        max_len = max(seq.shape[0] for layer_seqs in all_layers_sequences for seq in layer_seqs)
        hidden_size = all_layers_sequences[0][0].shape[1]

        # Create one big tensor: [num_layers, num_sequences, max_len, hidden_size]
        all_layers_tensor = torch.zeros((layer_count, collected, max_len, hidden_size))

        for layer_i, layer_seqs in enumerate(all_layers_sequences):
            for seq_i, seq in enumerate(layer_seqs):
                all_layers_tensor[layer_i, seq_i, :seq.shape[0], :] = seq

        torch.save(all_layers_tensor, output_file)
        logger.info(f"✅ Saved all layers tensor of shape {all_layers_tensor.shape} to {output_file}")

    def process_dataset(self, dataset_name, split="train[:2500]", output_file="trail_dataset_all_layers.pt", batch_size=10):
        logger.info(f"Loading dataset: {dataset_name}, split: {split}")
        ds = load_dataset(dataset_name, split=split)
        logger.info(f"Dataset loaded with {len(ds)} examples")
        
        all_layer_embeddings = defaultdict(list)
        all_labels = []
        all_queries = []
        all_gen_lengths = []

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
                        layer_embs, labels, query, gen_length = self.extract_embeddings_and_labels(prompt_text)
                        for layer_key, embs in layer_embs.items():
                            all_layer_embeddings[layer_key].extend(embs)
                        all_labels.extend(labels)
                        all_queries.append(query)
                        all_gen_lengths.append(gen_length)
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

        # Save the embeddings and labels
        logger.info(f"Saving embeddings and labels to {output_file}")
        saved_data = {key: torch.stack(val) for key, val in all_layer_embeddings.items()}
        saved_data["labels"] = torch.tensor(all_labels)
        torch.save(saved_data, output_file)
        
        # Count frequency of each label class
        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Save queries and generated lengths to a separate file
        metadata_file = os.path.splitext(output_file)[0] + "_metadata.pt"
        logger.info(f"Saving queries and generated lengths to {metadata_file}")
        metadata = {
            "queries": all_queries,
            "generated_lengths": all_gen_lengths,
            "label_counts": label_counts
        }
        torch.save(metadata, metadata_file)

        logger.info(f"\n✅ Dataset saved to {output_file}")
        logger.info(f"✅ Metadata saved to {metadata_file}")
        logger.info(f"🔢 Total examples: {len(all_labels)}")
        logger.info(f"🧠 Layers saved: {list(all_layer_embeddings.keys())}")
        logger.info(f"📊 Label distribution: {label_counts}")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset with all LLM layer embeddings for TRAIL")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
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
        max_length=args.max_length
    )

    # extractor.process_dataset(
    #     dataset_name=args.dataset,
    #     split=args.split,
    #     output_file=args.output
    # )

    extractor.extract_hidden_state_sequences_all_layers_to_single_tensor(
        dataset_name="tatsu-lab/alpaca",
        split="train[:2500]",
    )

    logger.info("Dataset generation completed successfully")


if __name__ == "__main__":
    main()