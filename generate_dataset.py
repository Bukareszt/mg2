import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
from collections import defaultdict

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
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.model.eval()

        # Bin edges for remaining length
        self.bin_edges = torch.linspace(0, self.max_length, self.num_bins + 1)

    def bin_remaining_length(self, length):
        return torch.bucketize(torch.tensor([length]), self.bin_edges)[0] - 1

    def extract_embeddings_and_labels(self, prompt_text):
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        output_length = self.max_length - input_ids.shape[1]

        if output_length <= 0:
            raise ValueError(f"Prompt too long ({input_ids.shape[1]} tokens), skipping.")

        generated = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=output_length,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

        tokens_generated = generated.sequences[0][input_ids.shape[1]:]
        hidden_states = generated.hidden_states  # Tuple: [num_layers+1][batch, seq_len, dim]

        layer_embeddings = defaultdict(list)
        labels = []

        for i in range(len(tokens_generated)):
            remaining = len(tokens_generated) - i - 1
            bin_id = self.bin_remaining_length(remaining)
            labels.append(bin_id)

            for layer_idx in range(*self.layer_range):  # Only layers 8 to 16 (inclusive)
                embedding = hidden_states[layer_idx][0][-len(tokens_generated) + i]
                layer_embeddings[f"layer_{layer_idx}"].append(embedding.detach().cpu())

        return layer_embeddings, labels

    def process_dataset(self, dataset_name, split="train[:1000]", output_file="trail_dataset_all_layers.pt"):
        ds = load_dataset(dataset_name, split=split)

        # Prepare dictionary to accumulate embeddings per layer
        all_layer_embeddings = defaultdict(list)
        all_labels = []

        for example in ds:
            try:
                prompt = example["text"]
                layer_embs, labels = self.extract_embeddings_and_labels(prompt)

                for layer_key, embs in layer_embs.items():
                    all_layer_embeddings[layer_key].extend(embs)

                all_labels.extend(labels)

            except Exception as e:
                print(f"Skipped: {e}")

        # Stack and save
        saved_data = {key: torch.stack(val) for key, val in all_layer_embeddings.items()}
        saved_data["labels"] = torch.tensor(all_labels)

        torch.save(saved_data, output_file)

        print(f"\n✅ Dataset saved to {output_file}")
        print(f"🔢 Total examples: {len(all_labels)}")
        print(f"🧠 Layers saved: {list(all_layer_embeddings.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset with all LLM layer embeddings for TRAIL")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins for length classification")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help="Dataset name")
    parser.add_argument("--split", type=str, default="train[:1000]", help="Dataset split")
    parser.add_argument("--output", type=str, default="trail_dataset_all_layers.pt", help="Output file path")

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
