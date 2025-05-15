# Binned classification version of token length prediction
# Implements TRAIL-style classification into bins and expected length regression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import logging
import gc
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import argparse
from logger import Logger

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Cleared CUDA cache. Current memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# --- Binned classifier ---
class BinnedLengthPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_bins=10):
        super(BinnedLengthPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins)
        )

    def forward(self, x):
        return self.model(x)

# --- Dataset class with binning ---
class BinnedDataset(Dataset):
    def __init__(self, embeddings, labels, bin_edges):
        self.embeddings = embeddings
        self.labels = labels
        self.bin_edges = bin_edges
        self.binned_labels = torch.tensor(
            np.clip(np.digitize(labels.numpy(), bin_edges) - 1, 0, len(bin_edges) - 2),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx],
            "bin_label": self.binned_labels[idx]
        }

# --- Collate function ---
def custom_collate_fn(batch):
    embeddings = torch.stack([item["embedding"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)
    bin_labels = torch.tensor([item["bin_label"] for item in batch], dtype=torch.long)
    return {"embeddings": embeddings, "labels": labels, "bin_labels": bin_labels}

# --- MAE from expected value ---
def compute_binned_mae(logits, true_lengths, bin_edges):
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    expected = (probs * midpoints).sum(axis=1)
    return mean_absolute_error(true_lengths.cpu().numpy(), expected)

# --- Aggregate embeddings from multiple layers ---
def aggregate_layers(data, layer_names, aggregation='concat'):
    """
    Aggregate embeddings from multiple layers using the specified method
    
    Args:
        data: Dictionary with layer embeddings
        layer_names: List of layer names to aggregate
        aggregation: Method to use for aggregation ('mean', 'sum', or 'concat')
        
    Returns:
        Aggregated tensor
    """
    if aggregation == 'mean':
        return torch.mean(torch.stack([data[layer] for layer in layer_names]), dim=0)
    elif aggregation == 'sum':
        return torch.sum(torch.stack([data[layer] for layer in layer_names]), dim=0)
    elif aggregation == 'concat':
        return torch.cat([data[layer] for layer in layer_names], dim=1)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")

# --- Data loader and split ---
def load_and_split_dataset(data_path, layer_names, bin_edges, aggregation='concat', seed=42):
    data = torch.load(data_path)
    embeddings = aggregate_layers(data, layer_names, aggregation)
    labels = data["labels"].float()

    indices = np.random.RandomState(seed).permutation(len(labels))
    train_split = int(0.7 * len(labels))
    val_split = int(0.85 * len(labels))
    train_idx, val_idx, test_idx = indices[:train_split], indices[train_split:val_split], indices[val_split:]

    return (
        BinnedDataset(embeddings[train_idx], labels[train_idx], bin_edges),
        BinnedDataset(embeddings[val_idx], labels[val_idx], bin_edges),
        BinnedDataset(embeddings[test_idx], labels[test_idx], bin_edges),
    )

# --- Training loop ---
def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bin_edges = np.linspace(0, 512, 11)
    train_set, val_set, _ = load_and_split_dataset(
        args.data_path, 
        args.layer_names, 
        bin_edges, 
        aggregation=args.aggregation,
        seed=args.seed
    )
    input_dim = train_set.embeddings.shape[1]

    model = BinnedLengthPredictor(input_dim=input_dim, hidden_dim=args.hidden_dim, num_bins=len(bin_edges) - 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=args.use_amp)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    best_val_mae = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x = batch['embeddings'].to(device)
            y_bin = batch['bin_labels'].to(device)

            optimizer.zero_grad()
            with autocast(enabled=args.use_amp):
                logits = model(x)
                loss = criterion(logits, y_bin)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_logits = []
        all_true = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['embeddings'].to(device)
                y_real = batch['labels'].to(device)
                logits = model(x)
                all_logits.append(logits)
                all_true.append(y_real)
        all_logits = torch.cat(all_logits)
        all_true = torch.cat(all_true)
        val_mae = compute_binned_mae(all_logits, all_true, bin_edges)
        logger.info(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val MAE: {val_mae:.4f}")

        scheduler.step(val_mae)
        if val_mae < best_val_mae - args.min_loss_improvement:
            best_val_mae = val_mae
            output_dir = os.path.join(args.output_dir, f"{args.aggregation}")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

# --- Seed ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- Main ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--layer_names', nargs='+', default=['layer_8'], help="List of layer names to aggregate")
    parser.add_argument('--aggregation', type=str, choices=['mean', 'sum', 'concat'], default='concat',
                        help="Method to aggregate embeddings across layers")
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_loss_improvement', type=float, default=0.001)
    parser.add_argument('--use_amp', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    free_gpu_memory()
    train_model(args)
