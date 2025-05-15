# graph_binned_classifier.py
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import mean_absolute_error
import numpy as np
import random
import os
import logging
import gc
import argparse
from tqdm import tqdm
from logger import Logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("graph_binned_training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Cleared CUDA cache. Current memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class LayerwiseGraphDataset(InMemoryDataset):
    def __init__(self, embeddings_by_layer, labels, edge_mode="sequential", bin_edges=None):
        super().__init__()
        self.layer_names = list(embeddings_by_layer.keys())
        self.bin_edges = bin_edges
        self.data_list = self._build_graphs(embeddings_by_layer, labels, self.layer_names, edge_mode)

    def _build_graphs(self, embeddings_by_layer, labels, layer_order, edge_mode):
        graphs = []
        num_tokens = len(labels)

        if self.bin_edges is not None:
            binned_labels = np.clip(np.digitize(labels.numpy(), self.bin_edges) - 1, 0, len(self.bin_edges) - 2)

        for i in range(num_tokens):
            node_features = [embeddings_by_layer[layer][i] for layer in layer_order]
            x = torch.stack(node_features)
            y = torch.tensor([labels[i]], dtype=torch.float)

            edge_index = self._build_edge_index(len(layer_order), edge_mode)
            data = Data(x=x, edge_index=edge_index, y=y)

            if self.bin_edges is not None:
                data.bin_label = torch.tensor(binned_labels[i], dtype=torch.long)

            graphs.append(data)
        return graphs

    def _build_edge_index(self, num_nodes, mode):
        if mode == "sequential":
            edges = [[i, i + 1] for i in range(num_nodes - 1)]
        elif mode == "fully_connected":
            edges = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

class GraphBinnedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_bins=10):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, num_bins)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)

def compute_binned_mae(logits, true_lengths, bin_edges):
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    expected = (probs * midpoints).sum(axis=1)
    return mean_absolute_error(true_lengths.cpu().numpy(), expected)

def train(model, loader, optimizer, loss_fn, device, scaler, bin_edges, use_amp=False):
    model.train()
    total_loss = 0
    preds, labels_list = [], []
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(logits, batch.bin_label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        preds.append(logits.detach())
        labels_list.append(batch.y.detach())
    logits = torch.cat(preds, dim=0)
    labels = torch.cat(labels_list, dim=0)
    mae = compute_binned_mae(logits, labels, bin_edges)
    return total_loss / len(loader), mae

def evaluate(model, loader, loss_fn, device, bin_edges, use_amp=False):
    model.eval()
    total_loss = 0
    preds, labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation"):
            batch = batch.to(device)
            with autocast(enabled=use_amp):
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(logits, batch.bin_label)
            total_loss += loss.item()
            preds.append(logits.detach())
            labels_list.append(batch.y.detach())
    logits = torch.cat(preds, dim=0)
    labels = torch.cat(labels_list, dim=0)
    mae = compute_binned_mae(logits, labels, bin_edges)
    return total_loss / len(loader), mae

def load_dataset(path, edge_mode="sequential", bin_edges=None):
    logger.info(f"Loading data from {path}...")
    data = torch.load(path)
    embeddings_by_layer = {k: v for k, v in data.items() if k.startswith("layer_")}
    labels = data["labels"]
    dataset = LayerwiseGraphDataset(embeddings_by_layer, labels, edge_mode=edge_mode, bin_edges=bin_edges)
    return dataset

def split_dataset(dataset, seed=42):
    set_seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_end = int(0.7 * len(indices))
    val_end = int(0.85 * len(indices))
    return [dataset[i] for i in indices[:train_end]], [dataset[i] for i in indices[train_end:val_end]], [dataset[i] for i in indices[val_end:]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge_mode", type=str, choices=["sequential", "fully_connected"], default="sequential")
    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bin_edges = np.linspace(0, 512, 11)

    dataset = load_dataset(args.data_path, edge_mode=args.edge_mode, bin_edges=bin_edges)
    train_data, val_data, _ = split_dataset(dataset, seed=args.seed)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    input_dim = train_data[0].x.shape[1]
    model = GraphBinnedClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim, num_bins=len(bin_edges)-1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=args.use_amp)

    best_val_mae = float('inf')
    for epoch in range(args.num_epochs):
        train_loss, train_mae = train(model, train_loader, optimizer, loss_fn, device, scaler, bin_edges, use_amp=args.use_amp)
        val_loss, val_mae = evaluate(model, val_loader, loss_fn, device, bin_edges, use_amp=args.use_amp)

        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train MAE={train_mae:.4f}, Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}")

        if val_mae < best_val_mae - 0.001:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            logger.info("Saved new best model.")

if __name__ == "__main__":
    main()
