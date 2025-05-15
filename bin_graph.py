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
from logger import Logger  # Assumes logger.py exists

# Set up logging
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

class LayerwiseGraphBinnedDataset(InMemoryDataset):
    def __init__(self, embeddings_by_layer, labels, bin_edges, edge_mode="sequential"):
        super().__init__()
        self.layer_names = list(embeddings_by_layer.keys())
        self.bin_edges = bin_edges
        self.true_lengths = labels
        self.binned_labels = torch.tensor(
            np.clip(np.digitize(labels.numpy(), bin_edges) - 1, 0, len(bin_edges) - 2),
            dtype=torch.long
        )
        self.data_list = self._build_graphs(embeddings_by_layer, self.binned_labels, self.layer_names, edge_mode)

    def _build_graphs(self, embeddings_by_layer, bin_labels, layer_order, edge_mode):
        graphs = []
        for i in range(len(bin_labels)):
            node_features = [embeddings_by_layer[layer][i] for layer in layer_order]
            x = torch.stack(node_features)
            y = bin_labels[i]

            num_nodes = len(layer_order)
            if edge_mode == "sequential":
                edge_index = [[j, j + 1] for j in range(num_nodes - 1)]
            elif edge_mode == "fully_connected":
                edge_index = [[a, b] for a in range(num_nodes) for b in range(num_nodes) if a != b]
            else:
                raise ValueError(f"Unsupported edge_mode: {edge_mode}")

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            graphs.append(Data(x=x, edge_index=edge_index, y=y))

        return graphs

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

class GraphBinnedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_bins=10):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.linear = nn.Linear(hidden_dim // 2, num_bins)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.linear(x)  # [batch_size, num_bins]

def compute_binned_mae(logits, true_lengths, bin_edges):
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    expected = (probs * midpoints).sum(axis=1)
    return mean_absolute_error(true_lengths.cpu().numpy(), expected)

def load_dataset(path, bin_edges, edge_mode="sequential"):
    logger.info(f"🔄 Loading data from {path}...")
    data = torch.load(path)
    embeddings_by_layer = {k: v for k, v in data.items() if k.startswith("layer_")}
    labels = data["labels"]
    dataset = LayerwiseGraphBinnedDataset(embeddings_by_layer, labels, bin_edges, edge_mode)
    logger.info(f"✅ Created dataset with {len(dataset)} graphs")
    return dataset, labels

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    set_seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_end = int(train_ratio * len(indices))
    val_end = train_end + int(val_ratio * len(indices))
    return (
        [dataset[i] for i in indices[:train_end]],
        [dataset[i] for i in indices[train_end:val_end]],
        [dataset[i] for i in indices[val_end:]]
    )

def train_and_evaluate(args):
    free_gpu_memory()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    bin_edges = np.linspace(0, 512, args.num_bins + 1)
    dataset, true_lengths = load_dataset(args.data_path, bin_edges, edge_mode=args.edge_mode)
    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    sample = train_set[0]
    input_dim = sample.x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphBinnedClassifier(input_dim, hidden_dim=args.hidden_dim, num_bins=args.num_bins).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=args.use_amp)

    wandb_logger = Logger(
        config=vars(args),
        model_name=f"graph-binned-{args.edge_mode}",
        project_name=args.wandb_project,
        enable_logging=args.use_wandb,
        log_model=args.log_model
    )

    best_val_mae = float('inf')
    early_stop_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            with autocast(enabled=args.use_amp):
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_logits, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with autocast(enabled=args.use_amp):
                    logits = model(batch.x, batch.edge_index, batch.batch)
                all_logits.append(logits)
                all_true.append(batch.y)

        logits = torch.cat(all_logits)
        true_bins = torch.cat(all_true)
        val_mae = compute_binned_mae(logits, true_lengths[true_bins.cpu()], bin_edges)

        logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")
        scheduler.step(val_mae)

        if args.use_wandb:
            wandb_logger.log_metrics({"train/loss": train_loss, "val/mae": val_mae, "lr": optimizer.param_groups[0]['lr']}, step=epoch)

        if val_mae < best_val_mae - args.min_loss_improvement:
            best_val_mae = val_mae
            early_stop_counter = 0
            path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), path)
            if args.use_wandb and args.log_model:
                wandb_logger.log_model_checkpoint(model, path, f"best_model_epoch_{epoch}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

    if args.use_wandb:
        wandb_logger.finish()
    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_loss_improvement', type=float, default=0.001)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--edge_mode', type=str, choices=['sequential', 'fully_connected'], default='sequential')
    parser.add_argument('--num_bins', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='graph-binned-predictor')
    parser.add_argument('--log_model', action='store_true')

    args = parser.parse_args()
    train_and_evaluate(args)