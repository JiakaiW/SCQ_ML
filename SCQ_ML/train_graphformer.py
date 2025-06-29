import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from make_dataset import CircuitDataset
import pandas as pd
import os
import argparse

from torch_geometric.nn import TransformerConv, global_mean_pool, GlobalAttention
from torch_geometric.nn.aggr import AttentionalAggregation
import torch.nn.functional as F
import torch.nn as nn

class DeepGraphTransformer(nn.Module):
    def __init__(self,
                 in_dim  = 4,
                 hidden  = 128,
                 heads   = 8,
                 layers  = 6,
                 dropout = 0.0):
        super().__init__()

        # 0️⃣  embed 4-D raw features → hidden
        self.input_proj = nn.Linear(in_dim, hidden)

        # ①  transformer blocks (all now hidden→hidden)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(
                nn.ModuleDict({
                    "conv": TransformerConv(
                        hidden, hidden,
                        heads=heads, concat=False,
                        dropout=dropout, beta=True),
                    "norm1": nn.LayerNorm(hidden),
                    "ffn": nn.Sequential(
                        nn.Linear(hidden, hidden*2),
                        nn.GELU(),
                        nn.Linear(hidden*2, hidden),
                        nn.Dropout(dropout)),
                    "norm2": nn.LayerNorm(hidden)
                }))

        # ②  attention-style read-out  (modern API)
        gate_nn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1))
        self.pool = AttentionalAggregation(gate_nn)   # PyG ≥ 2.4

        # ③  regression head in log-space
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2))

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.input_proj(x)                # (N, 128)

        for blk in self.blocks:
            h = blk["conv"](x, edge_index, edge_attr)
            x = blk["norm1"](F.gelu(h) + x)   # residual-1
            h2 = blk["ffn"](x)
            x = blk["norm2"](h2 + x)          # residual-2

        x = self.pool(x, batch)               # (B, 128)
        return self.mlp_out(x)                # log ω₀₁, log ω₁₂


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Train a GNN on circuit data.')
parser.add_argument('--dataset_path', type=str, required=True, 
                    help='Path to the root directory of the PyG dataset.')
parser.add_argument('--num_samples', type=int, default=1000,
                    help='Number of samples per topology in the dataset.')
args = parser.parse_args()

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories for weights and logs
os.makedirs('weights', exist_ok=True)
os.makedirs('logs', exist_ok=True)

print(f"Loading dataset from: {args.dataset_path}")
dataset   = CircuitDataset(args.dataset_path, num_samples=args.num_samples)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train, test = torch.utils.data.random_split(dataset, [train_size, test_size])
loader_tr = DataLoader(train, batch_size=32, shuffle=True)
loader_te = DataLoader(test,  batch_size=64)

# Print sample shapes for debugging
sample = next(iter(loader_tr))
print(f"Sample batch shapes:")
print(f"x: {sample.x.shape}")
print(f"edge_index: {sample.edge_index.shape}")
print(f"y: {sample.y.shape}")

def evaluate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch)
            total_loss += F.mse_loss(pred, data.y, reduction='sum').item()
    return total_loss / len(loader.dataset)

model = DeepGraphTransformer().to(device)
opt = torch.optim.AdamW(model.parameters())

print(f"Training on {len(train)} samples, testing on {len(test)} samples")

log_data = []
try:
    for epoch in range(200):
        model.train()
        loss_ema = 0
        for data in loader_tr:
            data = data.to(device)
            opt.zero_grad()
            pred = model(data.x, data.edge_index, data.batch)
            loss = F.mse_loss(pred, data.y)
            loss.backward()
            opt.step()
            loss_ema = 0.9*loss_ema + 0.1*loss.item() if loss_ema > 0 else loss.item()
        
        test_loss = evaluate(loader_te)
        print(f"epoch {epoch:3d}  train-MSE {loss_ema:.4e}  test-MSE {test_loss:.4e}")
        
        # Save model weights
        torch.save(model.state_dict(), f'weights/model_epoch_{epoch}_graphformer.pt')
        
        # Log data
        log_data.append({'epoch': epoch, 'train_loss': loss_ema, 'test_loss': test_loss})
finally:
    if log_data:
        df_log = pd.DataFrame(log_data)
        df_log.to_csv('logs/training_log_graphformer.csv', index=False)
        print("\nPartial or complete logs saved to logs/training_log_graphformer.csv")

print("Script finished.")
