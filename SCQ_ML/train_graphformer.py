import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from make_dataset import CircuitDataset
import pandas as pd
import os
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

# --- hyper-parameters ---
MAX_EPOCHS    = 1000
BATCH_SIZE    = 32
LR            = 1e-4
WEIGHT_DECAY  = 1e-4

# --- setup dirs, device ---
os.makedirs('weights', exist_ok=True)
os.makedirs('logs', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# --- load data ---
dataset  = CircuitDataset(args.dataset_path, num_samples=args.num_samples)
train_size = int(0.8 * len(dataset))
test_size  = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
loader_tr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
loader_te = DataLoader(test_ds,  batch_size=BATCH_SIZE)

print(f"Training on {len(train_ds)} samples, testing on {len(test_ds)} samples")





# --- model, optimizer, scheduler ---
model     = DeepGraphTransformer().to(device)
# compute your true mean in log-space
all_y = torch.cat([d.y for d in dataset], dim=0)  # shape: [N, 2]
mean_log_omega = all_y.mean(0).to(device)          # shape: [2]
model.mlp_out[-1].bias.data.copy_(mean_log_omega)
print("Initialized final bias to:", model.mlp_out[-1].bias.data)

opt       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)

# --- evaluation fn ---
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch)
            total_loss += F.mse_loss(pred, data.y, reduction='sum').item()
    return total_loss / len(loader.dataset)

# --- untrained baseline ---
baseline = evaluate(loader_te)
print(f"Untrained model test MSE (log-space): {baseline:.4e}")

# --- training loop ---
best_test = float('inf')
log_data  = []

try:
    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        loss_ema = 0.0

        for data in loader_tr:
            data = data.to(device)
            opt.zero_grad()
            pred = model(data.x, data.edge_index, data.batch)
            loss = F.mse_loss(pred, data.y)
            loss.backward()
            opt.step()
            # exponential moving average for logging
            loss_ema = 0.9*loss_ema + 0.1*loss.item() if loss_ema else loss.item()

        # evaluate
        test_loss = evaluate(loader_te)
        print(f"epoch {epoch:4d}  train-MSE {loss_ema:.4e}  test-MSE {test_loss:.4e}")

        # scheduler step
        scheduler.step(test_loss)

        # best-checkpoint
        if test_loss < best_test:
            best_test = test_loss
            ckpt_path = f'weights/best_graphformer.pt'
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↪️  New best model saved (test-MSE {best_test:.4e})")

        # log
        log_data.append({
            'epoch': epoch,
            'train_loss': loss_ema,
            'test_loss':  test_loss,
            'lr':          opt.param_groups[0]['lr']
        })

finally:
    if log_data:
        df = pd.DataFrame(log_data)
        df.to_csv('logs/training_log_graphformer.csv', index=False)
        print(f"\nLogs saved to logs/training_log_graphformer.csv")

print("Training complete. Best test-MSE (log-space):", best_test)