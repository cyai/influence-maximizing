"""
GNN Node Embedding Training via Graph Autoencoder (GAE)

Dataset  : p2p-Gnutella08 (6,301 nodes, 20,777 directed edges)
Objective: Learn node embeddings H such that
           P_uv = sigmoid(h_v · h_u^T) approximates edge existence.
           This matches the influence probability formula used downstream.
Device   : Apple MPS (M-series GPU) → CPU fallback
Outputs  : node_embeddings.npy  – shape (N, 128)
           node_embeddings.pt   – same as PyTorch tensor
           embeddings_meta.json – training metadata

nohup python train_gnn.py > train_gnn.log 2>&1 &
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling, to_undirected
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "data/p2p-Gnutella08.txt"
EMBED_DIM = 128       # larger embedding → more expressive influence probabilities
HIDDEN_DIM = 256      # wider hidden layers
NUM_LAYERS = 3        # deeper encoder
EPOCHS = int(os.environ.get("GNN_EPOCHS", 1500))
LR = 3e-3
WEIGHT_DECAY = 1e-5
LABEL_SMOOTHING = 0.05   # prevents overconfident scores, helps generalisation
DROPOUT = 0.3
SEED = 42
TRAIN_RATIO = 0.85
VAL_RATIO = 0.05
# TEST_RATIO = 0.10  (implied)
NEG_SAMPLING_RATIO = 1.0  # negatives per positive edge
PAGERANK_ITERS = 40       # power-iteration steps for PageRank feature

CHECKPOINT_DIR = "checkpoints"
OUT_EMBEDDINGS_NPY = "node_embeddings.npy"
OUT_EMBEDDINGS_PT = "node_embeddings.pt"
OUT_META_JSON = "embeddings_meta.json"


# ---------------------------------------------------------------------------
# Device setup — MPS for Apple Silicon, else CPU
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: Apple MPS (Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device


# ---------------------------------------------------------------------------
# Data loading + feature engineering
# ---------------------------------------------------------------------------

def _pagerank(edge_index: torch.Tensor, num_nodes: int,
              damping: float = 0.85, num_iters: int = PAGERANK_ITERS) -> torch.Tensor:
    """Power-iteration PageRank (no external dependency)."""
    out_deg = torch.zeros(num_nodes)
    out_deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    out_deg = out_deg.clamp(min=1.0)

    pr = torch.full((num_nodes,), 1.0 / num_nodes)
    for _ in range(num_iters):
        contrib = pr[edge_index[0]] / out_deg[edge_index[0]]
        new_pr = torch.zeros(num_nodes)
        new_pr.scatter_add_(0, edge_index[1], contrib)
        pr = damping * new_pr + (1.0 - damping) / num_nodes
    return pr


def load_graph(path: str) -> Data:
    """Parse edge-list file and return a PyG Data object with node features.

    Features (5 per node):
      0  norm_in_degree   — fraction of max in-degree
      1  norm_out_degree  — fraction of max out-degree
      2  norm_total_degree
      3  degree_ratio     — out / (in + 1)  captures hub vs. sink role
      4  pagerank         — importance in the global information flow
    """
    edges = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v = map(int, line.split())
            edges.append((u, v))

    edges_t = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
    num_nodes = int(edges_t.max().item()) + 1

    out_deg = torch.zeros(num_nodes, dtype=torch.float)
    in_deg  = torch.zeros(num_nodes, dtype=torch.float)
    out_deg.scatter_add_(0, edges_t[0], torch.ones(edges_t.size(1)))
    in_deg.scatter_add_(0,  edges_t[1], torch.ones(edges_t.size(1)))
    total_deg = out_deg + in_deg

    def norm(t: torch.Tensor) -> torch.Tensor:
        mx = t.max()
        return t / mx if mx > 0 else t

    deg_ratio = out_deg / (in_deg + 1.0)   # out/in ratio; high → influencer node
    pr = _pagerank(edges_t, num_nodes)

    x = torch.stack([
        norm(in_deg),
        norm(out_deg),
        norm(total_deg),
        norm(deg_ratio),
        norm(pr),
    ], dim=1)  # (N, 5)

    data = Data(x=x, edge_index=edges_t, num_nodes=num_nodes)
    print(f"Graph loaded  — nodes: {num_nodes:,}  directed edges: {edges_t.size(1):,}  "
          f"features per node: {x.size(1)}")
    return data


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class GraphSAGEEncoder(nn.Module):
    """Multi-layer GraphSAGE encoder with BatchNorm and residual connections.

    Architecture per layer:
        h = SAGEConv(x)  →  BN  →  ReLU  →  Dropout  →  h + skip(x)
    The skip connection uses a learned linear projection when dims differ,
    or identity when dims match. The final layer omits activation so the
    embedding space is unconstrained (suitable for dot-product decoding).
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(dims[i + 1]) for i in range(num_layers)]
        )
        # Residual projections: identity if same dim, else 1×1 linear
        self.skips = nn.ModuleList([
            nn.Identity() if dims[i] == dims[i + 1]
            else nn.Linear(dims[i], dims[i + 1], bias=False)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, bn, skip) in enumerate(zip(self.convs, self.bns, self.skips)):
            h = conv(x, edge_index)
            h = bn(h)
            is_last = (i == len(self.convs) - 1)
            if not is_last:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            x = h + skip(x)   # residual on every layer
        return x  # (N, out_channels)


class GAE(nn.Module):
    """Graph Autoencoder with dot-product decoder.

    Decoder: score(u, v) = h_u · h_v^T  (logit)
    Probability: P_uv = sigmoid(score(u, v))
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return raw logits for edges in edge_index."""
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)  # dot product per edge

    def decode_all_pairs(self, z: torch.Tensor) -> torch.Tensor:
        """Full N×N logit matrix (expensive — use only for small graphs / analysis)."""
        return torch.sigmoid(z @ z.t())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encode(x, edge_index)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_loss(
    model: GAE,
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    smoothing: float = LABEL_SMOOTHING,
) -> torch.Tensor:
    """BCEWithLogitsLoss with label smoothing on positive and negative edges.

    Label smoothing shifts hard 0/1 targets to (ε/2) and (1-ε/2) so the model
    cannot become overconfident, which improves generalisation on unseen edges.
    """
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1),
        method="sparse",
    )

    pos_logits = model.decode(z, pos_edge_index)
    neg_logits = model.decode(z, neg_edge_index)

    logits = torch.cat([pos_logits, neg_logits])
    # Smoothed labels: positives → 1-ε/2, negatives → ε/2
    pos_label = 1.0 - smoothing / 2.0
    neg_label = smoothing / 2.0
    labels = torch.cat([
        torch.full((pos_logits.size(0),), pos_label, device=logits.device),
        torch.full((neg_logits.size(0),), neg_label, device=logits.device),
    ])
    return F.binary_cross_entropy_with_logits(logits, labels)


@torch.no_grad()
def evaluate(
    model: GAE,
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
) -> float:
    """Compute AUC on held-out positive and negative edges."""
    pos_scores = torch.sigmoid(model.decode(z, pos_edge_index)).cpu().numpy()
    neg_scores = torch.sigmoid(model.decode(z, neg_edge_index)).cpu().numpy()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    return roc_auc_score(labels, scores)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(data_path: str = DATA_PATH) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Load graph ---
    data = load_graph(data_path)
    num_nodes = data.num_nodes

    # --- Edge split: keep directed edges for message passing undirected ---
    # For the training signal we use directed edges; for the GNN aggregation
    # we symmetrise so each node receives messages from both in- and out-neighbours.
    undirected_edge_index = to_undirected(data.edge_index, num_nodes=num_nodes)

    # Train/val/test split on the original directed edges
    transform = RandomLinkSplit(
        num_val=VAL_RATIO,
        num_test=1.0 - TRAIN_RATIO - VAL_RATIO,
        is_undirected=False,
        add_negative_train_samples=False,  # train negatives are sampled on-the-fly
        neg_sampling_ratio=1.0,            # add 1:1 negatives to val/test for AUC eval
    )
    train_data, val_data, test_data = transform(
        Data(x=data.x, edge_index=data.edge_index, num_nodes=num_nodes)
    )

    # Move everything to device
    x = data.x.to(device)
    msg_edge_index = undirected_edge_index.to(device)  # for GNN propagation

    train_pos = train_data.edge_label_index.to(device)
    val_pos   = val_data.edge_label_index[:, val_data.edge_label == 1].to(device)
    val_neg   = val_data.edge_label_index[:, val_data.edge_label == 0].to(device)

    print(f"Train edges: {train_pos.size(1):,}  "
          f"Val pos: {val_pos.size(1):,}  Val neg: {val_neg.size(1):,}")

    # --- Model & optimiser ---
    encoder = GraphSAGEEncoder(
        in_channels=data.x.size(1),
        hidden_channels=HIDDEN_DIM,
        out_channels=EMBED_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    model = GAE(encoder).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # ReduceLROnPlateau cuts LR when val AUC stops improving — better than fixed cosine
    # for link-prediction tasks that can plateau early.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=30, min_lr=1e-5
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    best_val_auc = 0.0
    best_epoch = 0
    best_state = None
    val_auc = 0.0
    train_start = time.time()

    pbar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        optimiser.zero_grad()

        z = model.encode(x, msg_edge_index)
        loss = compute_loss(model, z, train_pos, num_nodes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                z_eval = model.encode(x, msg_edge_index)
            val_auc = evaluate(model, z_eval, val_pos, val_neg)
            scheduler.step(val_auc)   # ReduceLROnPlateau needs the metric

            current_lr = optimiser.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                val_auc=f"{val_auc:.4f}",
                lr=f"{current_lr:.2e}",
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

    elapsed = time.time() - train_start
    print(f"\nTraining complete in {elapsed:.1f}s — "
          f"best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    # --- Extract final embeddings from best checkpoint ---
    if best_state is None:
        # Fallback: no checkpoint was saved (e.g. all val AUCs were NaN)
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z_final = model.encode(x, msg_edge_index)

    embeddings_np = z_final.cpu().numpy()   # (N, 64)
    embeddings_pt = z_final.cpu()

    np.save(OUT_EMBEDDINGS_NPY, embeddings_np)
    torch.save(embeddings_pt, OUT_EMBEDDINGS_PT)

    meta = {
        "num_nodes": num_nodes,
        "num_features": int(data.x.size(1)),
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "label_smoothing": LABEL_SMOOTHING,
        "epochs_trained": EPOCHS,
        "best_epoch": best_epoch,
        "best_val_auc": round(best_val_auc, 6),
        "train_edges": int(train_pos.size(1)),
        "model_params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "elapsed_seconds": round(elapsed, 2),
        "output_files": {
            "embeddings_npy": OUT_EMBEDDINGS_NPY,
            "embeddings_pt": OUT_EMBEDDINGS_PT,
        },
        "usage_note": (
            "P_uv = torch.sigmoid((H[v] * H[u]).sum(dim=-1))  "
            "where H = torch.load('node_embeddings.pt')"
        ),
    }
    with open(OUT_META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved:")
    print(f"  {OUT_EMBEDDINGS_NPY}  — shape {embeddings_np.shape}")
    print(f"  {OUT_EMBEDDINGS_PT}")
    print(f"  {OUT_META_JSON}")
    print("\nTo compute influence probability P(u -> v):")
    print("  H = torch.load('node_embeddings.pt')")
    print("  P_uv = torch.sigmoid((H[v] * H[u]).sum(dim=-1))")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
