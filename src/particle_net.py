"""
ParticleNet: Jet Tagging via Particle Clouds
=============================================
Based on: Qu & Gouskos, Phys. Rev. D 101, 056019 (2020)

Architecture:
  - Treats jets as unordered sets ("particle clouds")
  - Dynamic Graph CNN with EdgeConv operations
  - k-nearest neighbors (kNN) in learned feature space
  - Edge features capture local geometric structure

Physics motivation:
  Unlike jet images, particle clouds preserve the exact positions and
  momenta of jet constituents without binning losses. The kNN graph
  naturally captures the angular clustering of QCD radiation, and the
  dynamic graph updates allow the network to discover hierarchical
  structure (subjets within jets).

  This is analogous to how ATLAS/CMS cluster particles into subjets
  using sequential recombination algorithms (anti-kT, Cambridge/Aachen).

Implementation notes:
  This is a custom implementation that does NOT require PyTorch Geometric.
  We implement EdgeConv from scratch using batched kNN and edge feature
  construction, making this more portable and educational.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PARTICLENET_CONFIG


def knn(x, k):
    """
    Compute k-nearest neighbors using pairwise distances.

    In particle physics, the "nearest neighbors" in (η, φ) space
    correspond to particles that are angularly close — these are
    the particles most likely to be from the same parton shower.

    Args:
        x: (batch, N, C) point coordinates
        k: number of neighbors

    Returns:
        idx: (batch, N, k) indices of k nearest neighbors
    """
    # Pairwise squared distances: ||x_i - x_j||²
    inner = -2 * torch.matmul(x, x.transpose(2, 1))  # (B, N, N)
    xx = torch.sum(x**2, dim=2, keepdim=True)  # (B, N, 1)
    dist = xx + inner + xx.transpose(2, 1)  # (B, N, N)

    # Get k smallest distances (excluding self)
    # Use negative distance so topk gives nearest
    _, idx = (-dist).topk(k=k + 1, dim=-1)  # +1 to exclude self
    return idx[:, :, 1:]  # Remove self-loop, shape: (B, N, k)


def get_edge_features(x, idx):
    """
    Construct edge features for EdgeConv.

    For each point i and its neighbor j, the edge feature is:
        [x_j - x_i, x_i]  (concatenation)

    This encodes both the local structure (difference) and the
    absolute position (center point features).

    Args:
        x: (batch, N, C) point features
        idx: (batch, N, k) neighbor indices

    Returns:
        edge_features: (batch, N, k, 2*C)
    """
    B, N, C = x.shape
    k = idx.shape[2]

    # Gather neighbor features
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)  # (B, N, k, C)
    x_expanded = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, C)
    neighbors = torch.gather(x_expanded, 2, idx_expanded)  # (B, N, k, C)

    # Edge features: [neighbor - center, center]
    center = x.unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, k, C)
    edge_features = torch.cat([neighbors - center, center], dim=-1)  # (B, N, k, 2C)

    return edge_features


class EdgeConvBlock(nn.Module):
    """
    EdgeConv block from DGCNN (Dynamic Graph CNN).

    Steps:
      1. Compute kNN graph in current feature space
      2. Construct edge features [x_j - x_i || x_i]
      3. Apply shared MLP to each edge
      4. Aggregate (max) over edges → new point features
      5. Shortcut connection

    This mimics how physicists analyze jet substructure:
    look at local radiation patterns (edge features) and
    aggregate information hierarchically (max pooling).
    """

    def __init__(self, in_channels, mlp_dims, k=16):
        super().__init__()
        self.k = k

        # Shared MLP applied to each edge
        layers = []
        dims = [2 * in_channels] + mlp_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Conv2d(dims[i], dims[i + 1], 1, bias=False))
            layers.append(nn.BatchNorm2d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != mlp_dims[-1]:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, mlp_dims[-1], 1, bias=False),
                nn.BatchNorm1d(mlp_dims[-1]),
            )

    def forward(self, x, coords=None, mask=None):
        """
        Args:
            x: (batch, C, N) point features
            coords: (batch, N, 2) coordinates for kNN (optional, uses x if None)
            mask: (batch, N) boolean mask for real particles

        Returns:
            x_new: (batch, C_out, N) updated features
        """
        B, C, N = x.shape

        # Use coordinates for kNN if provided, else use features
        if coords is not None:
            knn_input = coords
        else:
            knn_input = x.transpose(1, 2)  # (B, N, C)

        # Compute kNN
        idx = knn(knn_input, self.k)  # (B, N, k)

        # Get edge features
        x_t = x.transpose(1, 2)  # (B, N, C)
        edge_feat = get_edge_features(x_t, idx)  # (B, N, k, 2C)
        edge_feat = edge_feat.permute(0, 3, 1, 2)  # (B, 2C, N, k) for Conv2d

        # Apply MLP to edges
        edge_feat = self.mlp(edge_feat)  # (B, C_out, N, k)

        # Aggregate: max over neighbors
        x_new = edge_feat.max(dim=-1)[0]  # (B, C_out, N)

        # Shortcut connection
        x_new = x_new + self.shortcut(x)

        # Apply mask
        if mask is not None:
            x_new = x_new * mask.unsqueeze(1)

        return x_new


class ParticleNet(nn.Module):
    """
    ParticleNet: Dynamic Graph CNN for jet classification.

    Architecture:
      Input features → EdgeConv(64) → EdgeConv(128) → EdgeConv(256)
      → Global Average Pool → FC(256) → FC(1)

    The dynamic graph is recomputed at each layer, allowing the network
    to discover different neighborhood structures at different scales.
    Early layers capture local radiation patterns; later layers see
    the global jet structure.
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or PARTICLENET_CONFIG
        k = cfg["k_neighbors"]
        edge_dims = cfg["edge_conv_dims"]
        fc_dims = cfg["fc_dims"]
        dropout = cfg["dropout"]
        input_features = cfg["input_features"]
        self.coord_features = cfg["coord_features"]

        # EdgeConv blocks
        self.edge_convs = nn.ModuleList()
        in_dim = input_features
        for dims in edge_dims:
            self.edge_convs.append(EdgeConvBlock(in_dim, dims, k=k))
            in_dim = dims[-1]

        # Classifier
        fc_layers = []
        fc_in = in_dim
        for fc_out in fc_dims:
            fc_layers.extend([
                nn.Linear(fc_in, fc_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            fc_in = fc_out
        fc_layers.append(nn.Linear(fc_in, 1))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, features, mask):
        """
        Args:
            features: (batch, N, C) particle features
            mask: (batch, N) boolean mask

        Returns:
            logits: (batch,) raw logits
        """
        # Extract coordinates for initial kNN
        # Use delta_eta, delta_phi (indices 4, 5 in raw features, but
        # after normalization we pass the full feature tensor)
        coords = features[:, :, :self.coord_features]  # (B, N, 2)

        # Transpose for Conv operations: (B, C, N)
        x = features.transpose(1, 2)

        # Apply EdgeConv blocks
        for i, edge_conv in enumerate(self.edge_convs):
            # First layer uses (η,φ) coordinates; later layers use learned features
            c = coords if i == 0 else None
            x = edge_conv(x, coords=c, mask=mask)

        # Global average pooling (masked)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        x = (x * mask.unsqueeze(1)).sum(dim=2) / mask_sum  # (B, C)

        return self.classifier(x).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ParticleNet()
    print(f"ParticleNet: {model.count_parameters():,} trainable parameters")

    B, N, C = 4, 100, 4
    features = torch.randn(B, N, C)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 50:] = False  # Half padded

    out = model(features, mask)
    print(f"Input: features={features.shape}, mask={mask.shape}")
    print(f"Output: {out.shape}")
