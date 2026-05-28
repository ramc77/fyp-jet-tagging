"""
Data Loading and Preprocessing for Top Quark Tagging
=====================================================
Handles:
  1. Loading HDF5 data from the Top Quark Tagging Reference Dataset
  2. Converting 4-momenta (E, px, py, pz) → physics features (pT, η, φ, ...)
  3. Computing high-level jet features for BDT baseline
  4. Creating jet images for CNN
  5. Preparing particle-level data for ParticleNet and Transformer
  6. PyTorch Dataset classes for each representation

Physics notes:
  - pT = sqrt(px² + py²)          [transverse momentum]
  - η = arctanh(pz/|p|)           [pseudorapidity]
  - φ = atan2(py, px)             [azimuthal angle]
  - ΔR = sqrt(Δη² + Δφ²)         [angular distance]
"""

import os
import gc
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


# ═══════════════════════════════════════════════════════════════
# 1. RAW DATA LOADING
# ═══════════════════════════════════════════════════════════════

def _read_pandas_hdf_via_pytables(filepath, n_load=None):
    """
    Read a pandas-HDFStore 'table' file directly with PyTables.

    pd.read_hdf breaks on some pandas/pytables combos with
    'TypeError: a bytes-like object is required, not str' — the check
    `"table" not in pt` compares a str to a bytes attribute. Going
    through pytables avoids that code path entirely while still
    decoding block-column metadata correctly.

    Returns a dict {column_name: 1-D numpy array}.
    """
    import tables  # local import so numpy/pandas import errors surface earlier

    columns = {}
    with tables.open_file(filepath, mode="r") as h5f:
        # Locate the first Table node (pandas default is /table/table)
        table_node = None
        for node in h5f.walk_nodes("/", "Table"):
            table_node = node
            break
        if table_node is None:
            raise RuntimeError(
                f"No PyTables Table node found in {filepath}. "
                f"Is this a pandas-HDFStore 'table'-format file?"
            )

        n_total = table_node.nrows
        if n_load is None or n_load > n_total:
            n_load = n_total

        rows = table_node.read(start=0, stop=n_load)

        table_attrs = table_node._v_attrs
        parent_attrs = table_node._v_parent._v_attrs

        def _get_attr(name):
            if hasattr(table_attrs, name):
                return getattr(table_attrs, name)
            if hasattr(parent_attrs, name):
                return getattr(parent_attrs, name)
            return None

        def _normalize_names(val):
            if val is None:
                return None
            if isinstance(val, (bytes, str, np.bytes_, np.str_)):
                val = [val]
            out = []
            for item in val:
                if isinstance(item, (bytes, np.bytes_)):
                    out.append(item.decode("utf-8", errors="replace"))
                else:
                    out.append(str(item))
            return out

        for field in rows.dtype.names:
            if field == "index":
                continue

            names = _normalize_names(_get_attr(f"{field}_kind"))
            if names is None and field.startswith("values_block_"):
                names = _normalize_names(_get_attr(f"{field}_items"))
            if names is None:
                names = [field]

            block = rows[field]
            if block.ndim == 2:
                for j, nm in enumerate(names[: block.shape[1]]):
                    columns[nm] = block[:, j]
            else:
                columns[names[0]] = block

    return columns


def load_raw_data(split="train"):
    """
    Load raw data from HDF5 file.

    The dataset stores each jet as a flat row with columns:
      E_0, PX_0, PY_0, PZ_0, E_1, PX_1, ..., E_199, PX_199, PY_199, PZ_199
    Plus labels: is_signal_new (1=top, 0=QCD)

    Returns:
        constituents: (N_jets, max_constit, 4) array of (E, px, py, pz)
        labels: (N_jets,) binary labels
    """
    filepath = os.path.join(DATA_DIR, f"{split}.h5")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Run 'python download_data.py' to download the dataset."
        )

    print(f"Loading {split} data from {filepath}...")

    # Determine how many jets to load
    if USE_SUBSET:
        subset_sizes = {"train": SUBSET_TRAIN, "val": SUBSET_VAL, "test": SUBSET_TEST}
        n_load = subset_sizes.get(split, None)
    else:
        n_load = None

    # Read columns directly via pytables (avoids pandas bytes-vs-str bug)
    cols = _read_pandas_hdf_via_pytables(filepath, n_load=n_load)

    if "is_signal_new" not in cols:
        raise KeyError(
            f"'is_signal_new' column not found in {filepath}. "
            f"Available columns: {sorted(cols.keys())[:10]}..."
        )

    labels = cols["is_signal_new"].astype(np.float32)

    # Determine how many constituents are in the file
    max_constit_in_file = 0
    for i in range(200):
        if f"E_{i}" not in cols:
            break
        max_constit_in_file = i + 1

    # Use the configured max or file max, whichever is smaller
    max_c = min(MAX_CONSTITUENTS, max_constit_in_file)

    # Extract constituent 4-momenta
    n_jets = len(labels)
    constituents = np.zeros((n_jets, max_c, 4), dtype=np.float32)

    for i in range(max_c):
        constituents[:, i, 0] = cols[f"E_{i}"]
        constituents[:, i, 1] = cols[f"PX_{i}"]
        constituents[:, i, 2] = cols[f"PY_{i}"]
        constituents[:, i, 3] = cols[f"PZ_{i}"]

    del cols
    gc.collect()

    print(f"  Loaded {len(labels)} jets | Signal: {labels.sum():.0f} ({labels.mean()*100:.1f}%) | Constituents: {max_c}")
    return constituents, labels


# ═══════════════════════════════════════════════════════════════
# 2. PHYSICS FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_particle_features(constituents):
    """
    Convert (E, px, py, pz) to physics-motivated features.

    For each particle, computes:
      - pT: transverse momentum
      - eta: pseudorapidity
      - phi: azimuthal angle
      - E: energy
      - delta_eta: η relative to jet axis
      - delta_phi: φ relative to jet axis
      - log_pT: log(pT + 1e-8) for better numerical behavior

    Args:
        constituents: (N, max_c, 4) array of (E, px, py, pz)

    Returns:
        features: (N, max_c, 7) array of physics features
        mask: (N, max_c) boolean mask of real particles (pT > 0)
    """
    E = constituents[:, :, 0]
    px = constituents[:, :, 1]
    py = constituents[:, :, 2]
    pz = constituents[:, :, 3]

    # Transverse momentum
    pT = np.sqrt(px**2 + py**2)

    # Total momentum
    p = np.sqrt(px**2 + py**2 + pz**2)

    # Pseudorapidity: η = arctanh(pz/|p|)
    cos_theta = np.divide(pz, p, out=np.zeros_like(pz), where=p > 1e-8)
    cos_theta = np.clip(cos_theta, -1 + 1e-8, 1 - 1e-8)
    eta = np.arctanh(cos_theta)

    # Azimuthal angle
    phi = np.arctan2(py, px)

    # Mask for real particles (non-zero padding)
    mask = pT > 1e-8  # (N, max_c)

    # Jet axis: pT-weighted centroid in (eta, phi)
    pT_sum = np.sum(pT * mask, axis=1, keepdims=True) + 1e-8
    jet_eta = np.sum(eta * pT * mask, axis=1, keepdims=True) / pT_sum
    jet_phi = np.sum(phi * pT * mask, axis=1, keepdims=True) / pT_sum

    # Relative coordinates
    delta_eta = eta - jet_eta
    delta_phi = phi - jet_phi
    # Wrap delta_phi to [-π, π]
    delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))

    # Log pT (good for neural networks — compresses dynamic range)
    log_pT = np.log(pT + 1e-8)

    # Stack features: (N, max_c, 7)
    features = np.stack([pT, eta, phi, E, delta_eta, delta_phi, log_pT], axis=-1)

    # Zero out padded particles
    features = features * mask[:, :, np.newaxis]

    return features.astype(np.float32), mask


def compute_jet_features(constituents, features, mask):
    """
    Compute high-level jet observables for BDT baseline and physics analysis.

    These are the classic jet substructure variables used by ATLAS/CMS:
      - Jet mass, pT, eta
      - N-subjettiness ratios (τ₂₁, τ₃₂)
      - Number of constituents
      - Energy correlation functions
      - Jet width

    Args:
        constituents: (N, max_c, 4) array of (E, px, py, pz)
        features: (N, max_c, 7) array from compute_particle_features
        mask: (N, max_c) boolean mask

    Returns:
        jet_features: (N, n_features) array of high-level features
        feature_names: list of feature names
    """
    E = constituents[:, :, 0]
    px = constituents[:, :, 1]
    py = constituents[:, :, 2]
    pz = constituents[:, :, 3]
    pT = features[:, :, 0]
    delta_eta = features[:, :, 4]
    delta_phi = features[:, :, 5]

    # Jet 4-momentum (sum of constituents)
    jet_E = np.sum(E * mask, axis=1)
    jet_px = np.sum(px * mask, axis=1)
    jet_py = np.sum(py * mask, axis=1)
    jet_pz = np.sum(pz * mask, axis=1)

    jet_pT = np.sqrt(jet_px**2 + jet_py**2)
    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)

    # Jet mass: m² = E² - |p|²
    jet_m2 = jet_E**2 - jet_p**2
    jet_mass = np.sqrt(np.maximum(jet_m2, 0))

    # Jet pseudorapidity
    cos_theta = np.divide(jet_pz, jet_p, out=np.zeros_like(jet_pz), where=jet_p > 1e-8)
    cos_theta = np.clip(cos_theta, -1 + 1e-8, 1 - 1e-8)
    jet_eta = np.arctanh(cos_theta)

    # Number of constituents
    n_constituents = np.sum(mask, axis=1)

    # Jet width: pT-weighted ΔR spread
    deltaR = np.sqrt(delta_eta**2 + delta_phi**2)
    pT_total = np.sum(pT * mask, axis=1, keepdims=True) + 1e-8
    jet_width = np.sum(pT * deltaR * mask, axis=1) / pT_total.squeeze()

    # Leading constituent pT fraction
    pT_sorted = np.sort(pT * mask, axis=1)[:, ::-1]
    lead_pT_frac = pT_sorted[:, 0] / (jet_pT + 1e-8)
    sublead_pT_frac = pT_sorted[:, 1] / (jet_pT + 1e-8)

    # Simple N-subjettiness approximation (τ_N ~ spread around N axes)
    tau1 = np.sum(pT * deltaR * mask, axis=1) / (pT_total.squeeze() + 1e-8)

    # τ₂ and τ₃: use the hardest constituents as axes
    tau2 = _compute_tau_n(pT, delta_eta, delta_phi, mask, n=2)
    tau3 = _compute_tau_n(pT, delta_eta, delta_phi, mask, n=3)

    # Ratios (the discriminating variables)
    tau21 = np.divide(tau2, tau1, out=np.ones_like(tau1), where=tau1 > 1e-8)
    tau32 = np.divide(tau3, tau2, out=np.ones_like(tau2), where=tau2 > 1e-8)

    # Energy-energy correlation (C₂)
    c2 = _compute_c2(pT, delta_eta, delta_phi, mask)

    jet_feats = np.stack([
        jet_mass, jet_pT, jet_eta, n_constituents, jet_width,
        lead_pT_frac, sublead_pT_frac,
        tau1, tau2, tau3, tau21, tau32, c2
    ], axis=-1)

    feature_names = [
        "jet_mass", "jet_pT", "jet_eta", "n_constituents", "jet_width",
        "lead_pT_frac", "sublead_pT_frac",
        "tau1", "tau2", "tau3", "tau21", "tau32", "C2"
    ]

    return jet_feats.astype(np.float32), feature_names


def _compute_tau_n(pT, delta_eta, delta_phi, mask, n=2):
    """Simplified N-subjettiness using hardest particles as axes."""
    N_jets = pT.shape[0]
    tau = np.zeros(N_jets, dtype=np.float32)

    for i in range(N_jets):
        m = mask[i]
        if m.sum() < n:
            continue
        pt_i = pT[i, m]
        de_i = delta_eta[i, m]
        dp_i = delta_phi[i, m]

        # Use n hardest constituents as axes
        top_idx = np.argsort(pt_i)[-n:]
        axes_eta = de_i[top_idx]
        axes_phi = dp_i[top_idx]

        # For each particle, compute min distance to any axis
        min_dR = np.full(len(pt_i), 1e10)
        for j in range(n):
            deta = de_i - axes_eta[j]
            dphi = dp_i - axes_phi[j]
            dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
            dR = np.sqrt(deta**2 + dphi**2)
            min_dR = np.minimum(min_dR, dR)

        tau[i] = np.sum(pt_i * min_dR) / (np.sum(pt_i) + 1e-8)

    return tau


def _compute_c2(pT, delta_eta, delta_phi, mask):
    """Energy correlation function C₂ (2-point correlator ratio)."""
    N_jets = pT.shape[0]
    c2 = np.zeros(N_jets, dtype=np.float32)

    # Use only top 30 particles for speed
    max_particles = 30

    for i in range(N_jets):
        m = mask[i]
        if m.sum() < 3:
            continue
        pt_i = pT[i, m][:max_particles]
        de_i = delta_eta[i, m][:max_particles]
        dp_i = delta_phi[i, m][:max_particles]
        n_part = len(pt_i)

        pt_total = pt_i.sum() + 1e-8
        z = pt_i / pt_total  # Energy fractions

        # Pairwise ΔR
        deta_ij = de_i[:, None] - de_i[None, :]
        dphi_ij = dp_i[:, None] - dp_i[None, :]
        dphi_ij = np.arctan2(np.sin(dphi_ij), np.cos(dphi_ij))
        dR_ij = np.sqrt(deta_ij**2 + dphi_ij**2)

        # e₂ = Σ_{i<j} z_i z_j ΔR_ij
        e2 = 0.5 * np.sum(z[:, None] * z[None, :] * dR_ij)

        # e₃ approximate with leading triplet
        e3 = 0.0
        if n_part >= 3:
            top3 = np.argsort(pt_i)[-3:]
            z3 = z[top3]
            dr3 = dR_ij[np.ix_(top3, top3)]
            e3 = z3[0] * z3[1] * z3[2] * np.max(dr3)

        if e2 > 1e-10:
            c2[i] = e3 / (e2**2)

    return c2


# ═══════════════════════════════════════════════════════════════
# 3. JET IMAGE CREATION
# ═══════════════════════════════════════════════════════════════

def create_jet_images(features, mask):
    """
    Pixelate jets into 2D images in (Δη, Δφ) plane.

    Each jet becomes a 3-channel image:
      - Channel 0: pT-weighted (sum of pT in each pixel)
      - Channel 1: Multiplicity (number of particles in each pixel)
      - Channel 2: pT² weighted (highlights hardest particles)

    Args:
        features: (N, max_c, 7) particle features
        mask: (N, max_c) boolean mask

    Returns:
        images: (N, 3, IMG_SIZE, IMG_SIZE) tensor
    """
    N = features.shape[0]
    images = np.zeros((N, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    pT = features[:, :, 0]
    delta_eta = features[:, :, 4]
    delta_phi = features[:, :, 5]

    bin_edges = np.linspace(-IMG_RANGE, IMG_RANGE, IMG_SIZE + 1)

    for i in range(N):
        m = mask[i]
        if not m.any():
            continue

        pt_i = pT[i, m]
        de_i = delta_eta[i, m]
        dp_i = delta_phi[i, m]

        # Digitize: map coordinates to pixel indices
        eta_idx = np.digitize(de_i, bin_edges) - 1
        phi_idx = np.digitize(dp_i, bin_edges) - 1

        # Clip to image boundaries
        eta_idx = np.clip(eta_idx, 0, IMG_SIZE - 1)
        phi_idx = np.clip(phi_idx, 0, IMG_SIZE - 1)

        # Fill channels
        for j in range(len(pt_i)):
            ei, pi = eta_idx[j], phi_idx[j]
            images[i, 0, ei, pi] += pt_i[j]        # pT sum
            images[i, 1, ei, pi] += 1.0             # Multiplicity
            images[i, 2, ei, pi] += pt_i[j]**2      # pT² sum

    # Normalize each image independently
    for ch in range(IMG_CHANNELS):
        for i in range(N):
            max_val = images[i, ch].max()
            if max_val > 0:
                images[i, ch] /= max_val

    return images


# ═══════════════════════════════════════════════════════════════
# 4. PAIRWISE INTERACTION FEATURES (for Particle Transformer)
#    Now computed LAZILY per-batch to avoid OOM
# ═══════════════════════════════════════════════════════════════

def compute_pairwise_features_batch(features, mask):
    """
    Compute pairwise particle interaction features for a SINGLE BATCH.

    Called on-the-fly inside the ParticleDataset __getitem__ or collate_fn,
    NOT precomputed for the entire dataset (which would require ~30 GB RAM).

    For each pair (i,j), computes:
      - log(ΔR_ij)                    [angular distance]
      - log(k_T) = log(min(pT) × ΔR)  [kT clustering distance]
      - z = min(pT)/(pT_i + pT_j)     [momentum sharing]
      - Δη_ij                          [pseudorapidity difference]

    Args:
        features: (M, 7) or (B, M, 7) particle features for one jet or batch
        mask: (M,) or (B, M) boolean mask

    Returns:
        pair_feats: (M, M, 4) or (B, M, M, 4)
    """
    single = features.ndim == 2
    if single:
        features = features[np.newaxis]
        mask = mask[np.newaxis]

    N, M, _ = features.shape

    pT = features[:, :, 0]
    delta_eta = features[:, :, 4]
    delta_phi = features[:, :, 5]

    # Pairwise ΔR
    deta = delta_eta[:, :, None] - delta_eta[:, None, :]  # (N, M, M)
    dphi = delta_phi[:, :, None] - delta_phi[:, None, :]
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    dR = np.sqrt(deta**2 + dphi**2 + 1e-8)

    # kT distance
    pT_min = np.minimum(pT[:, :, None], pT[:, None, :])
    kT = pT_min * dR

    # Momentum sharing fraction
    pT_sum = pT[:, :, None] + pT[:, None, :] + 1e-8
    z = pT_min / pT_sum

    # Log transforms
    log_dR = np.log(dR + 1e-8)
    log_kT = np.log(kT + 1e-8)

    pair_feats = np.stack([log_dR, log_kT, z, deta], axis=-1)  # (N, M, M, 4)

    # Mask out padded particles
    pair_mask = mask[:, :, None] & mask[:, None, :]  # (N, M, M)
    pair_feats = pair_feats * pair_mask[:, :, :, None]

    result = pair_feats.astype(np.float32)
    if single:
        result = result[0]
    return result


# ═══════════════════════════════════════════════════════════════
# 5. PyTorch DATASETS
# ═══════════════════════════════════════════════════════════════

class JetImageDataset(Dataset):
    """Dataset for CNN: returns jet images."""

    def __init__(self, images, labels):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class ParticleDataset(Dataset):
    """
    Dataset for ParticleNet and Particle Transformer.
    Returns particle-level features, mask, and optional pairwise features.

    Pairwise features are computed ON-THE-FLY per sample (not precomputed)
    to avoid massive memory usage.
    """

    def __init__(self, features, mask, labels, max_particles=100,
                 compute_pairs=False, features_raw=None):
        M = min(max_particles, features.shape[1])
        self.features = features[:, :M].astype(np.float32)
        self.mask = mask[:, :M]
        self.labels = labels
        self.compute_pairs = compute_pairs
        # Keep raw (unnormalized) features for pairwise computation
        if compute_pairs and features_raw is not None:
            self.features_raw = features_raw[:, :M].astype(np.float32)
        else:
            self.features_raw = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feats = torch.from_numpy(self.features[idx])
        mask = torch.from_numpy(self.mask[idx].astype(np.float32))
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        item = {
            "features": feats,
            "mask": mask,
            "label": label,
        }

        if self.compute_pairs:
            # Compute pairwise features on-the-fly for this single jet
            raw = self.features_raw[idx] if self.features_raw is not None else self.features[idx]
            m = self.mask[idx]
            pair = compute_pairwise_features_batch(raw, m)
            item["pair_features"] = torch.from_numpy(pair)

        return item


class JetFeatureDataset(Dataset):
    """Dataset for BDT baseline: returns high-level jet features."""

    def __init__(self, jet_features, labels):
        self.features = jet_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════
# 6. DATA PIPELINE (main entry point) — MEMORY EFFICIENT
# ═══════════════════════════════════════════════════════════════

def prepare_all_data(max_particles=100, models_needed=None):
    """
    Complete data preparation pipeline. Memory-efficient version that
    only computes what's needed for the requested models.

    Args:
        max_particles: max constituents for ParticleNet/Transformer
        models_needed: list of model types needed, e.g. ["cnn", "bdt"]
                       If None, prepares everything.

    Returns:
        dict with all data and metadata
    """
    if models_needed is None:
        models_needed = ["bdt", "cnn", "particlenet", "transformer"]

    need_images = "cnn" in models_needed
    need_jet_feats = "bdt" in models_needed or True  # Always compute for physics plots
    need_particles = any(m in models_needed for m in ["particlenet", "transformer"])

    data = {}

    for split in ["train", "val", "test"]:
        print(f"\n{'='*50}")
        print(f"Processing {split} split...")
        print(f"{'='*50}")

        # Load raw
        constituents, labels = load_raw_data(split)

        # Compute particle features (always needed)
        features, mask = compute_particle_features(constituents)

        split_data = {
            "labels": labels,
            "features": features,
            "mask": mask,
        }

        # Compute jet-level features (for BDT and physics plots)
        if need_jet_feats:
            print("  Computing jet substructure features...")
            jet_feats, feat_names = compute_jet_features(constituents, features, mask)
            split_data["jet_features"] = jet_feats

        # Create jet images (for CNN) — ~1.2 GB for 200k jets at 40x40x3
        if need_images:
            print("  Creating jet images...")
            images = create_jet_images(features, mask)
            split_data["images"] = images

        # Store constituents only if needed (for physics analysis)
        split_data["constituents"] = constituents

        # NOTE: Pairwise features are NOT precomputed here.
        # They are computed on-the-fly in ParticleDataset.__getitem__()

        data[split] = split_data

        # Memory report
        mem_mb = sum(v.nbytes for v in split_data.values() if hasattr(v, 'nbytes')) / 1e6
        print(f"  Memory for {split}: {mem_mb:.0f} MB")

    if need_jet_feats:
        data["feature_names"] = feat_names

    # Normalize particle features using training statistics
    print("\nComputing normalization statistics from training set...")
    train_feats = data["train"]["features"]
    train_mask = data["train"]["mask"]

    # Compute mean/std only on real (non-padded) particles
    valid = train_mask.flatten().astype(bool)
    flat_feats = train_feats.reshape(-1, train_feats.shape[-1])
    feat_mean = flat_feats[valid].mean(axis=0)
    feat_std = flat_feats[valid].std(axis=0) + 1e-8

    data["norm_stats"] = {"mean": feat_mean, "std": feat_std}

    # Apply normalization
    for split in ["train", "val", "test"]:
        f = data[split]["features"]
        m = data[split]["mask"]
        f_norm = (f - feat_mean) / feat_std
        f_norm = f_norm * m[:, :, np.newaxis]  # Keep padding as zeros
        data[split]["features_norm"] = f_norm.astype(np.float32)

    # Normalize jet features
    if need_jet_feats:
        jf_mean = data["train"]["jet_features"].mean(axis=0)
        jf_std = data["train"]["jet_features"].std(axis=0) + 1e-8
        data["jet_norm"] = {"mean": jf_mean, "std": jf_std}

        for split in ["train", "val", "test"]:
            data[split]["jet_features_norm"] = (
                (data[split]["jet_features"] - jf_mean) / jf_std
            ).astype(np.float32)

    print(f"\nData preparation complete!")
    print(f"  Training jets:   {len(data['train']['labels'])}")
    print(f"  Validation jets: {len(data['val']['labels'])}")
    print(f"  Test jets:       {len(data['test']['labels'])}")

    return data


def make_dataloaders(data, model_type="particle", max_particles=100):
    """
    Create PyTorch DataLoaders for a specific model type.

    Args:
        data: dict from prepare_all_data()
        model_type: "cnn", "particlenet", or "transformer"
        max_particles: for particle-based models

    Returns:
        train_loader, val_loader, test_loader
    """
    if model_type == "cnn":
        batch_size = CNN_CONFIG["batch_size"]
        train_ds = JetImageDataset(data["train"]["images"], data["train"]["labels"])
        val_ds = JetImageDataset(data["val"]["images"], data["val"]["labels"])
        test_ds = JetImageDataset(data["test"]["images"], data["test"]["labels"])

    elif model_type == "particlenet":
        batch_size = PARTICLENET_CONFIG["batch_size"]
        train_ds = ParticleDataset(
            data["train"]["features_norm"], data["train"]["mask"],
            data["train"]["labels"], max_particles=max_particles
        )
        val_ds = ParticleDataset(
            data["val"]["features_norm"], data["val"]["mask"],
            data["val"]["labels"], max_particles=max_particles
        )
        test_ds = ParticleDataset(
            data["test"]["features_norm"], data["test"]["mask"],
            data["test"]["labels"], max_particles=max_particles
        )

    elif model_type == "transformer":
        batch_size = PARTFORMER_CONFIG["batch_size"]
        # Pairwise features computed on-the-fly using raw (unnormalized) features
        train_ds = ParticleDataset(
            data["train"]["features_norm"], data["train"]["mask"],
            data["train"]["labels"], max_particles=max_particles,
            compute_pairs=True, features_raw=data["train"]["features"]
        )
        val_ds = ParticleDataset(
            data["val"]["features_norm"], data["val"]["mask"],
            data["val"]["labels"], max_particles=max_particles,
            compute_pairs=True, features_raw=data["val"]["features"]
        )
        test_ds = ParticleDataset(
            data["test"]["features_norm"], data["test"]["mask"],
            data["test"]["labels"], max_particles=max_particles,
            compute_pairs=True, features_raw=data["test"]["features"]
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader
