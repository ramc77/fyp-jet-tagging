"""
Unified Training Framework
===========================
Handles training, validation, and model checkpointing for all architectures.
Supports: JetCNN, ParticleNet, ParticleTransformer, and BDT baseline.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, MODEL_DIR, RESULTS_DIR


class Trainer:
    """
    Unified trainer for all jet classification models.

    Features:
      - Automatic mixed precision (AMP) for faster training on GPU
      - Learning rate scheduling with ReduceLROnPlateau
      - Early stopping based on validation AUC
      - Model checkpointing (saves best model)
      - Training history logging
    """

    def __init__(self, model, model_name, config, model_type="cnn"):
        """
        Args:
            model: PyTorch model
            model_name: string identifier (e.g., "jet_cnn", "particlenet")
            config: hyperparameter dict from config.py
            model_type: "cnn", "particlenet", or "transformer"
        """
        self.model = model.to(DEVICE)
        self.model_name = model_name
        self.config = config
        self.model_type = model_type

        # Loss function: Binary Cross-Entropy with Logits
        # (numerically more stable than sigmoid + BCE)
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        OptClass = AdamW if model_type in ("particlenet", "transformer") else Adam
        self.optimizer = OptClass(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5,
            patience=config["scheduler_patience"],
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
        self.use_amp = DEVICE.type == "cuda"

        # Training state
        self.best_auc = 0.0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}

        # Paths
        self.checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}_best.pt")
        # "Last" checkpoint captures the full training state so we can
        # resume mid-training after a Ctrl-C / laptop sleep / SIGTERM.
        self.last_path = os.path.join(MODEL_DIR, f"{model_name}_last.pt")
        # A marker file written when training has finished (either max
        # epochs reached or early stopping fired). If it exists, callers
        # can safely skip retraining.
        self.done_path = os.path.join(MODEL_DIR, f"{model_name}_done.json")
        self.history_path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"Trainer initialized: {model_name}")
        print(f"  Parameters: {n_params:,}")
        print(f"  Device: {DEVICE}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Optimizer: {OptClass.__name__} (lr={config['lr']})")
        print(f"{'='*60}")

    def _forward_batch(self, batch):
        """Route batch through model based on model_type."""
        if self.model_type == "cnn":
            images, labels = batch
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = self.model(images)
            return logits, labels

        elif self.model_type == "particlenet":
            features = batch["features"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = self.model(features, mask)
            return logits, labels

        elif self.model_type == "transformer":
            features = batch["features"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            pair_feats = batch.get("pair_features")
            if pair_feats is not None:
                pair_feats = pair_feats.to(DEVICE)
            logits, _ = self.model(features, mask, pair_feats)
            return logits, labels

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def train_epoch(self, train_loader):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits, labels = self._forward_batch(batch)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, labels = self._forward_batch(batch)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self, val_loader):
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        n_batches = 0

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits, labels = self._forward_batch(batch)
                    loss = self.criterion(logits, labels)
            else:
                logits, labels = self._forward_batch(batch)
                loss = self.criterion(logits, labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        auc = roc_auc_score(all_labels, all_probs)
        avg_loss = total_loss / n_batches

        return avg_loss, auc, all_probs, all_labels

    def _save_last(self, epoch):
        """Snapshot full training state so the next process can resume."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "epoch": epoch,
            "best_auc": self.best_auc,
            "patience_counter": self.patience_counter,
            "history": self.history,
            "config": self.config,
        }, self.last_path)

    def _try_resume(self):
        """If a last-checkpoint exists, restore state and return its epoch
        number; otherwise return 0 (start from epoch 1)."""
        if not os.path.exists(self.last_path):
            return 0
        try:
            ck = torch.load(self.last_path, map_location=DEVICE, weights_only=False)
        except Exception as e:
            print(f"  [resume] could not read {self.last_path}: {e}")
            return 0
        self.model.load_state_dict(ck["model_state_dict"])
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck and ck["scheduler_state_dict"] is not None:
            try:
                self.scheduler.load_state_dict(ck["scheduler_state_dict"])
            except Exception:
                pass
        if self.scaler is not None and ck.get("scaler_state_dict") is not None:
            try:
                self.scaler.load_state_dict(ck["scaler_state_dict"])
            except Exception:
                pass
        self.best_auc = float(ck.get("best_auc", 0.0))
        self.patience_counter = int(ck.get("patience_counter", 0))
        self.history = ck.get("history", self.history)
        last_epoch = int(ck.get("epoch", 0))
        print(f"  [resume] continuing from epoch {last_epoch} "
              f"(best AUC so far {self.best_auc:.5f})")
        return last_epoch

    def is_done(self):
        """True if training for this model has previously finished
        (either reached max epochs or triggered early stopping)."""
        return os.path.exists(self.done_path)

    def _mark_done(self, status, epoch):
        with open(self.done_path, "w") as f:
            json.dump({
                "status": status, "stopped_at_epoch": epoch,
                "best_auc": self.best_auc, "model_name": self.model_name,
            }, f, indent=2)

    def train(self, train_loader, val_loader):
        """
        Full training loop with early stopping.  Resumable: on restart,
        picks up from the per-epoch ``_last.pt`` checkpoint and continues.

        Returns:
            history: dict with training metrics per epoch
        """
        epochs = self.config["epochs"]
        patience = self.config["early_stop_patience"]

        # Resume if possible
        start_epoch = self._try_resume() + 1
        if start_epoch > epochs:
            print(f"\n[resume] training already past nominal epoch cap ({start_epoch} > {epochs}); marking done.")
            self._mark_done("max-epochs", start_epoch - 1)
            return self.history

        print(f"\nStarting training for up to {epochs} epochs "
              f"(starting at epoch {start_epoch})...")
        print(f"Early stopping patience: {patience}")
        start_time = time.time()

        last_epoch = start_epoch - 1
        stop_status = "max-epochs"
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()
            last_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_auc, _, _ = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_auc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.5f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Best-checkpoint
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_auc": self.best_auc,
                    "config": self.config,
                }, self.checkpoint_path)
                print(f"  → New best AUC: {val_auc:.5f} (model saved)")
            else:
                self.patience_counter += 1

            # Always save the per-epoch "last" checkpoint AND the
            # incremental history JSON, so a Ctrl-C never loses the
            # epoch's progress.
            self._save_last(epoch)
            with open(self.history_path, "w") as f:
                json.dump(self.history, f, indent=2)

            if self.patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                stop_status = "early-stop"
                break

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Best validation AUC: {self.best_auc:.5f}")

        # Mark this model as fully done so the orchestrator can skip it
        # on the next invocation.
        self._mark_done(stop_status, last_epoch)
        return self.history

    def load_best_model(self):
        """Load the best saved model checkpoint."""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from epoch {checkpoint['epoch']} (AUC: {checkpoint['best_auc']:.5f})")
        else:
            print("No checkpoint found, using current model weights.")

    @torch.no_grad()
    def predict(self, data_loader):
        """Get predictions for a dataset."""
        self.model.eval()
        all_probs = []
        all_labels = []

        for batch in tqdm(data_loader, desc="Predicting", leave=False):
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits, labels = self._forward_batch(batch)
            else:
                logits, labels = self._forward_batch(batch)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

        return np.concatenate(all_probs), np.concatenate(all_labels)

    @torch.no_grad()
    def get_attention_maps(self, data_loader, n_samples=10):
        """
        Extract attention maps from Particle Transformer.
        Useful for interpretability analysis.
        """
        if self.model_type != "transformer":
            raise ValueError("Attention maps only available for transformer models")

        self.model.eval()

        # Collect a CLASS-BALANCED sample so the per-class attention plots
        # (e.g. Top vs QCD panels) are never empty. The test loader is not
        # shuffled, so naively taking the first n_samples can yield jets of
        # a single class.
        per_class = max(1, n_samples // 2)
        buckets = {0: [], 1: []}  # label -> list of (attn_list, features)

        for batch in data_loader:
            features = batch["features"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            labels = batch["label"]
            pair_feats = batch.get("pair_features")
            if pair_feats is not None:
                pair_feats = pair_feats.to(DEVICE)

            _, attn_maps = self.model(features, mask, pair_feats)

            for i in range(len(labels)):
                lab = int(round(labels[i].item()))
                if lab not in buckets or len(buckets[lab]) >= per_class:
                    continue
                buckets[lab].append((
                    [a[i].cpu().numpy() for a in attn_maps],
                    features[i].cpu().numpy(),
                ))

            if len(buckets[1]) >= per_class and len(buckets[0]) >= per_class:
                break

        # Emit top jets first, then QCD, keeping labels aligned.
        all_attn, all_features, all_labels = [], [], []
        for lab in (1, 0):
            for attn, feats in buckets[lab]:
                all_attn.append(attn)
                all_features.append(feats)
                all_labels.append(lab)

        return all_attn, all_features, all_labels


def train_bdt_baseline(train_features, train_labels, val_features, val_labels,
                       test_features, test_labels, feature_names):
    """
    Train XGBoost BDT as a baseline comparison.

    BDTs on high-level jet features represent the "traditional" approach
    used at ATLAS/CMS before deep learning. They serve as an important
    baseline: if a DL model can't beat a well-tuned BDT on the same
    features, the additional complexity isn't justified.
    """
    import xgboost as xgb
    from config import BDT_CONFIG

    print("\n" + "=" * 60)
    print("Training XGBoost BDT Baseline")
    print("=" * 60)
    print(f"Features: {feature_names}")
    print(f"Training samples: {len(train_labels)}")

    # On macOS, XGBoost's bundled libomp can collide with the libomp loaded by
    # NumPy/OpenBLAS/torch, producing a segfault during xgb.train. Pin XGBoost
    # to a single thread (training 50k × 13 is sub-second anyway) so the other
    # steps can still parallelise via torch. Allow override via BDT_CONFIG or
    # the XGBOOST_NUM_THREADS env var.
    nthread = int(os.environ.get("XGBOOST_NUM_THREADS", BDT_CONFIG.get("nthread", 1)))

    # Ensure features are contiguous float32 (defensive: non-contiguous views
    # have caused crashes in older XGBoost builds)
    train_features = np.ascontiguousarray(train_features, dtype=np.float32)
    val_features = np.ascontiguousarray(val_features, dtype=np.float32)
    test_features = np.ascontiguousarray(test_features, dtype=np.float32)

    dtrain = xgb.DMatrix(train_features, label=train_labels, feature_names=feature_names, nthread=nthread)
    dval = xgb.DMatrix(val_features, label=val_labels, feature_names=feature_names, nthread=nthread)
    dtest = xgb.DMatrix(test_features, label=test_labels, feature_names=feature_names, nthread=nthread)

    params = {
        "objective": "binary:logistic",
        "max_depth": BDT_CONFIG["max_depth"],
        "learning_rate": BDT_CONFIG["learning_rate"],
        "subsample": BDT_CONFIG["subsample"],
        "colsample_bytree": BDT_CONFIG["colsample_bytree"],
        "eval_metric": "auc",
        "tree_method": "hist",
        "nthread": nthread,
        "seed": 42,
    }

    # Train with early stopping
    evals = [(dtrain, "train"), (dval, "val")]
    bst = xgb.train(
        params, dtrain,
        num_boost_round=BDT_CONFIG["n_estimators"],
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=50,
    )

    # Predictions
    train_probs = bst.predict(dtrain)
    val_probs = bst.predict(dval)
    test_probs = bst.predict(dtest)

    # Metrics
    train_auc = roc_auc_score(train_labels, train_probs)
    val_auc = roc_auc_score(val_labels, val_probs)
    test_auc = roc_auc_score(test_labels, test_probs)

    print(f"\nBDT Results:")
    print(f"  Train AUC: {train_auc:.5f}")
    print(f"  Val AUC:   {val_auc:.5f}")
    print(f"  Test AUC:  {test_auc:.5f}")

    # Feature importance
    importance = bst.get_score(importance_type="gain")

    # Save model
    bst.save_model(os.path.join(MODEL_DIR, "bdt_baseline.json"))

    return {
        "model": bst,
        "test_probs": test_probs,
        "test_labels": test_labels,
        "val_probs": val_probs,
        "val_labels": val_labels,
        "test_auc": test_auc,
        "importance": importance,
        "feature_names": feature_names,
    }
