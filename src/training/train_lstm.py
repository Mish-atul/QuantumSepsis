"""
QuantumSepsis Shield — LSTM Training Loop
==========================================

Complete training pipeline with:
  - Asymmetric focal loss (FN = 10× FP)
  - Early stopping on validation AUROC
  - Cosine annealing learning rate scheduler
  - Gradient clipping (max_norm = 1.0)
  - W&B experiment logging
  - Model checkpointing
  - Embedding extraction for quantum kernel
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import Config, get_default_config
from src.models.lstm import SepsisLSTM
from src.models.losses import AsymmetricFocalLoss
from src.evaluation.metrics import compute_all_metrics, format_metrics

logger = logging.getLogger(__name__)


class LSTMTrainer:
    """Trainer for the SepsisLSTM model.
    
    Handles the full training lifecycle:
        1. Model initialization
        2. Training loop with asymmetric focal loss
        3. Validation and early stopping
        4. Model checkpointing
        5. Final evaluation on test set
        6. Embedding extraction for quantum kernel
    """
    
    def __init__(self, config: Config, device: Optional[str] = None):
        self.config = config
        
        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Set random seed
        self._set_seed(config.training.seed)
        
        # Create directories
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.training.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = SepsisLSTM(config.lstm).to(self.device)
        logger.info(f"\n{self.model.summary()}")
        
        # Loss function
        self.criterion = AsymmetricFocalLoss(
            alpha_pos=config.training.focal_alpha_pos,
            alpha_neg=config.training.focal_alpha_neg,
            gamma=config.training.focal_gamma,
        )
        logger.info(f"Loss: {self.criterion}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.scheduler_t_max,
            eta_min=1e-6,
        )
        
        # Early stopping state
        self.best_val_auroc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
            "val_auprc": [],
            "learning_rate": [],
        }
        
        # W&B
        self.wandb_run = None
        if config.training.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.training.wandb_project,
                    entity=config.training.wandb_entity,
                    config={
                        "model": "SepsisLSTM",
                        "lstm_hidden_dim": config.lstm.hidden_dim,
                        "lstm_layers": config.lstm.n_layers,
                        "embedding_dim": config.lstm.embedding_dim,
                        "batch_size": config.training.batch_size,
                        "lr": config.training.learning_rate,
                        "focal_alpha_pos": config.training.focal_alpha_pos,
                        "focal_gamma": config.training.focal_gamma,
                        "seed": config.training.seed,
                    },
                    name=f"lstm_h{config.lstm.hidden_dim}_emb{config.lstm.embedding_dim}",
                )
                logger.info("W&B initialized")
            except Exception as e:
                logger.warning(f"W&B initialization failed: {e}. Continuing without.")
                self.wandb_run = None
    
    def _set_seed(self, seed: int):
        """Set all random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Run the full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
        
        Returns:
            Dictionary with best validation metrics
        """
        max_epochs = self.config.training.max_epochs
        patience = self.config.training.early_stopping_patience
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Starting LSTM Training")
        logger.info(f"  Max epochs: {max_epochs}")
        logger.info(f"  Early stopping patience: {patience}")
        logger.info(f"  Batch size: {self.config.training.batch_size}")
        logger.info(f"{'=' * 60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, max_epochs + 1):
            # Training
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self._validate(val_loader)
            val_loss = val_metrics.get("val_loss", float('inf'))
            val_auroc = val_metrics.get("val_auroc", 0.0)
            val_auprc = val_metrics.get("val_auprc", 0.0)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auroc"].append(val_auroc)
            self.history["val_auprc"].append(val_auprc)
            self.history["learning_rate"].append(current_lr)
            
            # Log
            logger.info(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUROC: {val_auroc:.4f} | "
                f"Val AUPRC: {val_auprc:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # W&B logging
            if self.wandb_run:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auroc": val_auroc,
                    "val_auprc": val_auprc,
                    "learning_rate": current_lr,
                    **{k: v for k, v in val_metrics.items() if k.startswith("val_")},
                })
            
            # Early stopping check
            if val_auroc > self.best_val_auroc:
                self.best_val_auroc = val_auroc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"  -> New best AUROC: {val_auroc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(
                        f"\nEarly stopping at epoch {epoch}. "
                        f"Best AUROC: {self.best_val_auroc:.4f} at epoch {self.best_epoch}"
                    )
                    break
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)
        
        elapsed = time.time() - start_time
        logger.info(f"\nTraining complete in {elapsed/60:.1f} minutes")
        logger.info(f"Best validation AUROC: {self.best_val_auroc:.4f} at epoch {self.best_epoch}")
        
        # Load best model
        self._load_best_checkpoint()
        
        if self.wandb_run:
            import wandb
            wandb.finish()
        
        return {
            "best_val_auroc": self.best_val_auroc,
            "best_epoch": self.best_epoch,
            "total_epochs": epoch,
            "training_time_minutes": elapsed / 60,
        }
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            output = self.model(batch_x)
            loss = self.criterion(output['logits'].squeeze(-1), batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.gradient_clip_norm,
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        all_scores = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            output = self.model(batch_x)
            loss = self.criterion(output['logits'].squeeze(-1), batch_y)
            
            all_scores.append(output['risk_score'].cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1
        
        y_true = np.concatenate(all_labels)
        y_score = np.concatenate(all_scores)
        
        metrics = compute_all_metrics(y_true, y_score, prefix="val_")
        metrics["val_loss"] = total_loss / max(n_batches, 1)
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Returns:
            Dictionary with test metrics
        """
        self.model.eval()
        
        all_scores = []
        all_labels = []
        
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            output = self.model(batch_x)
            all_scores.append(output['risk_score'].cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
        
        y_true = np.concatenate(all_labels)
        y_score = np.concatenate(all_scores)
        
        metrics = compute_all_metrics(y_true, y_score, prefix="test_")
        
        print(format_metrics(metrics, "Test Set Evaluation"))
        
        return metrics
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        data_loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 16-dim LSTM embeddings for quantum kernel.
        
        Args:
            data_loader: DataLoader to extract embeddings from
        
        Returns:
            embeddings: (N, 16) array
            labels: (N,) array
        """
        self.model.eval()
        
        all_embeddings = []
        all_labels = []
        
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device)
            
            embeddings = self.model.extract_embeddings(batch_x)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(batch_y.numpy())
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        logger.info(f"Extracted embeddings: {embeddings.shape}")
        logger.info(f"  Value range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        
        return embeddings, labels
    
    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_auroc": self.best_val_auroc,
            "metrics": metrics,
            "config": {
                "lstm": self.config.lstm.__dict__,
                "training": self.config.training.__dict__,
            },
        }
        
        if is_best:
            path = self.checkpoint_dir / "lstm_best.pt"
        else:
            path = self.checkpoint_dir / f"lstm_epoch_{epoch:03d}.pt"
        
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved to {path}")
    
    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        path = self.checkpoint_dir / "lstm_best.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        else:
            logger.warning("No best checkpoint found!")
    
    def load_checkpoint(self, path: str):
        """Load a specific checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")


def run_training(
    config_path: Optional[str] = None,
    data_path: str = "data/processed/features.h5",
) -> Dict[str, float]:
    """Run the complete LSTM training pipeline.
    
    Args:
        config_path: Path to YAML config (None = use defaults)
        data_path: Path to HDF5 features file
    
    Returns:
        Dictionary with training results
    """
    from src.data.dataset import create_dataloaders
    
    config = Config.from_yaml(config_path) if config_path else get_default_config()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(data_path, config)
    
    # Create trainer
    trainer = LSTMTrainer(config)
    
    # Train
    train_results = trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    
    # Extract embeddings for quantum kernel
    logger.info("\nExtracting embeddings for quantum kernel module...")
    train_emb, train_labels = trainer.extract_embeddings(train_loader)
    val_emb, val_labels = trainer.extract_embeddings(val_loader)
    test_emb, test_labels = trainer.extract_embeddings(test_loader)
    
    # Save embeddings
    output_dir = Path(config.data.processed_dir)
    np.savez(
        output_dir / "lstm_embeddings.npz",
        train_embeddings=train_emb,
        train_labels=train_labels,
        val_embeddings=val_emb,
        val_labels=val_labels,
        test_embeddings=test_emb,
        test_labels=test_labels,
    )
    logger.info(f"Embeddings saved to {output_dir / 'lstm_embeddings.npz'}")
    
    return {**train_results, **test_metrics}


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/lstm_training.log"),
        ],
    )
    
    parser = argparse.ArgumentParser(description="Train SepsisLSTM model")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--data", type=str, default="data/processed/features.h5",
                        help="Path to HDF5 features")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run with synthetic data for testing")
    args = parser.parse_args()
    
    if args.synthetic:
        # Create synthetic data for testing the training pipeline
        logger.info("Running with synthetic data...")
        
        config = get_default_config()
        config.training.max_epochs = 10
        config.training.batch_size = 64
        config.training.use_wandb = False
        
        np.random.seed(42)
        n_train, n_val, n_test = 2000, 400, 400
        
        X_train = np.random.randn(n_train, 6, 12).astype(np.float32)
        y_train = (np.random.random(n_train) > 0.75).astype(np.int8)
        X_val = np.random.randn(n_val, 6, 12).astype(np.float32)
        y_val = (np.random.random(n_val) > 0.75).astype(np.int8)
        X_test = np.random.randn(n_test, 6, 12).astype(np.float32)
        y_test = (np.random.random(n_test) > 0.75).astype(np.int8)
        
        from src.data.dataset import SepsisDataset
        from torch.utils.data import DataLoader
        
        train_ds = SepsisDataset(X_train, y_train)
        val_ds = SepsisDataset(X_val, y_val)
        test_ds = SepsisDataset(X_test, y_test)
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)
        test_loader = DataLoader(test_ds, batch_size=64)
        
        trainer = LSTMTrainer(config, device=args.device)
        results = trainer.train(train_loader, val_loader)
        
        test_metrics = trainer.evaluate(test_loader)
        
        embeddings, labels = trainer.extract_embeddings(test_loader)
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    else:
        results = run_training(args.config, args.data)
        print(f"\nFinal results: {results}")
