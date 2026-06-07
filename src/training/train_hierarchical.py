"""
QuantumSepsis Shield — Hierarchical LSTM Training
=================================================

Training pipeline for the hierarchical model with:
  - Multi-task learning across 3 hierarchical levels
  - Weighted loss combination
  - Spatial + temporal attention visualization
  - Enhanced interpretability

Based on FOU paper methodology (Wang et al., 2023).
"""

import os
import sys
import time
import json
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
from src.models.hierarchical_lstm import HierarchicalSepsisLSTM
from src.models.losses import AsymmetricFocalLoss
from src.evaluation.metrics import compute_all_metrics, format_metrics

logger = logging.getLogger(__name__)


class HierarchicalLoss(nn.Module):
    """Multi-task loss for hierarchical classification.
    
    Combines losses from all 3 levels with configurable weights.
    Uses asymmetric focal loss at each level to handle imbalance.
    """
    
    def __init__(
        self,
        alpha_pos: float = 0.9,
        alpha_neg: float = 0.1,
        gamma: float = 2.0,
        level_weights: Tuple[float, float, float] = (1.0, 0.5, 0.25),
    ):
        super().__init__()
        self.level_weights = level_weights
        
        # Separate focal loss for each level
        self.loss_l1 = AsymmetricFocalLoss(alpha_pos, alpha_neg, gamma)
        self.loss_l2 = AsymmetricFocalLoss(alpha_pos, alpha_neg, gamma)
        self.loss_l3 = AsymmetricFocalLoss(alpha_pos, alpha_neg, gamma)
    
    def forward(
        self,
        logits_l1: torch.Tensor,
        labels_l1: torch.Tensor,
        logits_l2: Optional[torch.Tensor] = None,
        labels_l2: Optional[torch.Tensor] = None,
        logits_l3: Optional[torch.Tensor] = None,
        labels_l3: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted hierarchical loss.
        
        Returns:
            Dictionary with 'total', 'l1', 'l2', 'l3' losses
        """
        loss_dict = {}
        
        # Level 1 (always computed)
        loss_dict['l1'] = self.loss_l1(logits_l1.squeeze(-1), labels_l1)
        total_loss = self.level_weights[0] * loss_dict['l1']
        
        # Level 2 (if provided)
        if logits_l2 is not None and labels_l2 is not None:
            loss_dict['l2'] = self.loss_l2(logits_l2.squeeze(-1), labels_l2)
            total_loss += self.level_weights[1] * loss_dict['l2']
        
        # Level 3 (if provided)
        if logits_l3 is not None and labels_l3 is not None:
            loss_dict['l3'] = self.loss_l3(logits_l3.squeeze(-1), labels_l3)
            total_loss += self.level_weights[2] * loss_dict['l3']
        
        loss_dict['total'] = total_loss
        return loss_dict


class HierarchicalTrainer:
    """Trainer for hierarchical sepsis model."""
    
    def __init__(
        self,
        config: Config,
        static_dim: int = 0,
        device: Optional[str] = None,
    ):
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
        self.model = HierarchicalSepsisLSTM(
            config.lstm,
            static_dim=static_dim,
        ).to(self.device)
        logger.info(f"\n{self.model.summary()}")
        
        # Loss function
        self.criterion = HierarchicalLoss(
            alpha_pos=config.training.focal_alpha_pos,
            alpha_neg=config.training.focal_alpha_neg,
            gamma=config.training.focal_gamma,
            level_weights=(1.0, 0.5, 0.25),  # Prioritize level 1
        )
        logger.info(f"Loss: Hierarchical with weights (1.0, 0.5, 0.25)")
        
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
            "val_auroc_l1": [],
            "val_auroc_l2": [],
            "val_auroc_l3": [],
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
                        "model": "HierarchicalSepsisLSTM",
                        "architecture": "spatial_temporal_attention",
                        "hierarchy_levels": 3,
                        "static_dim": static_dim,
                        **{k: v for k, v in config.lstm.__dict__.items()},
                        **{k: v for k, v in config.training.__dict__.items()},
                    },
                    name=f"hierarchical_h{config.lstm.hidden_dim}_static{static_dim}",
                )
                logger.info("W&B initialized")
            except Exception as e:
                logger.warning(f"W&B initialization failed: {e}")
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
        use_hierarchy: bool = True,
    ) -> Dict[str, float]:
        """Run the full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            use_hierarchy: If True, train all 3 levels; else only level 1
        
        Returns:
            Dictionary with best validation metrics
        """
        max_epochs = self.config.training.max_epochs
        patience = self.config.training.early_stopping_patience
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Starting Hierarchical LSTM Training")
        logger.info(f"  Max epochs: {max_epochs}")
        logger.info(f"  Early stopping patience: {patience}")
        logger.info(f"  Batch size: {self.config.training.batch_size}")
        logger.info(f"  Hierarchy: {'3 levels' if use_hierarchy else '1 level only'}")
        logger.info(f"{'=' * 60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, max_epochs + 1):
            # Training
            train_loss = self._train_epoch(train_loader, epoch, use_hierarchy)
            
            # Validation
            val_metrics = self._validate(val_loader, use_hierarchy)
            val_loss = val_metrics.get("val_loss", float('inf'))
            val_auroc_l1 = val_metrics.get("val_auroc_l1", 0.0)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auroc_l1"].append(val_auroc_l1)
            self.history["learning_rate"].append(current_lr)
            
            # Log
            log_str = (
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUROC L1: {val_auroc_l1:.4f}"
            )
            
            if use_hierarchy:
                val_auroc_l2 = val_metrics.get("val_auroc_l2", 0.0)
                val_auroc_l3 = val_metrics.get("val_auroc_l3", 0.0)
                log_str += f" | L2: {val_auroc_l2:.4f} | L3: {val_auroc_l3:.4f}"
                self.history["val_auroc_l2"].append(val_auroc_l2)
                self.history["val_auroc_l3"].append(val_auroc_l3)
            
            log_str += f" | LR: {current_lr:.2e}"
            logger.info(log_str)
            
            # W&B logging
            if self.wandb_run:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    **val_metrics,
                })
            
            # Early stopping check (based on level 1 AUROC)
            if val_auroc_l1 > self.best_val_auroc:
                self.best_val_auroc = val_auroc_l1
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"  -> New best AUROC: {val_auroc_l1:.4f}")
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
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        use_hierarchy: bool,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_data in train_loader:
            # Unpack batch (handle both (X, y) and (X, X_static, y) formats)
            if len(batch_data) == 2:
                batch_x, batch_y = batch_data
                batch_x_static = None
            else:
                batch_x, batch_x_static, batch_y = batch_data
                batch_x_static = batch_x_static.to(self.device)
            
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            level = 3 if use_hierarchy else 1
            output = self.model(batch_x, batch_x_static, level=level)
            
            # Compute loss
            if use_hierarchy:
                # For now, use same labels for all levels
                # In real scenario, you'd have separate labels per level
                loss_dict = self.criterion(
                    output['logits_l1'], batch_y,
                    output.get('logits_l2'), batch_y,
                    output.get('logits_l3'), batch_y,
                )
            else:
                loss_dict = self.criterion(output['logits_l1'], batch_y)
            
            loss = loss_dict['total']
            
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
    def _validate(
        self,
        val_loader: DataLoader,
        use_hierarchy: bool,
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        all_scores_l1 = []
        all_scores_l2 = []
        all_scores_l3 = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0
        
        for batch_data in val_loader:
            # Unpack batch
            if len(batch_data) == 2:
                batch_x, batch_y = batch_data
                batch_x_static = None
            else:
                batch_x, batch_x_static, batch_y = batch_data
                batch_x_static = batch_x_static.to(self.device)
            
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            level = 3 if use_hierarchy else 1
            output = self.model(batch_x, batch_x_static, level=level)
            
            # Compute loss
            if use_hierarchy:
                loss_dict = self.criterion(
                    output['logits_l1'], batch_y,
                    output.get('logits_l2'), batch_y,
                    output.get('logits_l3'), batch_y,
                )
            else:
                loss_dict = self.criterion(output['logits_l1'], batch_y)
            
            all_scores_l1.append(output['risk_score_l1'].cpu().numpy())
            if use_hierarchy:
                all_scores_l2.append(output['risk_score_l2'].cpu().numpy())
                all_scores_l3.append(output['risk_score_l3'].cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            total_loss += loss_dict['total'].item()
            n_batches += 1
        
        y_true = np.concatenate(all_labels)
        y_score_l1 = np.concatenate(all_scores_l1)
        
        metrics = compute_all_metrics(y_true, y_score_l1, prefix="val_")
        metrics["val_loss"] = total_loss / max(n_batches, 1)
        
        # Rename for clarity
        metrics["val_auroc_l1"] = metrics.pop("val_auroc")
        metrics["val_auprc_l1"] = metrics.pop("val_auprc")
        
        if use_hierarchy:
            y_score_l2 = np.concatenate(all_scores_l2)
            y_score_l3 = np.concatenate(all_scores_l3)
            
            metrics_l2 = compute_all_metrics(y_true, y_score_l2, prefix="val_l2_")
            metrics_l3 = compute_all_metrics(y_true, y_score_l3, prefix="val_l3_")
            
            metrics["val_auroc_l2"] = metrics_l2["val_l2_auroc"]
            metrics["val_auprc_l2"] = metrics_l2["val_l2_auprc"]
            metrics["val_auroc_l3"] = metrics_l3["val_l3_auroc"]
            metrics["val_auprc_l3"] = metrics_l3["val_l3_auprc"]
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        use_hierarchy: bool = True,
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        
        all_scores_l1 = []
        all_scores_l2 = []
        all_scores_l3 = []
        all_labels = []
        
        for batch_data in test_loader:
            if len(batch_data) == 2:
                batch_x, batch_y = batch_data
                batch_x_static = None
            else:
                batch_x, batch_x_static, batch_y = batch_data
                batch_x_static = batch_x_static.to(self.device)
            
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            level = 3 if use_hierarchy else 1
            output = self.model(batch_x, batch_x_static, level=level)
            
            all_scores_l1.append(output['risk_score_l1'].cpu().numpy())
            if use_hierarchy:
                all_scores_l2.append(output['risk_score_l2'].cpu().numpy())
                all_scores_l3.append(output['risk_score_l3'].cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
        
        y_true = np.concatenate(all_labels)
        y_score_l1 = np.concatenate(all_scores_l1)
        
        metrics = compute_all_metrics(y_true, y_score_l1, prefix="test_")
        metrics["test_auroc_l1"] = metrics.pop("test_auroc")
        metrics["test_auprc_l1"] = metrics.pop("test_auprc")
        
        if use_hierarchy:
            y_score_l2 = np.concatenate(all_scores_l2)
            y_score_l3 = np.concatenate(all_scores_l3)
            
            metrics_l2 = compute_all_metrics(y_true, y_score_l2, prefix="test_l2_")
            metrics_l3 = compute_all_metrics(y_true, y_score_l3, prefix="test_l3_")
            
            metrics["test_auroc_l2"] = metrics_l2["test_l2_auroc"]
            metrics["test_auprc_l2"] = metrics_l2["test_l2_auprc"]
            metrics["test_auroc_l3"] = metrics_l3["test_l3_auroc"]
            metrics["test_auprc_l3"] = metrics_l3["test_l3_auprc"]
        
        print(format_metrics(metrics, "Test Set Evaluation"))
        
        return metrics
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        data_loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 16-dim LSTM embeddings for quantum kernel."""
        self.model.eval()
        
        all_embeddings = []
        all_labels = []
        
        for batch_data in data_loader:
            if len(batch_data) == 2:
                batch_x, batch_y = batch_data
                batch_x_static = None
            else:
                batch_x, batch_x_static, batch_y = batch_data
                batch_x_static = batch_x_static.to(self.device)
            
            batch_x = batch_x.to(self.device)
            
            embeddings = self.model.extract_embeddings(batch_x, batch_x_static)
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
            path = self.checkpoint_dir / "hierarchical_lstm_best.pt"
        else:
            path = self.checkpoint_dir / f"hierarchical_lstm_epoch_{epoch:03d}.pt"
        
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved to {path}")
    
    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        path = self.checkpoint_dir / "hierarchical_lstm_best.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        else:
            logger.warning("No best checkpoint found!")


def run_hierarchical_training(
    config_path: Optional[str] = None,
    data_path: str = "data/processed/features.h5",
    static_dim: int = 0,
    use_hierarchy: bool = True,
) -> Dict[str, float]:
    """Run the complete hierarchical LSTM training pipeline.
    
    Args:
        config_path: Path to YAML config (None = use defaults)
        data_path: Path to HDF5 features file
        static_dim: Dimension of static features (0 = temporal only)
        use_hierarchy: Train all 3 levels or just level 1
    
    Returns:
        Dictionary with training results
    """
    from src.data.dataset import create_dataloaders
    
    config = Config.from_yaml(config_path) if config_path else get_default_config()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(data_path, config)
    
    # Create trainer
    trainer = HierarchicalTrainer(config, static_dim=static_dim)
    
    # Train
    train_results = trainer.train(train_loader, val_loader, use_hierarchy=use_hierarchy)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader, use_hierarchy=use_hierarchy)
    
    # Extract embeddings for quantum kernel
    logger.info("\nExtracting embeddings for quantum kernel module...")
    train_emb, train_labels = trainer.extract_embeddings(train_loader)
    val_emb, val_labels = trainer.extract_embeddings(val_loader)
    test_emb, test_labels = trainer.extract_embeddings(test_loader)
    
    # Save embeddings
    output_dir = Path(config.data.processed_dir)
    np.savez(
        output_dir / "hierarchical_lstm_embeddings.npz",
        train_embeddings=train_emb,
        train_labels=train_labels,
        val_embeddings=val_emb,
        val_labels=val_labels,
        test_embeddings=test_emb,
        test_labels=test_labels,
    )
    logger.info(f"Embeddings saved to {output_dir / 'hierarchical_lstm_embeddings.npz'}")
    
    # Save results
    results = {**train_results, **test_metrics}
    results_path = output_dir / "hierarchical_lstm_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Train Hierarchical SepsisLSTM")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--data", type=str, default="data/processed/features.h5")
    parser.add_argument("--static-dim", type=int, default=0, help="Static feature dimension")
    parser.add_argument("--no-hierarchy", action="store_true", help="Train only level 1")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    results = run_hierarchical_training(
        config_path=args.config,
        data_path=args.data,
        static_dim=args.static_dim,
        use_hierarchy=not args.no_hierarchy,
    )
    
    print(f"\n{'=' * 60}")
    print("Final Results:")
    print(json.dumps(results, indent=2))
    print(f"{'=' * 60}")
