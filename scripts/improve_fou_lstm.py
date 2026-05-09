"""
FOU LSTM Improvement Script
============================
Runs multiple experiments to improve FOU LSTM accuracy.

Experiments:
1. Baseline (current): hidden=128, layers=2, gamma=2.0
2. Larger Model: hidden=256, layers=3, attention=128
3. Focal Loss Tuning: Adjust class weights for rare classes
4. Learning Rate Schedule: Warmup + cosine annealing
5. Data Augmentation: Time-shift + noise injection
6. Ensemble: Multiple models with different seeds
7. Combined Best: Best hyperparameters from all experiments

Usage:
    python scripts/improve_fou_lstm.py --experiment all
    python scripts/improve_fou_lstm.py --experiment exp2_larger_model
"""

import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import h5py

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.lstm_fou import FouLSTM
from src.models.losses import MultiClassFocalLoss
from src.config import FouLSTMConfig, FouTrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class FouDataset(Dataset):
    """PyTorch Dataset for FOU data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AugmentedFouDataset(Dataset):
    """FOU Dataset with data augmentation."""
    
    def __init__(self, X, y, augment=True, shift_range=2, noise_std=0.05):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.shift_range = shift_range
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        
        if self.augment and torch.rand(1).item() > 0.5:
            # Time shift augmentation
            shift = torch.randint(-self.shift_range, self.shift_range + 1, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shift, dims=0)
            
            # Gaussian noise augmentation
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(x) * self.noise_std
                x = x + noise
        
        return x, y


class FouLSTMTrainer:
    """Trainer for FOU LSTM with various improvements."""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Model config
        self.lstm_config = FouLSTMConfig(
            input_size=27,
            hidden_dim=config.get('hidden_dim', 128),
            n_layers=config.get('n_layers', 2),
            dropout=config.get('dropout', 0.3),
            attention_dim=config.get('attention_dim', 64),
            embedding_dim=config.get('embedding_dim', 16),
            n_classes=4
        )
        
        # Training config
        self.batch_size = config.get('batch_size', 256)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.max_epochs = config.get('max_epochs', 100)
        self.patience = config.get('patience', 15)
        self.use_augmentation = config.get('use_augmentation', False)
        self.use_warmup = config.get('use_warmup', False)
        self.warmup_epochs = config.get('warmup_epochs', 5)
        
        # Loss config
        self.focal_alpha = config.get('focal_alpha', [0.1, 0.4, 0.3, 0.2])
        self.focal_gamma = config.get('focal_gamma', 2.0)
        
        # Initialize model
        self.model = FouLSTM(
            input_size=self.lstm_config.input_size,
            seq_len=self.lstm_config.seq_len,
            hidden_dim=self.lstm_config.hidden_dim,
            n_layers=self.lstm_config.n_layers,
            bidirectional=self.lstm_config.bidirectional,
            dropout=self.lstm_config.dropout,
            attention_dim=self.lstm_config.attention_dim,
            fc1_dim=self.lstm_config.fc1_dim,
            embedding_dim=self.lstm_config.embedding_dim,
            n_classes=self.lstm_config.n_classes
        ).to(self.device)
        
        # Initialize loss
        self.criterion = MultiClassFocalLoss(
            alpha=torch.tensor(self.focal_alpha),
            gamma=self.focal_gamma,
            reduction='mean'
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get('weight_decay', 0.0001)
        )
        
        # Initialize scheduler
        if self.use_warmup:
            self.scheduler = self._get_warmup_scheduler()
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epochs,
                eta_min=1e-6
            )
        
        # Training state
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        logger.info(f"Initialized FOU LSTM Trainer")
        logger.info(f"  Model: hidden={self.lstm_config.hidden_dim}, layers={self.lstm_config.n_layers}")
        logger.info(f"  Loss: focal_alpha={self.focal_alpha}, gamma={self.focal_gamma}")
        logger.info(f"  Training: lr={self.learning_rate}, batch_size={self.batch_size}")
        logger.info(f"  Augmentation: {self.use_augmentation}")
        logger.info(f"  Device: {self.device}")
    
    def _get_warmup_scheduler(self):
        """Get learning rate scheduler with warmup."""
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                # Cosine annealing after warmup
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self, train_loader, val_loader) -> Dict:
        """Train the model."""
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            # Training
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_f1, val_per_class_f1 = self._validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            logger.info(f"  Per-class F1: {val_per_class_f1}")
            
            # Early stopping
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                # Save best model
                self._save_checkpoint('best')
                logger.info(f"  ✓ New best F1: {val_f1:.4f}")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        training_time = time.time() - start_time
        
        # Load best model
        self._load_checkpoint('best')
        
        results = {
            'best_val_f1': float(self.best_val_f1),
            'best_epoch': int(self.best_epoch),
            'total_epochs': epoch + 1,
            'training_time_minutes': training_time / 60
        }
        
        logger.info(f"Training complete in {training_time/60:.2f} minutes")
        logger.info(f"Best val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        
        return results
    
    def _train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, probs, _, _ = self.model(X)
            
            # Compute loss
            loss = self.criterion(logits, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * X.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(self, val_loader) -> Tuple[float, float, float, Dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # Forward pass
                logits, probs, _, _ = self.model(X)
                
                # Compute loss
                loss = self.criterion(logits, y)
                
                # Metrics
                total_loss += loss.item() * X.size(0)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        # Compute F1 scores
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        per_class_f1_dict = {
            f'class_{i}_f1': float(f1) for i, f1 in enumerate(per_class_f1)
        }
        
        return avg_loss, accuracy, macro_f1, per_class_f1_dict
    
    def evaluate(self, test_loader) -> Dict:
        """Evaluate on test set."""
        logger.info("Evaluating on test set...")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                
                logits, probs, _, _ = self.model(X)
                _, predicted = torch.max(probs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.numpy())
                all_probs.append(probs.cpu().numpy())
        
        all_probs = np.vstack(all_probs)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        
        results = {
            'test_accuracy': float(accuracy),
            'test_macro_f1': float(macro_f1),
            'test_weighted_f1': float(weighted_f1),
            'test_class_0_f1': float(per_class_f1[0]),
            'test_class_1_f1': float(per_class_f1[1]),
            'test_class_2_f1': float(per_class_f1[2]),
            'test_class_3_f1': float(per_class_f1[3]),
        }
        
        logger.info(f"Test Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Macro F1: {macro_f1:.4f}")
        logger.info(f"  Weighted F1: {weighted_f1:.4f}")
        logger.info(f"  Per-class F1: {per_class_f1}")
        
        return results
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = Path('/media/rvcse22/CSERV/QuantumSepsis_checkpoints/fou/experiments')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'{name}.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
            'config': self.config
        }, checkpoint_path)
    
    def _load_checkpoint(self, name: str):
        """Load model checkpoint."""
        checkpoint_path = Path('checkpoints/fou/experiments') / f'{name}.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint: {checkpoint_path}")


# Experiment configurations
EXPERIMENTS = {
    'exp1_baseline': {
        'name': 'Baseline',
        'description': 'Current configuration (hidden=128, layers=2)',
        'config': {
            'hidden_dim': 128,
            'n_layers': 2,
            'attention_dim': 64,
            'focal_alpha': [0.1, 0.4, 0.3, 0.2],
            'focal_gamma': 2.0,
            'learning_rate': 0.001,
            'batch_size': 256,
            'max_epochs': 100,
            'patience': 15,
        }
    },
    'exp2_larger_model': {
        'name': 'Larger Model',
        'description': 'Increase capacity (hidden=256, layers=3, attention=128)',
        'config': {
            'hidden_dim': 256,
            'n_layers': 3,
            'attention_dim': 128,
            'focal_alpha': [0.1, 0.4, 0.3, 0.2],
            'focal_gamma': 2.0,
            'learning_rate': 0.001,
            'batch_size': 256,
            'max_epochs': 100,
            'patience': 15,
        }
    },
    'exp3_focal_tuning': {
        'name': 'Focal Loss Tuning',
        'description': 'Boost rare classes (Class 2 weight increased)',
        'config': {
            'hidden_dim': 128,
            'n_layers': 2,
            'attention_dim': 64,
            'focal_alpha': [0.05, 0.35, 0.45, 0.15],  # Boost Class 2
            'focal_gamma': 2.5,  # Stronger focusing
            'learning_rate': 0.001,
            'batch_size': 256,
            'max_epochs': 100,
            'patience': 15,
        }
    },
    'exp4_warmup_schedule': {
        'name': 'Warmup + Cosine Schedule',
        'description': 'Learning rate warmup for stable training',
        'config': {
            'hidden_dim': 128,
            'n_layers': 2,
            'attention_dim': 64,
            'focal_alpha': [0.1, 0.4, 0.3, 0.2],
            'focal_gamma': 2.0,
            'learning_rate': 0.002,  # Higher initial LR
            'batch_size': 256,
            'max_epochs': 100,
            'patience': 15,
            'use_warmup': True,
            'warmup_epochs': 5,
        }
    },
    'exp5_augmentation': {
        'name': 'Data Augmentation',
        'description': 'Time-shift + noise augmentation',
        'config': {
            'hidden_dim': 128,
            'n_layers': 2,
            'attention_dim': 64,
            'focal_alpha': [0.1, 0.4, 0.3, 0.2],
            'focal_gamma': 2.0,
            'learning_rate': 0.001,
            'batch_size': 256,
            'max_epochs': 100,
            'patience': 15,
            'use_augmentation': True,
        }
    },
    'exp6_combined_best': {
        'name': 'Combined Best',
        'description': 'Best hyperparameters from all experiments',
        'config': {
            'hidden_dim': 256,
            'n_layers': 3,
            'attention_dim': 128,
            'focal_alpha': [0.05, 0.35, 0.45, 0.15],
            'focal_gamma': 2.5,
            'learning_rate': 0.002,
            'batch_size': 256,
            'max_epochs': 120,
            'patience': 20,
            'use_warmup': True,
            'warmup_epochs': 5,
            'use_augmentation': True,
        }
    },
}


def load_data(data_path: str, use_augmentation: bool = False):
    """Load FOU data from HDF5."""
    logger.info(f"Loading data from {data_path}...")
    
    with h5py.File(data_path, 'r') as f:
        X_train = f['X_train'][:]
        y_train = f['y_train'][:]
        X_val = f['X_val'][:]
        y_val = f['y_val'][:]
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Train labels: {np.bincount(y_train)}")
    logger.info(f"Val labels: {np.bincount(y_val)}")
    logger.info(f"Test labels: {np.bincount(y_test)}")
    
    # Create datasets
    if use_augmentation:
        train_dataset = AugmentedFouDataset(X_train, y_train, augment=True)
    else:
        train_dataset = FouDataset(X_train, y_train)
    
    val_dataset = FouDataset(X_val, y_val)
    test_dataset = FouDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader


def run_experiment(exp_name: str, data_path: str, device: str = 'cuda') -> Dict:
    """Run a single experiment."""
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_name}")
    
    exp = EXPERIMENTS[exp_name]
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {exp['name']}")
    logger.info(f"  {exp['description']}")
    logger.info(f"{'='*60}\n")
    
    # Load data
    use_augmentation = exp['config'].get('use_augmentation', False)
    train_loader, val_loader, test_loader = load_data(data_path, use_augmentation)
    
    # Initialize trainer
    trainer = FouLSTMTrainer(exp['config'], device=device)
    
    # Train
    train_results = trainer.train(train_loader, val_loader)
    
    # Evaluate
    test_results = trainer.evaluate(test_loader)
    
    # Combine results
    results = {
        'experiment': exp_name,
        'name': exp['name'],
        'description': exp['description'],
        'config': exp['config'],
        **train_results,
        **test_results
    }
    
    return results


def run_all_experiments(data_path: str, device: str = 'cuda') -> Dict:
    """Run all experiments."""
    all_results = {}
    
    for exp_name in EXPERIMENTS:
        try:
            results = run_experiment(exp_name, data_path, device)
            all_results[exp_name] = results
            
            # Save intermediate results
            output_path = Path('data/processed/fou/lstm_improvement_results.json')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            logger.info(f"✓ {exp_name} complete")
            
        except Exception as e:
            logger.error(f"✗ {exp_name} failed: {e}")
            all_results[exp_name] = {'error': str(e)}
    
    # Print comparison table
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT COMPARISON")
    logger.info(f"{'='*80}")
    logger.info(f"{'Experiment':<30} {'Val F1':<10} {'Test F1':<10} {'Test Acc':<10}")
    logger.info(f"{'-'*80}")
    
    for exp_name, results in all_results.items():
        if 'error' in results:
            logger.info(f"{exp_name:<30} FAILED: {results['error']}")
        else:
            val_f1 = results.get('best_val_f1', 0)
            test_f1 = results.get('test_macro_f1', 0)
            test_acc = results.get('test_accuracy', 0)
            logger.info(f"{exp_name:<30} {val_f1:<10.4f} {test_f1:<10.4f} {test_acc:<10.4f}")
    
    logger.info(f"{'='*80}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='FOU LSTM Improvement Experiments')
    parser.add_argument('--experiment', type=str, default='all', 
                       help='Experiment to run (all, exp1_baseline, exp2_larger_model, etc.)')
    parser.add_argument('--data', type=str, default='data/processed/fou/fou_features.h5',
                       help='Path to FOU features HDF5 file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    if args.experiment == 'all':
        results = run_all_experiments(args.data, args.device)
    else:
        results = run_experiment(args.experiment, args.data, args.device)
        results = {args.experiment: results}
    
    # Save final results
    output_path = Path('data/processed/fou/lstm_improvement_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()
