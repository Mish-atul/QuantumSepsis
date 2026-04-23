"""
QuantumSepsis Shield — PyTorch Dataset
=======================================

Wraps HDF5 windowed data as PyTorch Dataset for efficient training.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import Config, get_default_config

logger = logging.getLogger(__name__)


class SepsisDataset(Dataset):
    """PyTorch Dataset for sepsis prediction windows.
    
    Loads data from HDF5 or numpy arrays.
    
    Each sample:
        X: (6, 12) float32 — 6-hour window of 12 features
        y: scalar int — binary sepsis label
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[callable] = None,
    ):
        """Initialize dataset.
        
        Args:
            X: (N, 6, 12) feature array
            y: (N,) label array
            transform: Optional transform function applied to X
        """
        assert X.ndim == 3, f"Expected 3D array, got {X.ndim}D"
        assert len(X) == len(y), f"X and y length mismatch: {len(X)} vs {len(y)}"
        
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.astype(np.float32))
        self.transform = transform
        
        self.n_samples = len(X)
        self.n_positive = int(y.sum())
        self.n_negative = self.n_samples - self.n_positive
        self.positive_ratio = self.n_positive / self.n_samples if self.n_samples > 0 else 0
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    @classmethod
    def from_hdf5(cls, path: str, split: str = "train", **kwargs) -> "SepsisDataset":
        """Load dataset from HDF5 file.
        
        Args:
            path: Path to HDF5 file
            split: "train", "val", or "test"
            **kwargs: Additional arguments to constructor
        """
        with h5py.File(path, 'r') as f:
            X = f[f"X_{split}"][:]
            y = f[f"y_{split}"][:]
        
        return cls(X, y, **kwargs)
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights."""
        if self.n_positive == 0 or self.n_negative == 0:
            return torch.FloatTensor([1.0])
        
        pos_weight = self.n_negative / self.n_positive
        return torch.FloatTensor([pos_weight])
    
    def summary(self) -> str:
        """Return a summary string."""
        return (
            f"SepsisDataset: {self.n_samples:,} samples "
            f"({self.n_positive:,} positive, {self.n_negative:,} negative, "
            f"ratio={self.positive_ratio:.2%})"
        )


class GaussianNoise:
    """Add Gaussian noise for data augmentation during training."""
    
    def __init__(self, std: float = 0.01):
        self.std = std
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


def create_dataloaders(
    data_path: str,
    config: Config,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from HDF5 file.
    
    Args:
        data_path: Path to features.h5
        config: Configuration
        augment_train: Whether to add noise augmentation to training
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    transform = GaussianNoise(std=0.01) if augment_train else None
    
    train_ds = SepsisDataset.from_hdf5(data_path, "train", transform=transform)
    val_ds = SepsisDataset.from_hdf5(data_path, "val")
    test_ds = SepsisDataset.from_hdf5(data_path, "test")
    
    logger.info(f"Train: {train_ds.summary()}")
    logger.info(f"Val:   {val_ds.summary()}")
    logger.info(f"Test:  {test_ds.summary()}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 6, 12).astype(np.float32)
    y = (np.random.random(1000) > 0.75).astype(np.int8)
    
    dataset = SepsisDataset(X, y)
    print(dataset.summary())
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"Batch X: {batch_x.shape}, Batch y: {batch_y.shape}")
    print(f"Class weights: {dataset.get_class_weights()}")
