"""
QuantumSepsis Shield — 6-Hour Sliding Window Generation
========================================================

Converts hourly feature DataFrames into sliding window tensors:
  Input:  (stay_id, hour, 12 features) DataFrame
  Output: (N_windows, 6, 12) tensor + labels + metadata

Window strategy:
  - For sepsis patients: windows ending 3-4 hours before onset → label=1
  - For non-sepsis patients: random windows → label=0
  - Stride: 1 hour for maximum data utilization
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import Config, get_default_config

logger = logging.getLogger(__name__)


class WindowGenerator:
    """Generates 6-hour sliding window tensors from hourly features."""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.data.processed_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.window_size = config.data.window_size_hours  # 6
        self.stride = config.data.window_stride_hours     # 1
        self.prediction_horizon = config.data.prediction_horizon_hours  # 4
        self.feature_names = config.data.feature_names
        self.n_features = config.data.n_features  # 12
    
    def generate_windows(
        self,
        features: pd.DataFrame,
        cohort: pd.DataFrame,
        split_name: str = "train",
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Generate sliding window tensors for a data split.
        
        Args:
            features: Preprocessed hourly features (stay_id, hour, 12 features)
            cohort: Cohort with sepsis_label, sepsis_onset_time, intime
            split_name: "train", "val", or "test"
        
        Returns:
            X: (N, 6, 12) float32 tensor
            y: (N,) int8 label array
            metadata: DataFrame with window metadata
        """
        logger.info(f"Generating {split_name} windows...")
        
        # Get unique stays in this split
        stay_ids = features["stay_id"].unique()
        
        all_X = []
        all_y = []
        all_meta = []
        
        for stay_id in tqdm(stay_ids, desc=f"  {split_name} windows"):
            stay_features = features[features["stay_id"] == stay_id].sort_values("hour")
            
            # Get cohort info for this stay
            stay_info = cohort[cohort["stay_id"] == stay_id]
            if len(stay_info) == 0:
                continue
            stay_info = stay_info.iloc[0]
            
            sepsis_label = int(stay_info["sepsis_label"])
            intime = stay_info["intime"]
            
            # Compute onset hour (relative to intime) — only for sepsis patients
            onset_hour = None
            if sepsis_label == 1 and pd.notna(stay_info.get("sepsis_onset_time")):
                onset_time = pd.to_datetime(stay_info["sepsis_onset_time"])
                onset_hour = (onset_time - pd.to_datetime(intime)).total_seconds() / 3600
            
            # Generate windows for this stay
            windows, labels, meta = self._generate_stay_windows(
                stay_features, stay_id, sepsis_label, onset_hour
            )
            
            if windows is not None:
                all_X.append(windows)
                all_y.append(labels)
                all_meta.append(meta)
        
        if not all_X:
            raise ValueError(f"No windows generated for {split_name}!")
        
        X = np.concatenate(all_X, axis=0).astype(np.float32)
        y = np.concatenate(all_y, axis=0).astype(np.int8)
        metadata = pd.concat(all_meta, ignore_index=True)
        
        logger.info(f"  {split_name} windows: X={X.shape}, y={y.shape}")
        logger.info(f"  Positive windows: {y.sum():,} ({y.mean():.1%})")
        logger.info(f"  Negative windows: {(1-y).sum():,} ({(1-y).mean():.1%})")
        
        return X, y, metadata
    
    def _generate_stay_windows(
        self,
        stay_features: pd.DataFrame,
        stay_id: int,
        sepsis_label: int,
        onset_hour: Optional[float],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Generate sliding windows for a single stay.
        
        Labeling strategy:
        - Sepsis patients: windows ending 3-4 hours before onset → label=1
                           windows ending >6 hours before onset → label=0
        - Non-sepsis patients: all windows → label=0
        
        Returns:
            windows: (n_windows, 6, 12) array
            labels: (n_windows,) array
            metadata: DataFrame with stay_id, window_end_hour, hours_before_onset
        """
        max_hour = int(stay_features["hour"].max())
        min_hour = int(stay_features["hour"].min())
        
        if max_hour - min_hour < self.window_size:
            return None, None, None
        
        # Create feature matrix for this stay
        hours = stay_features["hour"].values
        feature_matrix = stay_features[self.feature_names].values  # (n_hours, 12)
        
        # Create a dictionary for fast hour-to-features lookup
        hour_to_idx = {h: i for i, h in enumerate(hours)}
        
        windows = []
        labels = []
        meta_rows = []
        
        # Slide over all possible windows
        for window_end in range(min_hour + self.window_size - 1, max_hour + 1, self.stride):
            window_start = window_end - self.window_size + 1
            
            # Check that all hours in window exist
            window_hours = list(range(window_start, window_end + 1))
            
            # Extract window features
            window_data = np.full((self.window_size, self.n_features), np.nan)
            valid_hours = 0
            
            for i, h in enumerate(window_hours):
                if h in hour_to_idx:
                    window_data[i] = feature_matrix[hour_to_idx[h]]
                    valid_hours += 1
            
            # Require at least 50% valid hours
            if valid_hours < self.window_size // 2:
                continue
            
            # Fill remaining NaNs with column median
            for col in range(self.n_features):
                col_data = window_data[:, col]
                nan_mask = np.isnan(col_data)
                if nan_mask.any() and not nan_mask.all():
                    window_data[nan_mask, col] = np.nanmedian(col_data)
                elif nan_mask.all():
                    window_data[:, col] = 0.0  # All missing → zero (already normalized)
            
            # Determine label
            hours_before_onset = np.nan
            if sepsis_label == 1 and onset_hour is not None:
                hours_before_onset = onset_hour - window_end
                
                if 0 < hours_before_onset <= self.prediction_horizon + 2:
                    # Window is in the prediction zone (1-6 hours before onset)
                    label = 1
                elif hours_before_onset > self.prediction_horizon + 6:
                    # Window is too early (> 10 hours before onset)
                    label = 0
                elif hours_before_onset <= 0:
                    # Window is after onset — skip (data leakage risk)
                    continue
                else:
                    # Ambiguous zone — skip
                    continue
            else:
                # Non-sepsis patient
                label = 0
            
            windows.append(window_data)
            labels.append(label)
            meta_rows.append({
                "stay_id": stay_id,
                "window_end_hour": window_end,
                "hours_before_onset": hours_before_onset,
            })
        
        if not windows:
            return None, None, None
        
        return (
            np.array(windows),
            np.array(labels),
            pd.DataFrame(meta_rows),
        )
    
    def save_to_hdf5(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        meta_train: pd.DataFrame, meta_val: pd.DataFrame, meta_test: pd.DataFrame,
    ) -> str:
        """Save all windowed data to HDF5 file.
        
        Returns:
            Path to saved HDF5 file
        """
        output_path = self.output_dir / "features.h5"
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset("X_train", data=X_train, compression="gzip", compression_opts=4)
            f.create_dataset("y_train", data=y_train, compression="gzip")
            f.create_dataset("X_val", data=X_val, compression="gzip", compression_opts=4)
            f.create_dataset("y_val", data=y_val, compression="gzip")
            f.create_dataset("X_test", data=X_test, compression="gzip", compression_opts=4)
            f.create_dataset("y_test", data=y_test, compression="gzip")
            
            # Store metadata as attributes
            f.attrs["window_size"] = self.window_size
            f.attrs["n_features"] = self.n_features
            f.attrs["feature_names"] = self.feature_names
            f.attrs["prediction_horizon_hours"] = self.prediction_horizon
            
            # Shapes
            f.attrs["train_shape"] = X_train.shape
            f.attrs["val_shape"] = X_val.shape
            f.attrs["test_shape"] = X_test.shape
        
        logger.info(f"Saved windowed data to {output_path}")
        logger.info(f"  Train: {X_train.shape}, positives={y_train.sum()}")
        logger.info(f"  Val:   {X_val.shape}, positives={y_val.sum()}")
        logger.info(f"  Test:  {X_test.shape}, positives={y_test.sum()}")
        
        return str(output_path)


def load_from_hdf5(path: str) -> dict:
    """Load windowed data from HDF5 file.
    
    Returns:
        Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test
    """
    with h5py.File(path, 'r') as f:
        data = {
            "X_train": f["X_train"][:],
            "y_train": f["y_train"][:],
            "X_val": f["X_val"][:],
            "y_val": f["y_val"][:],
            "X_test": f["X_test"][:],
            "y_test": f["y_test"][:],
        }
    
    logger.info(f"Loaded windowed data from {path}")
    for key in ["X_train", "X_val", "X_test"]:
        logger.info(f"  {key}: {data[key].shape}")
    
    return data


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Generate sliding window tensors")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config) if args.config else get_default_config()
    
    # Test with synthetic data
    logger.info("Testing with synthetic data...")
    np.random.seed(42)
    
    n_stays = 100
    feature_names = config.data.feature_names
    
    # Create synthetic features
    rows = []
    cohort_rows = []
    for i in range(n_stays):
        los = np.random.randint(12, 72)
        sepsis = int(np.random.random() < 0.25)
        onset_hour = np.random.randint(8, los) if sepsis else None
        
        for h in range(los):
            row = {"stay_id": i, "hour": h}
            for feat in feature_names:
                row[feat] = np.random.randn()
            rows.append(row)
        
        cohort_rows.append({
            "stay_id": i,
            "hadm_id": i * 10,
            "subject_id": i * 100,
            "sepsis_label": sepsis,
            "sepsis_onset_time": None,
            "intime": pd.Timestamp("2150-01-01") + pd.Timedelta(hours=i*100),
            "outtime": pd.Timestamp("2150-01-01") + pd.Timedelta(hours=i*100 + los),
            "anchor_year_group": "2014 - 2016",
        })
    
    features = pd.DataFrame(rows)
    cohort = pd.DataFrame(cohort_rows)
    
    generator = WindowGenerator(config)
    X, y, meta = generator.generate_windows(features, cohort, "test_synthetic")
    
    print(f"\nSynthetic test:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Positive rate: {y.mean():.1%}")
    print(f"  Any NaN in X: {np.isnan(X).any()}")
