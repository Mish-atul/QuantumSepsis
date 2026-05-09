"""
FOU Windowing
=============
Create sliding windows for FOU detection:
- Window size: 24 hours
- Stride: 6 hours
- Prediction horizon: 48 hours (predict FOU category 48 hours ahead)

Output: HDF5 file with shape (N, 24, 27)
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FouWindower:
    """Create sliding windows for FOU detection."""

    def __init__(self, train_path: str, val_path: str, test_path: str, output_path: str):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.output_path = Path(output_path)

        # FOU windowing parameters
        self.window_size = 24  # hours
        self.stride = 6  # hours
        self.prediction_horizon = 48  # hours

        # Feature names (27 total)
        self.feature_names = [
            "heart_rate", "sbp", "dbp", "map", "temperature",
            "resp_rate", "spo2", "gcs_total",
            "lactate", "wbc", "creatinine", "platelets",
            "temp_max_24h", "temp_variability", "fever_duration_hours",
            "crp", "esr", "procalcitonin", "ferritin", "ldh", "albumin",
            "antibiotic_days", "culture_negative_count", "immunosuppressed",
            "weight_loss", "night_sweats_proxy", "rash_documented"
        ]

    def create_windows(self) -> None:
        """Create windows for train/val/test splits."""
        logger.info("Starting FOU windowing...")

        # Load data
        logger.info("Loading preprocessed data...")
        train_df = pd.read_parquet(self.train_path)
        val_df = pd.read_parquet(self.val_path)
        test_df = pd.read_parquet(self.test_path)

        logger.info(f"Train: {len(train_df)} rows")
        logger.info(f"Val: {len(val_df)} rows")
        logger.info(f"Test: {len(test_df)} rows")

        # Create windows for each split
        logger.info("Creating train windows...")
        X_train, y_train, meta_train = self._create_windows_for_split(train_df)

        logger.info("Creating val windows...")
        X_val, y_val, meta_val = self._create_windows_for_split(val_df)

        logger.info("Creating test windows...")
        X_test, y_test, meta_test = self._create_windows_for_split(test_df)

        # Save to HDF5
        logger.info(f"Saving windows to {self.output_path}...")
        with h5py.File(self.output_path, 'w') as f:
            # Train
            f.create_dataset('X_train', data=X_train, compression='gzip')
            f.create_dataset('y_train', data=y_train, compression='gzip')
            f.create_dataset('stay_ids_train', data=meta_train['stay_ids'], compression='gzip')
            f.create_dataset('window_starts_train', data=meta_train['window_starts'], compression='gzip')

            # Val
            f.create_dataset('X_val', data=X_val, compression='gzip')
            f.create_dataset('y_val', data=y_val, compression='gzip')
            f.create_dataset('stay_ids_val', data=meta_val['stay_ids'], compression='gzip')
            f.create_dataset('window_starts_val', data=meta_val['window_starts'], compression='gzip')

            # Test
            f.create_dataset('X_test', data=X_test, compression='gzip')
            f.create_dataset('y_test', data=y_test, compression='gzip')
            f.create_dataset('stay_ids_test', data=meta_test['stay_ids'], compression='gzip')
            f.create_dataset('window_starts_test', data=meta_test['window_starts'], compression='gzip')

            # Metadata
            f.attrs['window_size'] = self.window_size
            f.attrs['stride'] = self.stride
            f.attrs['prediction_horizon'] = self.prediction_horizon
            f.attrs['n_features'] = len(self.feature_names)
            f.attrs['feature_names'] = ','.join(self.feature_names)

        logger.info(f"Windows saved to {self.output_path}")
        logger.info(f"Train: {X_train.shape}")
        logger.info(f"Val: {X_val.shape}")
        logger.info(f"Test: {X_test.shape}")

    def _create_windows_for_split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Create windows for a single split."""
        windows = []
        labels = []
        stay_ids = []
        window_starts = []

        # Group by stay
        for stay_id, stay_df in df.groupby('stay_id'):
            stay_df = stay_df.sort_values('hour_bin')

            # Get label for this stay
            label = stay_df['fou_label'].iloc[0]

            # Get feature matrix
            feature_cols = [col for col in self.feature_names if col in stay_df.columns]
            features = stay_df[feature_cols].values

            # Pad missing features with zeros
            if len(feature_cols) < len(self.feature_names):
                missing_features = len(self.feature_names) - len(feature_cols)
                features = np.hstack([features, np.zeros((features.shape[0], missing_features))])

            # Create sliding windows
            max_hour = len(stay_df) - self.prediction_horizon
            for start_hour in range(0, max_hour, self.stride):
                end_hour = start_hour + self.window_size

                # Check if we have enough data for the window
                if end_hour <= len(stay_df):
                    window = features[start_hour:end_hour, :]

                    # Ensure window has correct shape (24, 27)
                    if window.shape[0] == self.window_size:
                        windows.append(window)
                        labels.append(label)
                        stay_ids.append(stay_id)
                        window_starts.append(start_hour)

        # Convert to arrays
        X = np.array(windows, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        meta = {
            'stay_ids': np.array(stay_ids, dtype=np.int32),
            'window_starts': np.array(window_starts, dtype=np.int32)
        }

        logger.info(f"  Created {len(windows)} windows")
        logger.info(f"  Label distribution: {np.bincount(y)}")

        return X, y, meta


def main():
    """Main execution."""
    import sys

    if len(sys.argv) < 5:
        print("Usage: python windowing_fou.py <train_path> <val_path> <test_path> <output_path>")
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]
    output_path = sys.argv[4]

    windower = FouWindower(train_path, val_path, test_path, output_path)
    windower.create_windows()

    # Verify output
    print(f"\n=== FOU Windowing Summary ===")
    with h5py.File(output_path, 'r') as f:
        print(f"X_train shape: {f['X_train'].shape}")
        print(f"y_train shape: {f['y_train'].shape}")
        print(f"X_val shape: {f['X_val'].shape}")
        print(f"y_val shape: {f['y_val'].shape}")
        print(f"X_test shape: {f['X_test'].shape}")
        print(f"y_test shape: {f['y_test'].shape}")
        print(f"\nMetadata:")
        print(f"  Window size: {f.attrs['window_size']} hours")
        print(f"  Stride: {f.attrs['stride']} hours")
        print(f"  Prediction horizon: {f.attrs['prediction_horizon']} hours")
        print(f"  Features: {f.attrs['n_features']}")


if __name__ == "__main__":
    main()
