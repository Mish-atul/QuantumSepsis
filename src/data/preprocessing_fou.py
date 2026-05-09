"""
FOU Preprocessing
=================
Preprocessing pipeline for FOU features:
1. Forward-fill missing values (limit: 6 hours)
2. Median imputation for remaining NaNs
3. Z-score normalization (train-set statistics)
4. Temporal train/val/test split by anchor_year_group
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FouPreprocessor:
    """Preprocess FOU features."""

    def __init__(self, features_path: str, cohort_path: str, output_dir: str):
        self.features_path = Path(features_path)
        self.cohort_path = Path(cohort_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # FOU-specific parameters
        self.forward_fill_limit = 6  # hours (longer than sepsis due to slower progression)

        # Feature names (27 total)
        self.feature_names = [
            # 12 reused from sepsis
            "heart_rate", "sbp", "dbp", "map", "temperature",
            "resp_rate", "spo2", "gcs_total",
            "lactate", "wbc", "creatinine", "platelets",
            # 15 FOU-specific
            "temp_max_24h", "temp_variability", "fever_duration_hours",
            "crp", "esr", "procalcitonin", "ferritin", "ldh", "albumin",
            "antibiotic_days", "culture_negative_count", "immunosuppressed",
            "weight_loss", "night_sweats_proxy", "rash_documented"
        ]

    def _get_file_path(self, base_dir: Path, filename: str) -> Path:
        """Get file path, checking for .gz extension."""
        base_path = base_dir / filename
        if base_path.exists():
            return base_path
        gz_path = base_dir / f"{filename}.gz"
        if gz_path.exists():
            return gz_path
        raise FileNotFoundError(f"Neither {base_path} nor {gz_path} found")

    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run full preprocessing pipeline."""
        logger.info("Starting FOU preprocessing...")

        # Load features and cohort
        logger.info(f"Loading features from {self.features_path}...")
        features = pd.read_parquet(self.features_path)
        logger.info(f"Loaded {len(features)} feature rows")

        logger.info(f"Loading cohort from {self.cohort_path}...")
        cohort = pd.read_csv(self.cohort_path)
        logger.info(f"Loaded {len(cohort)} cohort entries")

        # Merge with cohort to get labels and metadata
        features = features.merge(
            cohort[['stay_id', 'subject_id', 'fou_label']],
            on='stay_id',
            how='inner'
        )
        logger.info(f"After merge: {len(features)} rows")

        # Step 1: Forward-fill missing values
        logger.info("Step 1: Forward-filling missing values...")
        features = self._forward_fill(features)

        # Step 2: Median imputation
        logger.info("Step 2: Median imputation...")
        features = self._median_impute(features)

        # Step 3: Temporal split
        logger.info("Step 3: Temporal train/val/test split...")
        train_df, val_df, test_df = self._temporal_split(features, cohort)

        # Step 4: Z-score normalization (fit on train, apply to all)
        logger.info("Step 4: Z-score normalization...")
        train_df, val_df, test_df, norm_stats = self._normalize(train_df, val_df, test_df)

        # Save normalization statistics
        norm_stats_path = self.output_dir / "fou_normalization_stats.json"
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        logger.info(f"Normalization stats saved to {norm_stats_path}")

        # Save preprocessed data
        train_path = self.output_dir / "fou_train_features.parquet"
        val_path = self.output_dir / "fou_val_features.parquet"
        test_path = self.output_dir / "fou_test_features.parquet"

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        logger.info(f"Train: {len(train_df)} rows → {train_path}")
        logger.info(f"Val: {len(val_df)} rows → {val_path}")
        logger.info(f"Test: {len(test_df)} rows → {test_path}")

        return train_df, val_df, test_df

    def _forward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill missing values within each stay (limit: 6 hours)."""
        df = df.sort_values(['stay_id', 'hour_bin'])

        for feature in self.feature_names:
            if feature in df.columns:
                df[feature] = df.groupby('stay_id')[feature].fillna(method='ffill', limit=self.forward_fill_limit)

        return df

    def _median_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Median imputation for remaining NaNs."""
        for feature in self.feature_names:
            if feature in df.columns:
                median_val = df[feature].median()
                df[feature] = df[feature].fillna(median_val)
                logger.info(f"  {feature}: median={median_val:.2f}")

        return df

    def _temporal_split(self, features: pd.DataFrame, cohort: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Temporal train/val/test split by anchor_year_group."""
        # Load patients to get anchor_year_group
        try:
            patients_dir = Path(self.cohort_path).parent.parent / "raw" / "physionet.org" / "files" / "mimiciv" / "3.1" / "hosp"
            patients_path = self._get_file_path(patients_dir, "patients.csv")
            patients = pd.read_csv(patients_path)
            patients = patients[['subject_id', 'anchor_year_group']]
        except FileNotFoundError:
            logger.warning("patients.csv not found, using random split")
            # Fallback: random split
            train_stays = features['stay_id'].unique()[:int(0.7 * len(features['stay_id'].unique()))]
            val_stays = features['stay_id'].unique()[int(0.7 * len(features['stay_id'].unique())):int(0.85 * len(features['stay_id'].unique()))]
            test_stays = features['stay_id'].unique()[int(0.85 * len(features['stay_id'].unique())):]

            train_df = features[features['stay_id'].isin(train_stays)]
            val_df = features[features['stay_id'].isin(val_stays)]
            test_df = features[features['stay_id'].isin(test_stays)]

            return train_df, val_df, test_df

        # Merge with anchor_year_group
        features = features.merge(patients, on='subject_id', how='left')

        # Split by year group
        train_years = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"]
        test_years = ["2020 - 2022"]

        train_df = features[features['anchor_year_group'].isin(train_years)].copy()
        test_df = features[features['anchor_year_group'].isin(test_years)].copy()

        # Split train into train/val (85/15)
        train_stays = train_df['stay_id'].unique()
        np.random.seed(42)
        np.random.shuffle(train_stays)

        val_size = int(0.15 * len(train_stays))
        val_stays = train_stays[:val_size]
        train_stays = train_stays[val_size:]

        val_df = train_df[train_df['stay_id'].isin(val_stays)].copy()
        train_df = train_df[train_df['stay_id'].isin(train_stays)].copy()

        logger.info(f"  Train: {len(train_df)} rows, {len(train_stays)} stays")
        logger.info(f"  Val: {len(val_df)} rows, {len(val_stays)} stays")
        logger.info(f"  Test: {len(test_df)} rows, {len(test_df['stay_id'].nunique())} stays")

        return train_df, val_df, test_df

    def _normalize(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """Z-score normalization using train statistics."""
        norm_stats = {}

        for feature in self.feature_names:
            if feature not in train_df.columns:
                logger.warning(f"Feature {feature} not found, skipping normalization")
                continue

            # Compute mean and std from training set
            mean = train_df[feature].mean()
            std = train_df[feature].std()

            if std == 0 or np.isnan(std):
                logger.warning(f"  {feature}: std=0 or NaN, skipping normalization")
                norm_stats[feature] = {"mean": float(mean), "std": 1.0}
                continue

            # Apply normalization
            train_df[feature] = (train_df[feature] - mean) / std
            val_df[feature] = (val_df[feature] - mean) / std
            test_df[feature] = (test_df[feature] - mean) / std

            norm_stats[feature] = {"mean": float(mean), "std": float(std)}
            logger.info(f"  {feature}: mean={mean:.2f}, std={std:.2f}")

        return train_df, val_df, test_df, norm_stats


def main():
    """Main execution."""
    import sys

    if len(sys.argv) < 4:
        print("Usage: python preprocessing_fou.py <features_path> <cohort_path> <output_dir>")
        sys.exit(1)

    features_path = sys.argv[1]
    cohort_path = sys.argv[2]
    output_dir = sys.argv[3]

    preprocessor = FouPreprocessor(features_path, cohort_path, output_dir)
    train_df, val_df, test_df = preprocessor.preprocess()

    print(f"\n=== FOU Preprocessing Summary ===")
    print(f"Train: {len(train_df)} rows")
    print(f"Val: {len(val_df)} rows")
    print(f"Test: {len(test_df)} rows")
    print(f"\nLabel distribution (train):")
    print(train_df['fou_label'].value_counts())


if __name__ == "__main__":
    main()
