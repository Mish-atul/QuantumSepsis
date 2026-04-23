"""
QuantumSepsis Shield — Data Preprocessing
==========================================

Handles missing data imputation, normalization, and train/val/test splitting.

Imputation strategy:
  1. Forward fill up to 2 consecutive hours
  2. Fallback: per-variable training-set median

Normalization:
  - Per-variable z-score using training set statistics only
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import Config, get_default_config

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocesses extracted features: imputation, normalization, splitting."""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.data.processed_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = config.data.feature_names
        self.forward_fill_limit = config.data.forward_fill_limit_hours
        
        # Will be computed from training data
        self.train_mean: Dict[str, float] = {}
        self.train_std: Dict[str, float] = {}
        self.train_median: Dict[str, float] = {}
    
    def preprocess(
        self,
        features: pd.DataFrame,
        cohort: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run full preprocessing pipeline.
        
        Args:
            features: Hourly feature DataFrame (stay_id, hour, + 12 features)
            cohort: Cohort DataFrame with anchor_year_group and sepsis_label
        
        Returns:
            Tuple of (train_df, val_df, test_df) — all imputed and normalized
        """
        logger.info("=" * 60)
        logger.info("Preprocessing features")
        logger.info("=" * 60)
        
        # Step 1: Temporal train/test split
        train_stays, val_stays, test_stays = self._temporal_split(cohort)
        
        train_df = features[features["stay_id"].isin(train_stays)].copy()
        val_df = features[features["stay_id"].isin(val_stays)].copy()
        test_df = features[features["stay_id"].isin(test_stays)].copy()
        
        logger.info(f"Split sizes:")
        logger.info(f"  Train: {train_df['stay_id'].nunique()} stays, {len(train_df):,} records")
        logger.info(f"  Val:   {val_df['stay_id'].nunique()} stays, {len(val_df):,} records")
        logger.info(f"  Test:  {test_df['stay_id'].nunique()} stays, {len(test_df):,} records")
        
        # Step 2: Compute training statistics (before imputation)
        self._compute_train_stats(train_df)
        
        # Step 3: Forward fill within each stay
        train_df = self._forward_fill(train_df)
        val_df = self._forward_fill(val_df)
        test_df = self._forward_fill(test_df)
        
        # Step 4: Median imputation (remaining NaNs)
        train_df = self._median_impute(train_df)
        val_df = self._median_impute(val_df)
        test_df = self._median_impute(test_df)
        
        # Step 5: Z-score normalization
        train_df = self._normalize(train_df)
        val_df = self._normalize(val_df)
        test_df = self._normalize(test_df)
        
        # Report missing rate after imputation
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            missing = df[self.feature_names].isna().mean().mean()
            logger.info(f"  {name} remaining missing rate: {missing:.4%}")
        
        # Save normalization statistics
        self._save_stats()
        
        # Save processed splits
        train_df.to_parquet(self.output_dir / "train_features.parquet", index=False)
        val_df.to_parquet(self.output_dir / "val_features.parquet", index=False)
        test_df.to_parquet(self.output_dir / "test_features.parquet", index=False)
        
        logger.info("Preprocessing complete. Saved to parquet files.")
        
        return train_df, val_df, test_df
    
    def _temporal_split(
        self,
        cohort: pd.DataFrame,
    ) -> Tuple[set, set, set]:
        """Temporal train/val/test split based on anchor_year_group.
        
        Train: anchor_year_group <= 2017-2019
        Test:  anchor_year_group = 2020-2022
        Val:   15% random split from training stays (stratified by sepsis label)
        """
        train_groups = set(self.config.data.train_year_groups)
        test_groups = set(self.config.data.test_year_groups)
        
        train_cohort = cohort[cohort["anchor_year_group"].isin(train_groups)]
        test_cohort = cohort[cohort["anchor_year_group"].isin(test_groups)]
        
        # If anchor_year_group is not available, fall back to random split
        if len(train_cohort) == 0 or len(test_cohort) == 0:
            logger.warning("anchor_year_group not available. Using random 70/15/15 split.")
            from sklearn.model_selection import train_test_split
            
            stays = cohort["stay_id"].values
            labels = cohort["sepsis_label"].values
            
            train_val_stays, test_stays = train_test_split(
                stays, test_size=0.15, stratify=labels, random_state=42
            )
            
            train_val_labels = cohort[cohort["stay_id"].isin(train_val_stays)]["sepsis_label"].values
            train_stays, val_stays = train_test_split(
                train_val_stays, test_size=0.176,  # 0.176 of 0.85 ≈ 0.15
                stratify=train_val_labels, random_state=42
            )
            
            return set(train_stays), set(val_stays), set(test_stays)
        
        # Stratified validation split from training
        from sklearn.model_selection import train_test_split
        
        train_stays_all = train_cohort["stay_id"].values
        train_labels = train_cohort["sepsis_label"].values
        
        train_stays, val_stays = train_test_split(
            train_stays_all,
            test_size=self.config.data.val_fraction,
            stratify=train_labels,
            random_state=42,
        )
        
        test_stays = set(test_cohort["stay_id"].values)
        
        logger.info(f"Temporal split:")
        logger.info(f"  Train groups: {train_groups}")
        logger.info(f"  Test groups:  {test_groups}")
        
        # Log sepsis prevalence per split
        for name, stay_set in [("Train", set(train_stays)), ("Val", set(val_stays)), ("Test", test_stays)]:
            subset = cohort[cohort["stay_id"].isin(stay_set)]
            prev = subset["sepsis_label"].mean() if len(subset) > 0 else 0
            logger.info(f"  {name} sepsis prevalence: {prev:.1%}")
        
        return set(train_stays), set(val_stays), test_stays
    
    def _compute_train_stats(self, train_df: pd.DataFrame) -> None:
        """Compute training set statistics for normalization and imputation."""
        for feat in self.feature_names:
            vals = train_df[feat].dropna()
            self.train_mean[feat] = float(vals.mean()) if len(vals) > 0 else 0.0
            self.train_std[feat] = float(vals.std()) if len(vals) > 0 else 1.0
            self.train_median[feat] = float(vals.median()) if len(vals) > 0 else 0.0
            
            # Prevent division by zero
            if self.train_std[feat] < 1e-8:
                self.train_std[feat] = 1.0
        
        logger.info("Training statistics computed:")
        for feat in self.feature_names:
            logger.info(
                f"  {feat:15s}: mean={self.train_mean[feat]:8.2f}, "
                f"std={self.train_std[feat]:8.2f}, "
                f"median={self.train_median[feat]:8.2f}"
            )
    
    def _forward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward fill missing values within each stay (up to limit)."""
        result = df.copy()
        
        for stay_id in result["stay_id"].unique():
            mask = result["stay_id"] == stay_id
            stay_data = result.loc[mask, self.feature_names]
            
            # Forward fill with limit
            filled = stay_data.ffill(limit=self.forward_fill_limit)
            result.loc[mask, self.feature_names] = filled
        
        return result
    
    def _median_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill remaining NaNs with training set median."""
        result = df.copy()
        for feat in self.feature_names:
            result[feat] = result[feat].fillna(self.train_median[feat])
        return result
    
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalization using training set statistics."""
        result = df.copy()
        for feat in self.feature_names:
            result[feat] = (result[feat] - self.train_mean[feat]) / self.train_std[feat]
        return result
    
    def _save_stats(self) -> None:
        """Save normalization statistics to JSON."""
        stats = {
            "feature_names": self.feature_names,
            "train_mean": self.train_mean,
            "train_std": self.train_std,
            "train_median": self.train_median,
        }
        
        output_path = self.output_dir / "normalization_stats.json"
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Normalization stats saved to {output_path}")
    
    def load_stats(self, path: str) -> None:
        """Load previously computed normalization statistics."""
        with open(path, 'r') as f:
            stats = json.load(f)
        
        self.train_mean = stats["train_mean"]
        self.train_std = stats["train_std"]
        self.train_median = stats["train_median"]
        logger.info(f"Loaded normalization stats from {path}")


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Preprocess features")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--features", type=str, default="data/processed/hourly_features.parquet")
    parser.add_argument("--cohort", type=str, default="data/processed/cohort.csv")
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config) if args.config else get_default_config()
    features = pd.read_parquet(args.features)
    cohort = pd.read_csv(args.cohort, parse_dates=["intime", "outtime"])
    
    preprocessor = Preprocessor(config)
    train_df, val_df, test_df = preprocessor.preprocess(features, cohort)
    print(f"\nTrain: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
