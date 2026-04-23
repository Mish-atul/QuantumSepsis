"""
QuantumSepsis Shield — Feature Extraction from MIMIC-IV
========================================================

Extracts 12 physiological variables per patient per hour:
  - 8 vitals from chartevents (HR, SBP, DBP, MAP, Temp, RR, SpO2, GCS)
  - 4 labs from labevents (Lactate, WBC, Creatinine, Platelets)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import Config, get_default_config
from src.data.cohort_extraction_optimized import load_table_chunked

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts hourly physiological features for each ICU stay.
    
    Output: DataFrame with columns:
        stay_id, hour (relative to intime), and 12 feature columns
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = config.data.mimic_raw_dir
        self.output_dir = Path(config.data.processed_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.vitals_item_ids = config.data.vitals_item_ids
        self.lab_item_ids = config.data.lab_item_ids
        self.feature_names = config.data.feature_names
    
    def extract_features(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Extract hourly features for all stays in the cohort.
        
        Args:
            cohort: DataFrame with columns [stay_id, hadm_id, subject_id, intime, outtime]
        
        Returns:
            DataFrame with columns: stay_id, hour, + 12 feature columns
        """
        logger.info("=" * 60)
        logger.info("Extracting physiological features")
        logger.info("=" * 60)
        
        # Load raw data tables
        vitals_df = self._load_vitals(cohort)
        labs_df = self._load_labs(cohort)
        
        # Extract per-stay hourly features
        all_features = []
        
        for _, stay in tqdm(cohort.iterrows(), total=len(cohort),
                           desc="Extracting features"):
            stay_features = self._extract_stay_features(
                stay, vitals_df, labs_df
            )
            if stay_features is not None:
                all_features.append(stay_features)
        
        if not all_features:
            raise ValueError("No features extracted! Check data paths and cohort.")
        
        features = pd.concat(all_features, ignore_index=True)
        logger.info(f"Extracted features: {features.shape}")
        logger.info(f"  Stays with features: {features['stay_id'].nunique()}")
        logger.info(f"  Total hourly records: {len(features):,}")
        
        # Save
        output_path = self.output_dir / "hourly_features.parquet"
        features.to_parquet(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
        
        return features
    
    def _load_vitals(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Load vital signs from chartevents."""
        logger.info("Loading vital signs from chartevents...")
        
        # Collect all item IDs
        all_item_ids = set()
        item_to_feature = {}
        for feature_name, item_ids in self.vitals_item_ids.items():
            all_item_ids.update(item_ids)
            for iid in item_ids:
                item_to_feature[iid] = feature_name
        
        # Load chartevents with chunked filtering to avoid OOM on full table
        charts = load_table_chunked(
            self.data_dir, "chartevents", "icu",
            usecols=["stay_id", "charttime", "itemid", "valuenum"],
            filter_col="itemid",
            filter_values=all_item_ids,
            chunksize=1_000_000,
        )
        if len(charts) == 0:
            logger.warning("No vital sign rows matched configured item IDs")
            return pd.DataFrame(columns=["stay_id", "charttime", "feature", "valuenum"])

        charts["charttime"] = pd.to_datetime(charts["charttime"])
        
        # Filter to cohort stays
        stay_ids = set(cohort["stay_id"].values)
        charts = charts[charts["stay_id"].isin(stay_ids)].copy()
        
        # Map item IDs to feature names
        charts["feature"] = charts["itemid"].map(item_to_feature)
        
        # Remove physiologically impossible values
        charts = self._clean_vitals(charts)
        
        logger.info(f"  Loaded {len(charts):,} vital sign measurements")
        return charts[["stay_id", "charttime", "feature", "valuenum"]]
    
    def _load_labs(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Load lab values from labevents."""
        logger.info("Loading lab values from labevents...")
        
        lab_ids = set(self.lab_item_ids.values())
        id_to_feature = {v: k for k, v in self.lab_item_ids.items()}
        
        # Load labevents with chunked filtering to avoid OOM on full table
        labs = load_table_chunked(
            self.data_dir, "labevents", "hosp",
            usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
            filter_col="itemid",
            filter_values=lab_ids,
            chunksize=1_000_000,
        )
        if len(labs) == 0:
            logger.warning("No lab rows matched configured item IDs")
            return pd.DataFrame(columns=["stay_id", "charttime", "feature", "valuenum"])

        labs["charttime"] = pd.to_datetime(labs["charttime"])
        
        # Filter to relevant items
        hadm_ids = set(cohort["hadm_id"].values)
        labs = labs[
            (labs["hadm_id"].isin(hadm_ids))
        ].copy()
        
        # Map item IDs to feature names
        labs["feature"] = labs["itemid"].map(id_to_feature)
        
        # Remove impossible values
        labs = self._clean_labs(labs)
        
        # Map hadm_id to stay_id
        hadm_to_stay = cohort.set_index("hadm_id")["stay_id"].to_dict()
        labs["stay_id"] = labs["hadm_id"].map(hadm_to_stay)
        labs = labs.dropna(subset=["stay_id"])
        labs["stay_id"] = labs["stay_id"].astype(int)
        
        logger.info(f"  Loaded {len(labs):,} lab measurements")
        return labs[["stay_id", "charttime", "feature", "valuenum"]]
    
    def _clean_vitals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove physiologically impossible vital sign values."""
        n_before = len(df)
        
        # Define valid ranges
        valid_ranges = {
            "heart_rate": (0, 300),
            "sbp": (0, 400),
            "dbp": (0, 300),
            "map": (0, 300),
            "temperature": (25, 45),  # Celsius
            "resp_rate": (0, 80),
            "spo2": (0, 100),
            "gcs_total": (3, 15),
        }
        
        masks = []
        for feature, (low, high) in valid_ranges.items():
            feature_mask = df["feature"] == feature
            value_mask = (df["valuenum"] >= low) & (df["valuenum"] <= high)
            masks.append(~feature_mask | (feature_mask & value_mask))
        
        combined_mask = pd.concat(masks, axis=1).all(axis=1) if masks else pd.Series(True, index=df.index)
        df = df[combined_mask]
        
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info(f"  Removed {n_removed:,} invalid vital sign values")
        
        return df
    
    def _clean_labs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove physiologically impossible lab values."""
        n_before = len(df)
        
        valid_ranges = {
            "lactate": (0, 30),
            "wbc": (0, 200),
            "creatinine": (0, 50),
            "platelets": (0, 2000),
        }
        
        masks = []
        for feature, (low, high) in valid_ranges.items():
            feature_mask = df["feature"] == feature
            value_mask = (df["valuenum"] >= low) & (df["valuenum"] <= high)
            masks.append(~feature_mask | (feature_mask & value_mask))
        
        combined_mask = pd.concat(masks, axis=1).all(axis=1) if masks else pd.Series(True, index=df.index)
        df = df[combined_mask]
        
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info(f"  Removed {n_removed:,} invalid lab values")
        
        return df
    
    def _extract_stay_features(
        self,
        stay: pd.Series,
        vitals_df: pd.DataFrame,
        labs_df: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """Extract hourly features for a single ICU stay.
        
        Args:
            stay: Series with stay info (stay_id, intime, outtime)
            vitals_df: All vital signs data
            labs_df: All lab data
        
        Returns:
            DataFrame with columns: stay_id, hour, + 12 feature columns
        """
        stay_id = stay["stay_id"]
        intime = stay["intime"]
        outtime = stay["outtime"]

        # Skip stays with missing or invalid ICU timestamps.
        if pd.isna(intime) or pd.isna(outtime) or outtime <= intime:
            return None
        
        los_hours = int((outtime - intime).total_seconds() / 3600)
        if los_hours < 6:  # Need at least 6 hours for one window
            return None
        
        # Get data for this stay
        stay_vitals = vitals_df[vitals_df["stay_id"] == stay_id]
        stay_labs = labs_df[labs_df["stay_id"] == stay_id]
        
        if len(stay_vitals) == 0 and len(stay_labs) == 0:
            return None
        
        # Combine vitals and labs
        all_data = pd.concat([stay_vitals, stay_labs], ignore_index=True)
        
        # Compute relative hour from ICU admission
        all_data["hour"] = (
            (all_data["charttime"] - intime).dt.total_seconds() / 3600
        ).astype(int)
        
        # Filter to valid ICU period
        all_data = all_data[(all_data["hour"] >= 0) & (all_data["hour"] < los_hours)]
        
        if len(all_data) == 0:
            return None
        
        # Pivot: aggregate to hourly values (median within each hour)
        hourly = (
            all_data
            .groupby(["hour", "feature"])["valuenum"]
            .median()
            .reset_index()
        )
        
        # Create feature matrix
        hours = range(los_hours)
        feature_matrix = []
        
        for h in hours:
            row = {"stay_id": stay_id, "hour": h}
            hour_data = hourly[hourly["hour"] == h]
            
            for feat in self.feature_names:
                vals = hour_data[hour_data["feature"] == feat]["valuenum"]
                row[feat] = vals.values[0] if len(vals) > 0 else np.nan
            
            feature_matrix.append(row)
        
        return pd.DataFrame(feature_matrix)


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Extract physiological features")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cohort", type=str, default="data/processed/cohort.csv")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override MIMIC-IV data directory")
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config) if args.config else get_default_config()
    if args.data_dir:
        config.data.mimic_raw_dir = args.data_dir

    cohort = pd.read_csv(args.cohort, parse_dates=["intime", "outtime"])
    
    extractor = FeatureExtractor(config)
    features = extractor.extract_features(cohort)
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Missing rate per feature:")
    for feat in config.data.feature_names:
        miss = features[feat].isna().mean()
        print(f"  {feat}: {miss:.1%}")
