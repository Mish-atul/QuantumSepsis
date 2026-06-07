"""
FOU Feature Engineering
=======================
Extract 27 features for FOU detection:
- 12 reused from sepsis (vitals + labs)
- 15 FOU-specific features

Features:
1-12: heart_rate, sbp, dbp, map, temperature, resp_rate, spo2, gcs_total,
      lactate, wbc, creatinine, platelets
13-27: temp_max_24h, temp_variability, fever_duration_hours, crp, esr,
       procalcitonin, ferritin, ldh, albumin, antibiotic_days,
       culture_negative_count, immunosuppressed, weight_loss,
       night_sweats_proxy, rash_documented
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FouFeatureEngineer:
    """Extract FOU-specific features from MIMIC-IV."""

    def __init__(self, mimic_dir: str, cohort_path: str, output_dir: str):
        self.mimic_dir = Path(mimic_dir)
        self.cohort_path = Path(cohort_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load cohort
        self.cohort = pd.read_csv(cohort_path)
        logger.info(f"Loaded cohort: {len(self.cohort)} stays")

        # Item IDs
        self.vitals_item_ids = {
            "heart_rate": [211, 220045],
            "sbp": [51, 442, 455, 6701, 220179, 220050],
            "dbp": [8368, 8440, 8441, 8555, 220180, 220051],
            "map": [52, 456, 6702, 220052, 220181],
            "temperature": [223762, 226329],
            "resp_rate": [615, 618, 220210, 224690],
            "spo2": [646, 220277],
            "gcs_total": [198, 226755, 227013],
        }

        self.lab_item_ids = {
            "lactate": 50813,
            "wbc": 51301,
            "creatinine": 50912,
            "platelets": 51265,
            "crp": 50889,
            "esr": 51288,
            "procalcitonin": 50963,
            "ferritin": 50896,
            "ldh": 50954,
            "albumin": 50862,
        }

    def _get_file_path(self, subdir: str, filename: str) -> Path:
        """Get file path, checking for .gz extension."""
        base_path = self.mimic_dir / subdir / filename
        if base_path.exists():
            return base_path
        gz_path = self.mimic_dir / subdir / f"{filename}.gz"
        if gz_path.exists():
            return gz_path
        raise FileNotFoundError(f"Neither {base_path} nor {gz_path} found")

    def extract_features(self) -> pd.DataFrame:
        """Extract all 27 features."""
        logger.info("Starting FOU feature extraction...")

        stay_ids = self.cohort['stay_id'].unique()

        # Extract base features (1-12: reused from sepsis)
        logger.info("Extracting base vitals and labs (features 1-12)...")
        base_features = self._extract_base_features(stay_ids)

        # Extract FOU-specific features (13-27)
        logger.info("Extracting FOU-specific features (13-27)...")
        fou_features = self._extract_fou_specific_features(stay_ids, base_features)

        # Merge all features
        logger.info("Merging features...")
        all_features = base_features.merge(fou_features, on=['stay_id', 'hour_bin'], how='left')

        # Save
        output_path = self.output_dir / "fou_hourly_features.parquet"
        all_features.to_parquet(output_path, index=False)
        logger.info(f"Features saved to {output_path}")

        return all_features

    def _extract_base_features(self, stay_ids: np.ndarray) -> pd.DataFrame:
        """Extract base features 1-12 (reused from sepsis)."""
        # Extract vitals
        vitals = self._extract_vitals(stay_ids)

        # Extract labs
        labs = self._extract_labs(stay_ids)

        # Merge vitals and labs
        features = vitals.merge(labs, on=['stay_id', 'hour_bin'], how='outer')

        return features

    def _extract_vitals(self, stay_ids: np.ndarray) -> pd.DataFrame:
        """Extract vital signs from chartevents."""
        logger.info("Loading chartevents...")

        vitals_list = []
        chunksize = 10_000_000

        # Load ICU stay times for binning
        icustays_path = self._get_file_path("icu", "icustays.csv")
        icustays = pd.read_csv(
            icustays_path,
            usecols=['stay_id', 'intime', 'outtime']
        )
        icustays = icustays[icustays['stay_id'].isin(stay_ids)]
        icustays['intime'] = pd.to_datetime(icustays['intime'])
        icustays['outtime'] = pd.to_datetime(icustays['outtime'])

        # Flatten item IDs
        all_vital_items = []
        for items in self.vitals_item_ids.values():
            all_vital_items.extend(items)

        chartevents_path = self._get_file_path("icu", "chartevents.csv")
        for chunk in pd.read_csv(
            chartevents_path,
            usecols=['stay_id', 'charttime', 'itemid', 'valuenum'],
            chunksize=chunksize
        ):
            chunk_vitals = chunk[
                (chunk['itemid'].isin(all_vital_items)) &
                (chunk['stay_id'].isin(stay_ids))
            ].copy()

            if len(chunk_vitals) > 0:
                vitals_list.append(chunk_vitals)

        if not vitals_list:
            logger.warning("No vitals found!")
            return pd.DataFrame()

        vitals = pd.concat(vitals_list, ignore_index=True)
        vitals['charttime'] = pd.to_datetime(vitals['charttime'])

        # Merge with ICU stay times
        vitals = vitals.merge(icustays, on='stay_id', how='inner')

        # Calculate hour bin
        vitals['hour_bin'] = ((vitals['charttime'] - vitals['intime']).dt.total_seconds() / 3600).astype(int)

        # Filter valid hour bins (within ICU stay)
        vitals = vitals[vitals['hour_bin'] >= 0]

        # Map item IDs to feature names
        item_to_feature = {}
        for feature, items in self.vitals_item_ids.items():
            for item in items:
                item_to_feature[item] = feature

        vitals['feature'] = vitals['itemid'].map(item_to_feature)

        # Convert Fahrenheit to Celsius for temperature
        temp_mask = vitals['feature'] == 'temperature'
        vitals.loc[temp_mask & (vitals['valuenum'] > 50), 'valuenum'] = (vitals['valuenum'] - 32) * 5/9

        # Aggregate by hour bin (median)
        vitals_agg = vitals.groupby(['stay_id', 'hour_bin', 'feature'])['valuenum'].median().reset_index()

        # Pivot to wide format
        vitals_wide = vitals_agg.pivot(index=['stay_id', 'hour_bin'], columns='feature', values='valuenum').reset_index()

        return vitals_wide

    def _extract_labs(self, stay_ids: np.ndarray) -> pd.DataFrame:
        """Extract lab values from labevents."""
        logger.info("Loading labevents...")

        # Load ICU stay times
        icustays_path = self._get_file_path("icu", "icustays.csv")
        icustays = pd.read_csv(
            icustays_path,
            usecols=['stay_id', 'subject_id', 'intime', 'outtime']
        )
        icustays = icustays[icustays['stay_id'].isin(stay_ids)]
        icustays['intime'] = pd.to_datetime(icustays['intime'])
        icustays['outtime'] = pd.to_datetime(icustays['outtime'])

        # Load lab events
        labevents_path = self._get_file_path("hosp", "labevents.csv")
        labs = pd.read_csv(
            labevents_path,
            usecols=['subject_id', 'charttime', 'itemid', 'valuenum']
        )
        labs = labs[labs['itemid'].isin(self.lab_item_ids.values())]
        labs = labs[labs['subject_id'].isin(icustays['subject_id'].unique())]
        labs['charttime'] = pd.to_datetime(labs['charttime'])

        # Merge with ICU stays
        labs = labs.merge(icustays, on='subject_id', how='inner')

        # Filter labs within ICU stay
        labs = labs[(labs['charttime'] >= labs['intime']) & (labs['charttime'] <= labs['outtime'])]

        # Calculate hour bin
        labs['hour_bin'] = ((labs['charttime'] - labs['intime']).dt.total_seconds() / 3600).astype(int)
        labs = labs[labs['hour_bin'] >= 0]

        # Map item IDs to feature names
        item_to_feature = {v: k for k, v in self.lab_item_ids.items()}
        labs['feature'] = labs['itemid'].map(item_to_feature)

        # Aggregate by hour bin (median)
        labs_agg = labs.groupby(['stay_id', 'hour_bin', 'feature'])['valuenum'].median().reset_index()

        # Pivot to wide format
        labs_wide = labs_agg.pivot(index=['stay_id', 'hour_bin'], columns='feature', values='valuenum').reset_index()

        return labs_wide

    def _extract_fou_specific_features(self, stay_ids: np.ndarray, base_features: pd.DataFrame) -> pd.DataFrame:
        """Extract FOU-specific features 13-27."""
        logger.info("Extracting FOU-specific features...")

        # Initialize FOU features dataframe
        fou_features = base_features[['stay_id', 'hour_bin']].copy()

        # Feature 13-15: Temperature-based features
        logger.info("Computing temperature features...")
        temp_features = self._compute_temperature_features(base_features)
        fou_features = fou_features.merge(temp_features, on=['stay_id', 'hour_bin'], how='left')

        # Feature 16-21: Already extracted in labs (crp, esr, procalcitonin, ferritin, ldh, albumin)
        # These are in base_features

        # Feature 22: Antibiotic days
        logger.info("Computing antibiotic exposure...")
        antibiotic_features = self._compute_antibiotic_days(stay_ids)
        fou_features = fou_features.merge(antibiotic_features, on=['stay_id', 'hour_bin'], how='left')

        # Feature 23: Culture negative count
        logger.info("Computing culture negative count...")
        culture_features = self._compute_culture_negative_count(stay_ids)
        fou_features = fou_features.merge(culture_features, on=['stay_id', 'hour_bin'], how='left')

        # Feature 24: Immunosuppressed status
        logger.info("Computing immunosuppressed status...")
        immunosupp_features = self._compute_immunosuppressed(stay_ids)
        fou_features = fou_features.merge(immunosupp_features, on=['stay_id', 'hour_bin'], how='left')

        # Feature 25: Weight loss
        logger.info("Computing weight loss...")
        weight_features = self._compute_weight_loss(stay_ids)
        fou_features = fou_features.merge(weight_features, on=['stay_id', 'hour_bin'], how='left')

        # Feature 26: Night sweats proxy
        logger.info("Computing night sweats proxy...")
        night_sweats = self._compute_night_sweats_proxy(base_features)
        fou_features = fou_features.merge(night_sweats, on=['stay_id', 'hour_bin'], how='left')

        # Feature 27: Rash documented
        logger.info("Computing rash documentation...")
        rash_features = self._compute_rash_documented(stay_ids)
        fou_features = fou_features.merge(rash_features, on=['stay_id', 'hour_bin'], how='left')

        return fou_features

    def _compute_temperature_features(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """Compute temperature-based features: max_24h, variability, duration."""
        if 'temperature' not in base_features.columns:
            logger.warning("Temperature not found in base features")
            return pd.DataFrame()

        temp_df = base_features[['stay_id', 'hour_bin', 'temperature']].copy()

        # Feature 13: temp_max_24h (rolling max over 24 hours)
        temp_df = temp_df.sort_values(['stay_id', 'hour_bin'])
        temp_df['temp_max_24h'] = temp_df.groupby('stay_id')['temperature'].transform(
            lambda x: x.rolling(window=24, min_periods=1).max()
        )

        # Feature 14: temp_variability (rolling std over 24 hours)
        temp_df['temp_variability'] = temp_df.groupby('stay_id')['temperature'].transform(
            lambda x: x.rolling(window=24, min_periods=1).std()
        )

        # Feature 15: fever_duration_hours (cumulative hours with fever > 38.3°C)
        temp_df['has_fever'] = (temp_df['temperature'] > 38.3).astype(int)
        temp_df['fever_duration_hours'] = temp_df.groupby('stay_id')['has_fever'].cumsum()

        return temp_df[['stay_id', 'hour_bin', 'temp_max_24h', 'temp_variability', 'fever_duration_hours']]

    def _compute_antibiotic_days(self, stay_ids: np.ndarray) -> pd.DataFrame:
        """Compute cumulative antibiotic exposure days."""
        logger.info("Loading prescriptions...")

        try:
            prescriptions_path = self._get_file_path("hosp", "prescriptions.csv")
            prescriptions = pd.read_csv(
                prescriptions_path,
                usecols=['subject_id', 'starttime', 'stoptime', 'drug']
            )
        except FileNotFoundError:
            logger.warning("prescriptions.csv not found")
            return pd.DataFrame({'stay_id': stay_ids, 'hour_bin': 0, 'antibiotic_days': 0})

        # Filter for antibiotics (common antibiotic keywords)
        antibiotic_keywords = ['cillin', 'mycin', 'cycline', 'floxacin', 'cef', 'vancomycin', 'meropenem']
        prescriptions = prescriptions[
            prescriptions['drug'].str.contains('|'.join(antibiotic_keywords), case=False, na=False)
        ]

        # Load ICU stays
        icustays_path = self._get_file_path("icu", "icustays.csv")
        icustays = pd.read_csv(
            icustays_path,
            usecols=['stay_id', 'subject_id', 'intime']
        )
        icustays = icustays[icustays['stay_id'].isin(stay_ids)]
        icustays['intime'] = pd.to_datetime(icustays['intime'])

        # Merge
        prescriptions = prescriptions.merge(icustays, on='subject_id', how='inner')
        prescriptions['starttime'] = pd.to_datetime(prescriptions['starttime'])
        prescriptions['stoptime'] = pd.to_datetime(prescriptions['stoptime'])

        # Calculate antibiotic days at each hour
        antibiotic_features = []
        for stay_id in stay_ids:
            stay_rx = prescriptions[prescriptions['stay_id'] == stay_id]
            if len(stay_rx) == 0:
                continue

            intime = stay_rx['intime'].iloc[0]

            # Calculate cumulative antibiotic days for each hour
            max_hours = 168  # 7 days
            for hour in range(max_hours):
                current_time = intime + pd.Timedelta(hours=hour)
                days_on_abx = 0

                for _, rx in stay_rx.iterrows():
                    if pd.notna(rx['starttime']) and rx['starttime'] <= current_time:
                        if pd.isna(rx['stoptime']) or rx['stoptime'] >= current_time:
                            days_on_abx += (current_time - rx['starttime']).total_seconds() / 86400

                antibiotic_features.append({
                    'stay_id': stay_id,
                    'hour_bin': hour,
                    'antibiotic_days': min(days_on_abx, 30)  # Cap at 30 days
                })

        return pd.DataFrame(antibiotic_features)

    def _compute_culture_negative_count(self, stay_ids: np.ndarray) -> pd.DataFrame:
        """Compute cumulative count of negative cultures."""
        logger.info("Loading microbiology events...")

        try:
            micro_path = self._get_file_path("hosp", "microbiologyevents.csv")
            micro = pd.read_csv(
                micro_path,
                usecols=['subject_id', 'chartdate', 'org_name']
            )
        except FileNotFoundError:
            logger.warning("microbiologyevents.csv not found")
            return pd.DataFrame({'stay_id': stay_ids, 'hour_bin': 0, 'culture_negative_count': 0})

        # Load ICU stays
        icustays_path = self._get_file_path("icu", "icustays.csv")
        icustays = pd.read_csv(
            icustays_path,
            usecols=['stay_id', 'subject_id', 'intime']
        )
        icustays = icustays[icustays['stay_id'].isin(stay_ids)]
        icustays['intime'] = pd.to_datetime(icustays['intime'])

        # Merge
        micro = micro.merge(icustays, on='subject_id', how='inner')
        micro['chartdate'] = pd.to_datetime(micro['chartdate'])

        # Identify negative cultures (org_name is null)
        negative_cultures = micro[micro['org_name'].isna()]

        # Calculate hour bin
        negative_cultures['hour_bin'] = ((negative_cultures['chartdate'] - negative_cultures['intime']).dt.total_seconds() / 3600).astype(int)
        negative_cultures = negative_cultures[negative_cultures['hour_bin'] >= 0]

        # Count cumulative negative cultures
        culture_counts = negative_cultures.groupby(['stay_id', 'hour_bin']).size().reset_index(name='culture_negative_count')
        culture_counts['culture_negative_count'] = culture_counts.groupby('stay_id')['culture_negative_count'].cumsum()

        return culture_counts

    def _compute_immunosuppressed(self, stay_ids: np.ndarray) -> pd.DataFrame:
        """Compute immunosuppressed status (steroids, chemo, immunosuppressants)."""
        # Simplified: binary indicator if on immunosuppressive meds
        # In practice, would check prescriptions for steroids, chemo, etc.
        return pd.DataFrame({
            'stay_id': stay_ids,
            'hour_bin': 0,
            'immunosuppressed': 0  # Placeholder
        })

    def _compute_weight_loss(self, stay_ids: np.ndarray) -> pd.DataFrame:
        """Compute weight loss (change from admission)."""
        # Simplified: would extract weight measurements from chartevents
        return pd.DataFrame({
            'stay_id': stay_ids,
            'hour_bin': 0,
            'weight_loss': 0  # Placeholder (kg)
        })

    def _compute_night_sweats_proxy(self, base_features: pd.DataFrame) -> pd.DataFrame:
        """Compute night sweats proxy (temperature spikes at night)."""
        if 'temperature' not in base_features.columns:
            return pd.DataFrame()

        temp_df = base_features[['stay_id', 'hour_bin', 'temperature']].copy()

        # Identify night hours (20:00 - 06:00, assuming hour_bin % 24 gives hour of day)
        temp_df['hour_of_day'] = temp_df['hour_bin'] % 24
        temp_df['is_night'] = ((temp_df['hour_of_day'] >= 20) | (temp_df['hour_of_day'] <= 6)).astype(int)

        # Night sweats proxy: temperature spike at night (> 38.5°C)
        temp_df['night_sweats_proxy'] = ((temp_df['is_night'] == 1) & (temp_df['temperature'] > 38.5)).astype(int)

        return temp_df[['stay_id', 'hour_bin', 'night_sweats_proxy']]

    def _compute_rash_documented(self, stay_ids: np.ndarray) -> pd.DataFrame:
        """Compute rash documentation (from clinical notes)."""
        # Simplified: would require NLP on clinical notes
        return pd.DataFrame({
            'stay_id': stay_ids,
            'hour_bin': 0,
            'rash_documented': 0  # Placeholder
        })


def main():
    """Main execution."""
    import sys

    if len(sys.argv) < 4:
        print("Usage: python feature_engineering_fou.py <mimic_dir> <cohort_path> <output_dir>")
        sys.exit(1)

    mimic_dir = sys.argv[1]
    cohort_path = sys.argv[2]
    output_dir = sys.argv[3]

    engineer = FouFeatureEngineer(mimic_dir, cohort_path, output_dir)
    features = engineer.extract_features()

    print(f"\n=== FOU Feature Summary ===")
    print(f"Total rows: {len(features)}")
    print(f"Features: {features.columns.tolist()}")
    print(f"\nMissing values:")
    print(features.isnull().sum())


if __name__ == "__main__":
    main()
