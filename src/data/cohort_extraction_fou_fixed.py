"""
FOU Cohort Extraction from MIMIC-IV - FIXED VERSION
====================================================
Extract Fever of Unknown Origin (FOU) cohort from MIMIC-IV v3.1.

FIXED: Now properly includes Class 0 (No FOU) samples for balanced 4-class classification.

Inclusion Criteria:
- ICU stay ≥ 3 days (configurable)
- For FOU classes (1-3): Fever > 38.0°C on ≥ 2 occasions
- For No FOU class (0): ICU stay without persistent fever

Labels:
- 0: No FOU (NEW: properly implemented)
- 1: Infectious FOU
- 2: Non-infectious FOU
- 3: Undiagnosed FOU
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FouCohortExtractorFixed:
    """Extract FOU cohort from MIMIC-IV with proper Class 0 (No FOU) inclusion."""

    def __init__(self, mimic_dir: str, output_dir: str, 
                 min_icu_days: int = 3, min_fever_temp: float = 38.0, 
                 min_fever_count: int = 2, include_no_fou: bool = True,
                 no_fou_sample_ratio: float = 0.3):
        self.mimic_dir = Path(mimic_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # FOU criteria
        self.fever_threshold = min_fever_temp  # °C
        self.min_fever_episodes = min_fever_count
        self.min_icu_stay_days = min_icu_days
        self.include_no_fou = include_no_fou
        self.no_fou_sample_ratio = no_fou_sample_ratio  # Ratio of No FOU to FOU samples

        # Temperature item IDs
        self.temp_item_ids = [223762, 226329]

        # ICD-10 codes for FOU categories
        self.infectious_icd = self._get_infectious_icd_codes()
        self.noninfectious_icd = self._get_noninfectious_icd_codes()
        
        logger.info(f"FOU Criteria: min_icu_days={min_icu_days}, min_fever_temp={min_fever_temp}°C, min_fever_count={min_fever_count}")
        logger.info(f"Include No FOU: {include_no_fou}, No FOU ratio: {no_fou_sample_ratio}")

    def _get_file_path(self, subdir: str, filename: str) -> Path:
        """Get file path, checking for .gz extension."""
        base_path = self.mimic_dir / subdir / filename
        if base_path.exists():
            return base_path
        gz_path = self.mimic_dir / subdir / f"{filename}.gz"
        if gz_path.exists():
            return gz_path
        raise FileNotFoundError(f"Neither {base_path} nor {gz_path} found")

    def _get_infectious_icd_codes(self) -> List[str]:
        """Get ICD-10 codes for infectious FOU causes."""
        return [
            # Tuberculosis
            'A15', 'A16', 'A17', 'A18', 'A19',
            # Endocarditis
            'I33', 'I38', 'I39',
            # Abscess
            'K65', 'K75.0', 'L02', 'J85',
            # Fungal infections
            'B37', 'B38', 'B39', 'B40', 'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47', 'B48',
            # Other infections
            'A40', 'A41',  # Sepsis
            'B20', 'B21', 'B22', 'B23', 'B24',  # HIV
        ]

    def _get_noninfectious_icd_codes(self) -> List[str]:
        """Get ICD-10 codes for non-infectious FOU causes."""
        return [
            # Autoimmune/inflammatory
            'M05', 'M06', 'M32', 'M33', 'M34', 'M35',  # Rheumatoid, lupus, etc.
            'M30', 'M31',  # Vasculitis
            # Malignancy
            'C81', 'C82', 'C83', 'C84', 'C85',  # Lymphoma
            'C91', 'C92', 'C93', 'C94', 'C95',  # Leukemia
            # Drug fever
            'T88.3',
            # Thromboembolism
            'I26', 'I80', 'I81', 'I82',
        ]

    def extract_cohort(self) -> pd.DataFrame:
        """Extract FOU cohort from MIMIC-IV with proper Class 0 inclusion."""
        logger.info("Starting FOU cohort extraction (FIXED VERSION)...")

        # Step 1: Load ICU stays
        logger.info("Loading ICU stays...")
        icustays_path = self._get_file_path("icu", "icustays.csv")
        icustays = pd.read_csv(icustays_path)
        logger.info(f"Total ICU stays: {len(icustays)}")

        # Step 2: Filter by ICU stay duration
        icustays['intime'] = pd.to_datetime(icustays['intime'])
        icustays['outtime'] = pd.to_datetime(icustays['outtime'])
        icustays['los_days'] = (icustays['outtime'] - icustays['intime']).dt.total_seconds() / 86400

        long_stays = icustays[icustays['los_days'] >= self.min_icu_stay_days].copy()
        logger.info(f"ICU stays ≥ {self.min_icu_stay_days} days: {len(long_stays)}")

        # Step 3: Extract temperature measurements
        logger.info("Extracting temperature measurements...")
        temps = self._extract_temperatures(long_stays['stay_id'].unique())

        # Step 4: Identify fever episodes
        logger.info("Identifying fever episodes...")
        fever_stays, no_fever_stays = self._identify_fever_episodes(temps, long_stays)
        logger.info(f"Stays with ≥{self.min_fever_episodes} fever episodes: {len(fever_stays)}")
        logger.info(f"Stays WITHOUT persistent fever: {len(no_fever_stays)}")

        # Step 5: Process FOU cases (Classes 1-3)
        logger.info("Processing FOU cases (Classes 1-3)...")
        fou_cohort = self._process_fou_cases(fever_stays)
        logger.info(f"FOU cases extracted: {len(fou_cohort)}")

        # Step 6: Process No FOU cases (Class 0) - NEW!
        if self.include_no_fou:
            logger.info("Processing No FOU cases (Class 0)...")
            no_fou_cohort = self._process_no_fou_cases(no_fever_stays, target_count=int(len(fou_cohort) * self.no_fou_sample_ratio))
            logger.info(f"No FOU cases extracted: {len(no_fou_cohort)}")

            # Combine FOU and No FOU cases
            cohort = pd.concat([fou_cohort, no_fou_cohort], ignore_index=True)
        else:
            cohort = fou_cohort

        logger.info(f"Final cohort size: {len(cohort)}")
        logger.info(f"Label distribution:\n{cohort['fou_label'].value_counts().sort_index()}")

        # Step 7: Add patient demographics
        logger.info("Adding patient demographics...")
        cohort = self._add_demographics(cohort)

        # Save cohort
        output_path = self.output_dir / "fou_cohort.csv"
        cohort.to_csv(output_path, index=False)
        logger.info(f"Cohort saved to {output_path}")

        return cohort

    def _extract_temperatures(self, stay_ids: np.ndarray) -> pd.DataFrame:
        """Extract temperature measurements for given stays."""
        logger.info("Loading chartevents (temperature)...")

        temps_list = []
        chunksize = 10_000_000
        chartevents_path = self._get_file_path("icu", "chartevents.csv")

        for chunk in pd.read_csv(
            chartevents_path,
            usecols=['stay_id', 'charttime', 'itemid', 'valuenum'],
            chunksize=chunksize
        ):
            chunk_temps = chunk[
                (chunk['itemid'].isin(self.temp_item_ids)) &
                (chunk['stay_id'].isin(stay_ids))
            ].copy()

            if len(chunk_temps) > 0:
                temps_list.append(chunk_temps)

        if not temps_list:
            logger.warning("No temperature measurements found!")
            return pd.DataFrame()

        temps = pd.concat(temps_list, ignore_index=True)
        temps['charttime'] = pd.to_datetime(temps['charttime'])

        # Convert Fahrenheit to Celsius if needed
        temps.loc[temps['valuenum'] > 50, 'valuenum'] = (temps['valuenum'] - 32) * 5/9

        # Filter valid temperature range (30-45°C)
        temps = temps[(temps['valuenum'] >= 30) & (temps['valuenum'] <= 45)]

        logger.info(f"Temperature measurements extracted: {len(temps)}")
        return temps

    def _identify_fever_episodes(self, temps: pd.DataFrame, icustays: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify stays with and without persistent fever.
        
        Returns:
            fever_stays: Stays with ≥min_fever_episodes fever episodes (FOU candidates)
            no_fever_stays: Stays without persistent fever (No FOU candidates)
        """
        if temps.empty:
            return pd.DataFrame(), icustays

        # Merge with ICU stay times
        temps = temps.merge(
            icustays[['stay_id', 'intime', 'outtime']],
            on='stay_id',
            how='inner'
        )

        # Filter temperatures within ICU stay
        temps = temps[
            (temps['charttime'] >= temps['intime']) &
            (temps['charttime'] <= temps['outtime'])
        ]

        # Identify fever episodes
        fever_temps = temps[temps['valuenum'] > self.fever_threshold]

        # Count fever episodes per stay (6-hour windows)
        fever_temps['hour_bin'] = (fever_temps['charttime'] - fever_temps['intime']).dt.total_seconds() / 3600 // 6
        fever_counts = fever_temps.groupby('stay_id')['hour_bin'].nunique().reset_index()
        fever_counts.columns = ['stay_id', 'fever_episode_count']

        # Split into fever and no-fever stays
        fever_stays = icustays.merge(fever_counts, on='stay_id', how='inner')
        fever_stays = fever_stays[fever_stays['fever_episode_count'] >= self.min_fever_episodes]

        # No fever stays: either no fever episodes or < min_fever_episodes
        fever_stay_ids = set(fever_stays['stay_id'])
        no_fever_stays = icustays[~icustays['stay_id'].isin(fever_stay_ids)]

        return fever_stays, no_fever_stays

    def _process_fou_cases(self, fever_stays: pd.DataFrame) -> pd.DataFrame:
        """Process FOU cases (Classes 1-3)."""
        # Check for negative initial cultures
        logger.info("Checking initial cultures...")
        negative_culture_stays = self._check_negative_cultures(fever_stays['stay_id'].unique())
        logger.info(f"Stays with negative initial cultures: {len(negative_culture_stays)}")

        # Exclude obvious infections
        logger.info("Excluding obvious infections...")
        fou_candidates = self._exclude_obvious_infections(negative_culture_stays)
        logger.info(f"FOU candidates (no obvious infection): {len(fou_candidates)}")

        # Label FOU categories (1, 2, or 3)
        logger.info("Labeling FOU categories...")
        fou_cohort = self._label_fou_categories(fou_candidates)

        return fou_cohort

    def _process_no_fou_cases(self, no_fever_stays: pd.DataFrame, target_count: int) -> pd.DataFrame:
        """
        Process No FOU cases (Class 0).
        
        Criteria for No FOU:
        - ICU stay ≥ min_icu_days
        - No persistent fever (< min_fever_episodes)
        - No FOU-related diagnoses
        """
        if len(no_fever_stays) == 0:
            logger.warning("No stays without persistent fever found!")
            return pd.DataFrame()

        # Exclude patients with FOU-related diagnoses
        logger.info("Excluding FOU-related diagnoses from No FOU candidates...")
        no_fou_candidates = self._exclude_fou_diagnoses(no_fever_stays)
        logger.info(f"No FOU candidates after diagnosis exclusion: {len(no_fou_candidates)}")

        # Sample to target count
        if len(no_fou_candidates) > target_count:
            no_fou_candidates = no_fou_candidates.sample(n=target_count, random_state=42)
            logger.info(f"Sampled {target_count} No FOU cases")

        # Label as Class 0
        no_fou_candidates['fou_label'] = 0

        return no_fou_candidates

    def _exclude_fou_diagnoses(self, stays: pd.DataFrame) -> pd.DataFrame:
        """Exclude stays with FOU-related diagnoses."""
        try:
            diagnoses_path = self._get_file_path("hosp", "diagnoses_icd.csv")
            diagnoses = pd.read_csv(
                diagnoses_path,
                usecols=['subject_id', 'icd_code']
            )
        except FileNotFoundError:
            logger.warning("diagnoses_icd.csv not found, cannot exclude FOU diagnoses")
            return stays

        # Combine all FOU-related ICD codes
        fou_related_codes = self.infectious_icd + self.noninfectious_icd

        # Find patients with FOU-related diagnoses
        fou_diagnoses = diagnoses[
            diagnoses['icd_code'].str.startswith(tuple(fou_related_codes), na=False)
        ]

        # Exclude these patients
        no_fou_stays = stays[~stays['subject_id'].isin(fou_diagnoses['subject_id'])]

        return no_fou_stays

    def _check_negative_cultures(self, stay_ids: np.ndarray) -> pd.DataFrame:
        """Check for negative initial cultures (first 48 hours)."""
        logger.info("Loading microbiology events...")

        try:
            micro_path = self._get_file_path("hosp", "microbiologyevents.csv")
            micro = pd.read_csv(
                micro_path,
                usecols=['subject_id', 'chartdate', 'org_name']
            )
        except FileNotFoundError:
            logger.warning("microbiologyevents.csv not found, skipping culture check")
            return pd.DataFrame({'stay_id': stay_ids})

        micro['chartdate'] = pd.to_datetime(micro['chartdate'])

        # Load ICU stays
        icustays_path = self._get_file_path("icu", "icustays.csv")
        icustays = pd.read_csv(
            icustays_path,
            usecols=['stay_id', 'subject_id', 'intime']
        )
        icustays['intime'] = pd.to_datetime(icustays['intime'])
        icustays = icustays[icustays['stay_id'].isin(stay_ids)]

        # Merge cultures with ICU stays
        micro = micro.merge(icustays[['subject_id', 'stay_id', 'intime']], on='subject_id', how='inner')

        # Filter cultures in first 48 hours
        micro['hours_from_admission'] = (micro['chartdate'] - micro['intime']).dt.total_seconds() / 3600
        initial_cultures = micro[micro['hours_from_admission'] <= 48]

        # Identify stays with positive cultures
        positive_culture_stays = initial_cultures[initial_cultures['org_name'].notna()]['stay_id'].unique()

        # Return stays with negative initial cultures
        negative_stays = icustays[~icustays['stay_id'].isin(positive_culture_stays)]

        return negative_stays

    def _exclude_obvious_infections(self, stays: pd.DataFrame) -> pd.DataFrame:
        """Exclude stays with obvious infection diagnoses."""
        logger.info("Loading diagnoses...")

        try:
            diagnoses_path = self._get_file_path("hosp", "diagnoses_icd.csv")
            diagnoses = pd.read_csv(
                diagnoses_path,
                usecols=['subject_id', 'icd_code']
            )
        except FileNotFoundError:
            logger.warning("diagnoses_icd.csv not found, skipping obvious infection exclusion")
            return stays

        # Obvious infection ICD codes
        obvious_infection_codes = [
            'J18', 'J15', 'J13', 'J14',  # Pneumonia
            'N39.0', 'N30',  # UTI
            'T81.4', 'T81.40', 'T81.41',  # Surgical site infection
        ]

        # Filter for obvious infections
        obvious_infections = diagnoses[
            diagnoses['icd_code'].str.startswith(tuple(obvious_infection_codes), na=False)
        ]

        # Exclude these stays
        excluded_stays = stays[~stays['subject_id'].isin(obvious_infections['subject_id'])]

        return excluded_stays

    def _label_fou_categories(self, stays: pd.DataFrame) -> pd.DataFrame:
        """Label FOU categories (1, 2, or 3) based on final diagnoses."""
        logger.info("Loading final diagnoses for labeling...")

        try:
            diagnoses_path = self._get_file_path("hosp", "diagnoses_icd.csv")
            diagnoses = pd.read_csv(
                diagnoses_path,
                usecols=['subject_id', 'icd_code']
            )
        except FileNotFoundError:
            logger.warning("diagnoses_icd.csv not found, labeling all as undiagnosed")
            stays['fou_label'] = 3  # Undiagnosed
            return stays

        # Merge with diagnoses
        cohort = stays.merge(diagnoses, on='subject_id', how='left')

        # Label based on ICD codes
        def assign_label(row):
            if pd.isna(row['icd_code']):
                return 3  # Undiagnosed

            icd = str(row['icd_code'])

            # Check infectious causes
            for code in self.infectious_icd:
                if icd.startswith(code):
                    return 1  # Infectious FOU

            # Check non-infectious causes
            for code in self.noninfectious_icd:
                if icd.startswith(code):
                    return 2  # Non-infectious FOU

            # Default: undiagnosed
            return 3

        cohort['fou_label'] = cohort.apply(assign_label, axis=1)

        # Take the most specific label per stay
        cohort = cohort.sort_values('fou_label').groupby('stay_id').first().reset_index()

        return cohort

    def _add_demographics(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Add patient demographics."""
        logger.info("Loading patient demographics...")

        try:
            patients_path = self._get_file_path("hosp", "patients.csv")
            patients = pd.read_csv(
                patients_path,
                usecols=['subject_id', 'gender', 'anchor_age']
            )

            cohort = cohort.merge(patients, on='subject_id', how='left')
        except FileNotFoundError:
            logger.warning("patients.csv not found, skipping demographics")

        return cohort


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract FOU cohort from MIMIC-IV (FIXED VERSION)")
    parser.add_argument('mimic_dir', type=str, help='Path to MIMIC-IV data directory')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('--min-icu-days', type=int, default=3, help='Minimum ICU stay days (default: 3)')
    parser.add_argument('--min-fever-temp', type=float, default=38.0, help='Minimum fever temperature in °C (default: 38.0)')
    parser.add_argument('--min-fever-count', type=int, default=2, help='Minimum fever episode count (default: 2)')
    parser.add_argument('--include-no-fou', action='store_true', default=True, help='Include "No FOU" class (default: True)')
    parser.add_argument('--no-fou-ratio', type=float, default=0.3, help='Ratio of No FOU to FOU samples (default: 0.3)')
    
    args = parser.parse_args()

    extractor = FouCohortExtractorFixed(
        args.mimic_dir, 
        args.output_dir,
        min_icu_days=args.min_icu_days,
        min_fever_temp=args.min_fever_temp,
        min_fever_count=args.min_fever_count,
        include_no_fou=args.include_no_fou,
        no_fou_sample_ratio=args.no_fou_ratio
    )
    cohort = extractor.extract_cohort()

    print(f"\n=== FOU Cohort Summary (FIXED) ===")
    print(f"Total cases: {len(cohort)}")
    print(f"\nLabel distribution:")
    label_counts = cohort['fou_label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = ['No FOU', 'Infectious FOU', 'Non-infectious FOU', 'Undiagnosed FOU'][label]
        print(f"  Class {label} ({label_name}): {count} ({count/len(cohort)*100:.1f}%)")
    
    if 'gender' in cohort.columns:
        print(f"\nGender distribution:")
        print(cohort['gender'].value_counts())
    
    if 'anchor_age' in cohort.columns:
        print(f"\nAge statistics:")
        print(cohort['anchor_age'].describe())


if __name__ == "__main__":
    main()
