"""
Fever Etiology Cohort Extraction for MIMIC-IV
Adapted from FUO paper methodology for ICU setting

This extracts a cohort of febrile ICU patients and classifies them into
a hierarchical etiology structure inspired by the FUO paper:

Root
├── Infectious Disease
│   ├── Bacterial Infection
│   ├── Viral Infection
│   └── Fungal Infection
└── Noninfectious Disease
    ├── Neoplastic Disease (malignancy)
    ├── Autoimmune/Inflammatory
    └── Drug-Induced Fever

Note: This is NOT classical FUO (which requires >3 weeks fever + 1 week workup).
This is "Early Fever Etiology Classification in ICU" - a related but distinct problem.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeverCohortExtractor:
    """Extract febrile ICU patients and label by discharge diagnosis etiology"""
    
    # ICD-10 code mappings to etiology hierarchy
    # Based on FUO paper's classification + clinical guidelines
    ETIOLOGY_MAPPING = {
        'bacterial': {
            'icd10_prefixes': ['A00', 'A01', 'A02', 'A03', 'A04', 'A05',  # Intestinal infections
                               'A20', 'A21', 'A22', 'A23', 'A24', 'A25',  # Zoonotic bacterial
                               'A30', 'A31', 'A32', 'A33', 'A34', 'A35',  # Mycobacterial
                               'A36', 'A37', 'A38', 'A39', 'A40', 'A41',  # Sepsis/meningococcal
                               'A48', 'A49',  # Other bacterial
                               'J13', 'J14', 'J15', 'J16',  # Bacterial pneumonia
                               'N10', 'N12', 'N30', 'N39.0'],  # UTI
            'description': 'Bacterial Infection'
        },
        'viral': {
            'icd10_prefixes': ['A80', 'A81', 'A82', 'A83', 'A84', 'A85',  # Viral CNS
                               'A87', 'A88', 'A89',  # Other viral
                               'B00', 'B01', 'B02', 'B05', 'B06',  # Herpes, measles, rubella
                               'B15', 'B16', 'B17', 'B18', 'B19',  # Viral hepatitis
                               'B20', 'B21', 'B22', 'B23', 'B24',  # HIV
                               'B25', 'B26', 'B27',  # CMV, mumps, EBV
                               'B33', 'B34',  # Other viral
                               'J09', 'J10', 'J11', 'J12.0', 'J12.1'],  # Influenza, viral pneumonia
            'description': 'Viral Infection'
        },
        'fungal': {
            'icd10_prefixes': ['B35', 'B36', 'B37', 'B38', 'B39',  # Mycoses
                               'B40', 'B41', 'B42', 'B43', 'B44',  # Systemic mycoses
                               'B45', 'B46', 'B47', 'B48', 'B49',  # Other mycoses
                               'J17.2'],  # Fungal pneumonia
            'description': 'Fungal Infection'
        },
        'neoplastic': {
            'icd10_prefixes': ['C00', 'C01', 'C02', 'C03', 'C04', 'C05',  # Malignant neoplasms (all C codes)
                               'C81', 'C82', 'C83', 'C84', 'C85',  # Lymphomas
                               'C90', 'C91', 'C92', 'C93', 'C94', 'C95'],  # Leukemias
            'description': 'Neoplastic Disease'
        },
        'autoimmune': {
            'icd10_prefixes': ['M05', 'M06',  # Rheumatoid arthritis
                               'M30', 'M31', 'M32', 'M33', 'M34', 'M35',  # Systemic connective tissue disorders
                               'M45', 'M46',  # Spondylopathies
                               'K50', 'K51',  # Crohn's, ulcerative colitis
                               'D86'],  # Sarcoidosis
            'description': 'Autoimmune/Inflammatory Disease'
        },
        'drug_fever': {
            'icd10_prefixes': ['T36', 'T37', 'T38', 'T39',  # Drug poisoning
                               'T40', 'T41', 'T42', 'T43',
                               'T44', 'T45', 'T46', 'T47',
                               'T48', 'T49', 'T50'],
            'description': 'Drug-Induced Fever'
        }
    }
    
    # Temperature item IDs in MIMIC-IV chartevents
    TEMP_ITEMIDS = [223762, 226329]  # Fahrenheit and Celsius
    
    # Fever threshold (Celsius)
    FEVER_THRESHOLD_C = 38.3
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_table_chunked(self, table_name: str, usecols: List[str] = None,
                          filter_col: str = None, filter_values: set = None,
                          chunksize: int = 500_000) -> pd.DataFrame:
        """
        Load large MIMIC-IV tables in chunks to avoid OOM
        Reuses the chunked loading strategy from sepsis pipeline
        """
        filepath = self.data_dir / f"{table_name}.csv.gz"
        if not filepath.exists():
            filepath = self.data_dir / f"{table_name}.csv"
        
        logger.info(f"Loading {table_name} in chunks...")
        chunks = []
        
        reader = pd.read_csv(filepath, usecols=usecols, chunksize=chunksize)
        for i, chunk in enumerate(reader):
            if filter_col and filter_values:
                chunk = chunk[chunk[filter_col].isin(filter_values)]
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {(i+1)*chunksize:,} rows...")
        
        result = pd.concat(chunks, ignore_index=True)
        logger.info(f"  Loaded {len(result):,} rows")
        return result
    
    def extract_febrile_patients(self) -> pd.DataFrame:
        """
        Step 1: Identify ICU patients with fever >38.3°C in first 48 hours
        """
        logger.info("Step 1: Extracting febrile ICU patients...")
        
        # Load ICU stays
        icustays = pd.read_csv(self.data_dir / "icu" / "icustays.csv.gz")
        logger.info(f"Total ICU stays: {len(icustays):,}")
        
        # Convert times
        icustays['intime'] = pd.to_datetime(icustays['intime'])
        icustays['outtime'] = pd.to_datetime(icustays['outtime'])
        
        # Load temperature measurements (chunked)
        stay_ids = set(icustays['stay_id'].unique())
        temps = self.load_table_chunked(
            'icu/chartevents',
            usecols=['stay_id', 'charttime', 'itemid', 'valuenum'],
            filter_col='stay_id',
            filter_values=stay_ids
        )
        
        # Filter to temperature items only
        temps = temps[temps['itemid'].isin(self.TEMP_ITEMIDS)]
        temps['charttime'] = pd.to_datetime(temps['charttime'])
        
        # Merge with ICU stays to get admission time
        temps = temps.merge(icustays[['stay_id', 'intime']], on='stay_id')
        
        # Keep only first 48 hours
        temps['hours_since_admission'] = (temps['charttime'] - temps['intime']).dt.total_seconds() / 3600
        temps = temps[temps['hours_since_admission'] <= 48]
        
        # Convert Fahrenheit to Celsius if needed (itemid 223762 is Fahrenheit)
        temps.loc[temps['itemid'] == 223762, 'valuenum'] = (temps['valuenum'] - 32) * 5/9
        
        # Find patients with fever >38.3°C
        febrile_stays = temps[temps['valuenum'] > self.FEVER_THRESHOLD_C]['stay_id'].unique()
        logger.info(f"Febrile ICU stays (>38.3°C in first 48h): {len(febrile_stays):,}")
        
        # Filter cohort
        cohort = icustays[icustays['stay_id'].isin(febrile_stays)].copy()
        
        return cohort
    
    def exclude_obvious_sepsis(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Exclude patients who meet Sepsis-3 criteria in first 24h
        (These are handled by the sepsis pipeline, not fever etiology classification)
        """
        logger.info("Step 2: Excluding obvious sepsis cases...")
        
        # Load prescriptions to check for antibiotics + cultures (suspected infection)
        stay_ids = set(cohort['stay_id'].unique())
        prescriptions = self.load_table_chunked(
            'hosp/prescriptions',
            usecols=['hadm_id', 'starttime', 'drug'],
            filter_col='hadm_id',
            filter_values=set(cohort['hadm_id'].unique())
        )
        
        # Simple antibiotic detection (extend this list as needed)
        antibiotic_keywords = ['vancomycin', 'cefepime', 'meropenem', 'piperacillin',
                               'levofloxacin', 'azithromycin', 'ceftriaxone']
        prescriptions['is_antibiotic'] = prescriptions['drug'].str.lower().str.contains(
            '|'.join(antibiotic_keywords), na=False
        )
        
        # Find admissions with early antibiotic orders
        early_abx = prescriptions[prescriptions['is_antibiotic']].groupby('hadm_id').size()
        sepsis_suspect_hadm = set(early_abx[early_abx >= 2].index)  # At least 2 antibiotic orders
        
        logger.info(f"Suspected sepsis admissions (early antibiotics): {len(sepsis_suspect_hadm):,}")
        
        # Exclude from cohort
        cohort = cohort[~cohort['hadm_id'].isin(sepsis_suspect_hadm)].copy()
        logger.info(f"Remaining after sepsis exclusion: {len(cohort):,}")
        
        return cohort
    
    def assign_etiology_labels(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Assign hierarchical etiology labels based on discharge diagnoses
        """
        logger.info("Step 3: Assigning etiology labels from discharge diagnoses...")
        
        # Load diagnosis codes
        hadm_ids = set(cohort['hadm_id'].unique())
        diagnoses = self.load_table_chunked(
            'hosp/diagnoses_icd',
            usecols=['hadm_id', 'icd_code', 'icd_version', 'seq_num'],
            filter_col='hadm_id',
            filter_values=hadm_ids
        )
        
        # Focus on ICD-10 (version 10) and primary diagnoses (seq_num <= 5)
        diagnoses = diagnoses[(diagnoses['icd_version'] == 10) & (diagnoses['seq_num'] <= 5)]
        
        # Map each admission to etiology
        def map_etiology(hadm_id: int) -> Dict[str, any]:
            hadm_dx = diagnoses[diagnoses['hadm_id'] == hadm_id]['icd_code'].tolist()
            
            # Check each etiology category
            for etiology, config in self.ETIOLOGY_MAPPING.items():
                for prefix in config['icd10_prefixes']:
                    if any(code.startswith(prefix) for code in hadm_dx):
                        # Determine parent category
                        if etiology in ['bacterial', 'viral', 'fungal']:
                            parent = 'infectious'
                        else:
                            parent = 'noninfectious'
                        
                        return {
                            'etiology_fine': etiology,
                            'etiology_coarse': parent,
                            'etiology_description': config['description']
                        }
            
            # No match found
            return {
                'etiology_fine': 'unknown',
                'etiology_coarse': 'unknown',
                'etiology_description': 'Unknown Etiology'
            }
        
        # Apply mapping
        etiology_data = cohort['hadm_id'].apply(map_etiology).apply(pd.Series)
        cohort = pd.concat([cohort, etiology_data], axis=1)
        
        # Log distribution
        logger.info("\nEtiology Distribution:")
        logger.info(cohort['etiology_coarse'].value_counts())
        logger.info("\nFine-Grained Etiology:")
        logger.info(cohort['etiology_fine'].value_counts())
        
        # Exclude unknown etiologies (can't train on these)
        cohort = cohort[cohort['etiology_fine'] != 'unknown'].copy()
        logger.info(f"\nFinal cohort size (after excluding unknown): {len(cohort):,}")
        
        return cohort
    
    def add_patient_demographics(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Add patient demographics (age, gender)
        """
        logger.info("Step 4: Adding patient demographics...")
        
        # Load patients and admissions
        patients = pd.read_csv(self.data_dir / "hosp" / "patients.csv.gz")
        admissions = pd.read_csv(self.data_dir / "hosp" / "admissions.csv.gz")
        
        # Merge
        cohort = cohort.merge(
            admissions[['hadm_id', 'admittime', 'dischtime', 'admission_type', 'admission_location']],
            on='hadm_id',
            how='left'
        )
        cohort = cohort.merge(
            patients[['subject_id', 'gender', 'anchor_age', 'anchor_year_group']],
            on='subject_id',
            how='left'
        )
        
        # Rename for consistency
        cohort.rename(columns={'anchor_age': 'age'}, inplace=True)
        
        return cohort
    
    def run(self, output_path: str = None) -> pd.DataFrame:
        """
        Full pipeline: extract febrile cohort with etiology labels
        """
        logger.info("="*80)
        logger.info("FEVER ETIOLOGY COHORT EXTRACTION")
        logger.info("="*80)
        
        # Step 1: Find febrile patients
        cohort = self.extract_febrile_patients()
        
        # Step 2: Exclude sepsis
        cohort = self.exclude_obvious_sepsis(cohort)
        
        # Step 3: Assign etiology labels
        cohort = self.assign_etiology_labels(cohort)
        
        # Step 4: Add demographics
        cohort = self.add_patient_demographics(cohort)
        
        # Save
        if output_path is None:
            output_path = "data/processed/fever_cohort.csv"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cohort.to_csv(output_path, index=False)
        logger.info(f"\nSaved cohort to: {output_path}")
        
        # Summary statistics
        logger.info("\n" + "="*80)
        logger.info("COHORT SUMMARY")
        logger.info("="*80)
        logger.info(f"Total ICU stays: {len(cohort):,}")
        logger.info(f"Unique patients: {cohort['subject_id'].nunique():,}")
        logger.info(f"Unique admissions: {cohort['hadm_id'].nunique():,}")
        logger.info(f"\nAge: {cohort['age'].mean():.1f} ± {cohort['age'].std():.1f} years")
        logger.info(f"Gender: {cohort['gender'].value_counts().to_dict()}")
        logger.info(f"\nCoarse Etiology Distribution:")
        for etiology, count in cohort['etiology_coarse'].value_counts().items():
            pct = 100 * count / len(cohort)
            logger.info(f"  {etiology}: {count:,} ({pct:.1f}%)")
        
        return cohort


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract fever etiology cohort from MIMIC-IV")
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to MIMIC-IV data directory')
    parser.add_argument('--output', type=str, default='data/processed/fever_cohort.csv',
                       help='Output path for cohort CSV')
    
    args = parser.parse_args()
    
    extractor = FeverCohortExtractor(args.data_dir)
    cohort = extractor.run(args.output)
    
    print(f"\n✅ Cohort extraction complete: {len(cohort):,} patients")


if __name__ == '__main__':
    main()
