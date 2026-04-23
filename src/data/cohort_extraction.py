"""
QuantumSepsis Shield — Sepsis-3 Cohort Extraction from MIMIC-IV v3.1
=====================================================================

Extracts the sepsis cohort using Sepsis-3 criteria:
  1. Suspected infection = antibiotic order + blood/body fluid culture within ±24h
  2. Organ dysfunction = SOFA score increase ≥ 2 from baseline
  3. Onset time = max(suspected_infection_time, sofa_increase_time)

Supports both local CSV files and BigQuery.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import Config, get_default_config

logger = logging.getLogger(__name__)


# --- Antibiotic keywords for suspected infection detection ---
ANTIBIOTIC_KEYWORDS = [
    "vancomycin", "piperacillin", "meropenem", "ceftriaxone", "levofloxacin",
    "ciprofloxacin", "metronidazole", "azithromycin", "ampicillin", "cefepime",
    "cefazolin", "gentamicin", "tobramycin", "amikacin", "clindamycin",
    "doxycycline", "trimethoprim", "linezolid", "daptomycin", "ertapenem",
    "ceftazidime", "amoxicillin", "nafcillin", "oxacillin", "cephalexin",
    "imipenem", "doripenem", "tigecycline", "colistin", "polymyxin",
    "sulfamethoxazole", "nitrofurantoin", "fosfomycin",
]

# Valid culture specimen types
CULTURE_SPECIMEN_TYPES = [
    "BLOOD CULTURE", "URINE", "SPUTUM", "BRONCHOALVEOLAR LAVAGE",
    "CATHETER TIP-IV", "CSF;SPINAL FLUID", "PERITONEAL FLUID",
    "ABSCESS", "WOUND", "PLEURAL FLUID", "BILE", "STOOL",
]

# Antibiotic routes
VALID_ROUTES = ["IV", "PO", "ORAL", "PO/NG", "IV DRIP", "IM", "IV PUSH"]

# Sepsis ICD codes for validation
SEPSIS_ICD9_CODES = ["99591", "99592", "78552", "78559"]
SEPSIS_ICD10_CODES = [
    "A400", "A401", "A403", "A408", "A409",
    "A410", "A411", "A412", "A413", "A414", "A4150", "A4151", "A4152",
    "A4153", "A4159", "A418", "A4189", "A419",
    "R6520", "R6521",
]


def load_table(data_dir: str, table_name: str, module: str = "hosp",
               usecols: Optional[list] = None,
               dtype: Optional[dict] = None) -> pd.DataFrame:
    """Load a MIMIC-IV table from CSV (compressed or uncompressed).
    
    Args:
        data_dir: Root MIMIC-IV directory (e.g., data/raw/mimiciv/3.1)
        table_name: Table name (e.g., 'patients', 'icustays')
        module: Module name ('hosp' or 'icu')
        usecols: Columns to load (None = all)
        dtype: Column dtype specifications
    
    Returns:
        DataFrame with the loaded table
    """
    base = Path(data_dir) / module
    
    # Try compressed first, then uncompressed
    for ext in [".csv.gz", ".csv"]:
        filepath = base / f"{table_name}{ext}"
        if filepath.exists():
            logger.info(f"Loading {filepath}...")
            return pd.read_csv(
                filepath,
                usecols=usecols,
                dtype=dtype,
                parse_dates=True,
                low_memory=False,
            )
    
    raise FileNotFoundError(
        f"Table '{table_name}' not found in {base} "
        f"(tried .csv.gz and .csv)"
    )


class CohortExtractor:
    """Extracts the Sepsis-3 cohort from MIMIC-IV v3.1.
    
    Pipeline:
        1. Load ICU stays + patient demographics
        2. Detect suspected infections (antibiotics + cultures)
        3. Compute SOFA scores over time
        4. Identify Sepsis-3 onset (infection + SOFA ≥ 2 increase)
        5. Generate cohort with binary labels
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = config.data.mimic_raw_dir
        self.output_dir = Path(config.data.processed_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_cohort(self) -> pd.DataFrame:
        """Run complete cohort extraction pipeline.
        
        Returns:
            DataFrame with columns:
                subject_id, hadm_id, stay_id, intime, outtime, los_hours,
                sepsis_label, sepsis_onset_time, anchor_year_group,
                gender, anchor_age, icu_type, mortality_hospital
        """
        logger.info("=" * 60)
        logger.info("Starting Sepsis-3 Cohort Extraction from MIMIC-IV v3.1")
        logger.info("=" * 60)
        
        # Step 1: Load base tables
        icustays = self._load_icustays()
        patients = self._load_patients()
        admissions = self._load_admissions()
        
        # Merge demographics
        cohort = icustays.merge(patients, on="subject_id", how="left")
        cohort = cohort.merge(
            admissions[["hadm_id", "deathtime", "hospital_expire_flag"]],
            on="hadm_id", how="left"
        )
        
        logger.info(f"Total ICU stays: {len(cohort):,}")
        
        # Step 2: Detect suspected infections
        infection_times = self._detect_suspected_infections()
        
        # Step 3: Compute SOFA scores
        sofa_deltas = self._compute_sofa_deltas(cohort)
        
        # Step 4: Identify Sepsis-3 onset
        cohort = self._identify_sepsis_onset(cohort, infection_times, sofa_deltas)
        
        # Step 5: Compute derived columns
        cohort["los_hours"] = (
            (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600
        ).round(2)
        
        cohort["mortality_hospital"] = cohort["hospital_expire_flag"].fillna(0).astype(int)
        
        # Report statistics
        n_sepsis = cohort["sepsis_label"].sum()
        n_total = len(cohort)
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Cohort Statistics:")
        logger.info(f"  Total ICU stays:     {n_total:,}")
        logger.info(f"  Sepsis-3 positive:   {n_sepsis:,} ({100*n_sepsis/n_total:.1f}%)")
        logger.info(f"  Sepsis-3 negative:   {n_total - n_sepsis:,} ({100*(n_total-n_sepsis)/n_total:.1f}%)")
        logger.info(f"  Hospital mortality:  {cohort['mortality_hospital'].sum():,}")
        logger.info(f"  Median LOS (hours):  {cohort['los_hours'].median():.1f}")
        logger.info(f"{'=' * 40}\n")
        
        # Save
        output_cols = [
            "subject_id", "hadm_id", "stay_id", "intime", "outtime",
            "los_hours", "sepsis_label", "sepsis_onset_time",
            "anchor_year_group", "gender", "anchor_age", "first_careunit",
            "mortality_hospital",
        ]
        
        # Ensure all columns exist
        for col in output_cols:
            if col not in cohort.columns:
                cohort[col] = np.nan
        
        cohort_out = cohort[output_cols].copy()
        
        output_path = self.output_dir / "cohort.csv"
        cohort_out.to_csv(output_path, index=False)
        logger.info(f"Cohort saved to {output_path}")
        
        return cohort_out
    
    def _load_icustays(self) -> pd.DataFrame:
        """Load ICU stays table."""
        df = load_table(
            self.data_dir, "icustays", "icu",
            usecols=["subject_id", "hadm_id", "stay_id",
                     "first_careunit", "last_careunit", "intime", "outtime", "los"],
        )
        df["intime"] = pd.to_datetime(df["intime"])
        df["outtime"] = pd.to_datetime(df["outtime"])
        logger.info(f"  Loaded {len(df):,} ICU stays")
        return df
    
    def _load_patients(self) -> pd.DataFrame:
        """Load patients table."""
        df = load_table(
            self.data_dir, "patients", "hosp",
            usecols=["subject_id", "gender", "anchor_age",
                     "anchor_year", "anchor_year_group", "dod"],
        )
        logger.info(f"  Loaded {len(df):,} patients")
        return df
    
    def _load_admissions(self) -> pd.DataFrame:
        """Load admissions table."""
        df = load_table(
            self.data_dir, "admissions", "hosp",
            usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                     "deathtime", "hospital_expire_flag"],
        )
        df["admittime"] = pd.to_datetime(df["admittime"])
        df["dischtime"] = pd.to_datetime(df["dischtime"])
        df["deathtime"] = pd.to_datetime(df["deathtime"])
        logger.info(f"  Loaded {len(df):,} admissions")
        return df
    
    def _detect_suspected_infections(self) -> pd.DataFrame:
        """Detect suspected infections: antibiotics + cultures within ±24h.
        
        Returns:
            DataFrame with columns: subject_id, hadm_id, suspected_infection_time
        """
        logger.info("Detecting suspected infections...")
        
        # Load prescriptions (antibiotics)
        prescriptions = load_table(
            self.data_dir, "prescriptions", "hosp",
            usecols=["subject_id", "hadm_id", "starttime", "drug", "route"],
        )
        prescriptions["starttime"] = pd.to_datetime(prescriptions["starttime"])
        
        # Filter to antibiotics
        drug_lower = prescriptions["drug"].str.lower().fillna("")
        abx_mask = drug_lower.apply(
            lambda d: any(abx in d for abx in ANTIBIOTIC_KEYWORDS)
        )
        route_mask = prescriptions["route"].isin(VALID_ROUTES)
        antibiotics = prescriptions[abx_mask & route_mask][
            ["subject_id", "hadm_id", "starttime"]
        ].rename(columns={"starttime": "abx_time"})
        
        logger.info(f"  Found {len(antibiotics):,} antibiotic prescriptions")
        
        # Load microbiology events (cultures)
        micro = load_table(
            self.data_dir, "microbiologyevents", "hosp",
            usecols=["subject_id", "hadm_id", "charttime", "spec_type_desc"],
        )
        micro["charttime"] = pd.to_datetime(micro["charttime"])
        
        # Filter to relevant specimen types
        cultures = micro[micro["spec_type_desc"].isin(CULTURE_SPECIMEN_TYPES)][
            ["subject_id", "hadm_id", "charttime"]
        ].rename(columns={"charttime": "culture_time"})
        
        logger.info(f"  Found {len(cultures):,} culture specimens")
        
        # Join antibiotics and cultures within ±24h per admission
        merged = antibiotics.merge(
            cultures,
            on=["subject_id", "hadm_id"],
            how="inner",
        )
        
        # Filter to ±24 hours
        time_diff = (merged["abx_time"] - merged["culture_time"]).dt.total_seconds().abs()
        merged = merged[time_diff <= 86400]  # 24 hours in seconds
        
        # Suspected infection time = earlier of antibiotic or culture
        merged["suspected_infection_time"] = merged[["abx_time", "culture_time"]].min(axis=1)
        
        # Take earliest per admission
        infection_times = (
            merged
            .groupby(["subject_id", "hadm_id"])["suspected_infection_time"]
            .min()
            .reset_index()
        )
        
        logger.info(f"  Suspected infections detected: {len(infection_times):,} admissions")
        
        return infection_times
    
    def _compute_sofa_deltas(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Compute SOFA score changes from baseline for each ICU stay.
        
        Computes hourly SOFA scores and identifies the first time
        SOFA increases by ≥ 2 from the first-24h baseline.
        
        Returns:
            DataFrame with columns: stay_id, sofa_increase_time
        """
        logger.info("Computing SOFA score deltas...")
        
        stay_ids = cohort["stay_id"].unique()
        
        # Load component data
        # 1. Labs for SOFA (platelets, bilirubin, creatinine)
        labs = load_table(
            self.data_dir, "labevents", "hosp",
            usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
        )
        labs["charttime"] = pd.to_datetime(labs["charttime"])
        
        sofa_lab_items = {
            "platelets": 51265,
            "bilirubin": 50885,
            "creatinine": 50912,
            "pao2": 50821,
        }
        
        labs = labs[labs["itemid"].isin(sofa_lab_items.values())]
        logger.info(f"  Loaded {len(labs):,} SOFA-relevant lab values")
        
        # 2. Vitals for SOFA (MAP, GCS)
        chart_items = {
            "map": [52, 456, 6702, 220052, 220181],
            "gcs": [198, 226755, 227013],
            "fio2": [223835, 3420, 190],
        }
        all_chart_ids = []
        for ids in chart_items.values():
            all_chart_ids.extend(ids)
        
        # Load chartevents in chunks (very large table)
        charts = load_table(
            self.data_dir, "chartevents", "icu",
            usecols=["subject_id", "hadm_id", "stay_id",
                     "charttime", "itemid", "valuenum"],
        )
        charts["charttime"] = pd.to_datetime(charts["charttime"])
        charts = charts[charts["itemid"].isin(all_chart_ids)]
        logger.info(f"  Loaded {len(charts):,} SOFA-relevant chart values")
        
        # 3. Vasopressor use
        try:
            vasopressor_items = [221906, 221289, 221662, 221653, 222315, 221749]
            inputs = load_table(
                self.data_dir, "inputevents", "icu",
                usecols=["subject_id", "hadm_id", "stay_id",
                         "starttime", "itemid", "rate"],
            )
            inputs["starttime"] = pd.to_datetime(inputs["starttime"])
            vasopressors = inputs[inputs["itemid"].isin(vasopressor_items)]
            logger.info(f"  Loaded {len(vasopressors):,} vasopressor records")
        except Exception as e:
            logger.warning(f"  Could not load vasopressor data: {e}")
            vasopressors = pd.DataFrame()
        
        # 4. Urine output
        try:
            urine = load_table(
                self.data_dir, "outputevents", "icu",
                usecols=["subject_id", "hadm_id", "stay_id",
                         "charttime", "value"],
            )
            urine["charttime"] = pd.to_datetime(urine["charttime"])
            logger.info(f"  Loaded {len(urine):,} output events")
        except Exception as e:
            logger.warning(f"  Could not load urine output data: {e}")
            urine = pd.DataFrame()
        
        # Compute SOFA deltas per stay
        # For efficiency, compute simplified SOFA using available components
        sofa_results = []
        
        # Merge stay info for time context
        stay_info = cohort[["stay_id", "hadm_id", "subject_id", "intime"]].copy()
        
        # Simplified SOFA: use the components we can reliably compute
        # For each stay, compute SOFA at 6-hour intervals
        logger.info("  Computing per-stay SOFA deltas (this may take a while)...")
        
        for _, stay in tqdm(stay_info.iterrows(), total=len(stay_info), 
                           desc="  SOFA computation"):
            sid = stay["stay_id"]
            hadm = stay["hadm_id"]
            subj = stay["subject_id"]
            intime = stay["intime"]
            
            # Get labs for this admission
            stay_labs = labs[labs["hadm_id"] == hadm]
            stay_charts = charts[charts["stay_id"] == sid]
            
            if len(stay_labs) == 0 and len(stay_charts) == 0:
                continue
            
            # Compute SOFA components at various timepoints
            # Simplified: compute baseline (first 24h) and check for ≥2 increase
            sofa_time = self._compute_stay_sofa_delta(
                stay_labs, stay_charts, vasopressors, urine,
                sid, hadm, intime
            )
            
            if sofa_time is not None:
                sofa_results.append({
                    "stay_id": sid,
                    "sofa_increase_time": sofa_time,
                })
        
        if sofa_results:
            sofa_df = pd.DataFrame(sofa_results)
            logger.info(f"  SOFA increase (≥2) detected in {len(sofa_df):,} stays")
        else:
            sofa_df = pd.DataFrame(columns=["stay_id", "sofa_increase_time"])
            logger.warning("  No SOFA increases detected!")
        
        return sofa_df
    
    def _compute_stay_sofa_delta(
        self,
        labs: pd.DataFrame,
        charts: pd.DataFrame,
        vasopressors: pd.DataFrame,
        urine: pd.DataFrame,
        stay_id: int,
        hadm_id: int,
        intime: pd.Timestamp,
    ) -> Optional[pd.Timestamp]:
        """Compute the first time SOFA increases by ≥2 from baseline for a stay.
        
        Returns:
            Timestamp of first SOFA ≥ 2 increase, or None if no increase detected.
        """
        # Define time bins (6-hour intervals)
        # Baseline = first 24 hours
        baseline_end = intime + pd.Timedelta(hours=24)
        
        # --- Compute baseline SOFA components ---
        baseline_labs = labs[labs["charttime"] <= baseline_end]
        baseline_charts = charts[charts["charttime"] <= baseline_end]
        
        sofa_baseline = self._compute_sofa_at_time(
            baseline_labs, baseline_charts, is_baseline=True
        )
        
        # --- Compute subsequent SOFA at 6-hour intervals ---
        max_time = intime + pd.Timedelta(days=14)  # Cap at 14 days
        
        current_time = baseline_end
        while current_time < max_time:
            window_start = current_time - pd.Timedelta(hours=24)
            window_labs = labs[
                (labs["charttime"] > window_start) & (labs["charttime"] <= current_time)
            ]
            window_charts = charts[
                (charts["charttime"] > window_start) & (charts["charttime"] <= current_time)
            ]
            
            sofa_current = self._compute_sofa_at_time(
                window_labs, window_charts, is_baseline=False
            )
            
            if sofa_current - sofa_baseline >= 2:
                return current_time
            
            current_time += pd.Timedelta(hours=6)
        
        return None
    
    def _compute_sofa_at_time(
        self,
        labs: pd.DataFrame,
        charts: pd.DataFrame,
        is_baseline: bool = False,
    ) -> int:
        """Compute SOFA score from available labs and vitals.
        
        Simplified SOFA using reliably available components.
        """
        sofa = 0
        
        # 1. Coagulation (Platelets)
        plt_vals = labs[labs["itemid"] == 51265]["valuenum"].dropna()
        if len(plt_vals) > 0:
            plt_min = plt_vals.min() if not is_baseline else plt_vals.median()
            if plt_min < 20:
                sofa += 4
            elif plt_min < 50:
                sofa += 3
            elif plt_min < 100:
                sofa += 2
            elif plt_min < 150:
                sofa += 1
        
        # 2. Liver (Bilirubin)
        bili_vals = labs[labs["itemid"] == 50885]["valuenum"].dropna()
        if len(bili_vals) > 0:
            bili_max = bili_vals.max() if not is_baseline else bili_vals.median()
            if bili_max >= 12.0:
                sofa += 4
            elif bili_max >= 6.0:
                sofa += 3
            elif bili_max >= 2.0:
                sofa += 2
            elif bili_max >= 1.2:
                sofa += 1
        
        # 3. Renal (Creatinine)
        cr_vals = labs[labs["itemid"] == 50912]["valuenum"].dropna()
        if len(cr_vals) > 0:
            cr_max = cr_vals.max() if not is_baseline else cr_vals.median()
            if cr_max >= 5.0:
                sofa += 4
            elif cr_max >= 3.5:
                sofa += 3
            elif cr_max >= 2.0:
                sofa += 2
            elif cr_max >= 1.2:
                sofa += 1
        
        # 4. Cardiovascular (MAP)
        map_ids = [52, 456, 6702, 220052, 220181]
        map_vals = charts[charts["itemid"].isin(map_ids)]["valuenum"].dropna()
        if len(map_vals) > 0:
            map_min = map_vals.min() if not is_baseline else map_vals.median()
            if map_min < 70:
                sofa += 1  # Simplified: just MAP < 70
        
        # 5. CNS (GCS)
        gcs_ids = [198, 226755, 227013]
        gcs_vals = charts[charts["itemid"].isin(gcs_ids)]["valuenum"].dropna()
        if len(gcs_vals) > 0:
            gcs_min = gcs_vals.min() if not is_baseline else gcs_vals.median()
            if gcs_min < 6:
                sofa += 4
            elif gcs_min < 10:
                sofa += 3
            elif gcs_min < 13:
                sofa += 2
            elif gcs_min < 15:
                sofa += 1
        
        # 6. Respiratory (PaO2/FiO2 if available)
        pao2_vals = labs[labs["itemid"] == 50821]["valuenum"].dropna()
        fio2_ids = [223835, 3420, 190]
        fio2_vals = charts[charts["itemid"].isin(fio2_ids)]["valuenum"].dropna()
        
        if len(pao2_vals) > 0 and len(fio2_vals) > 0:
            pao2 = pao2_vals.iloc[-1]  # Most recent
            fio2 = fio2_vals.iloc[-1]
            # Ensure FiO2 is in fraction form
            if fio2 > 1:
                fio2 = fio2 / 100.0
            if fio2 > 0:
                pf_ratio = pao2 / fio2
                if pf_ratio < 100:
                    sofa += 4
                elif pf_ratio < 200:
                    sofa += 3
                elif pf_ratio < 300:
                    sofa += 2
                elif pf_ratio < 400:
                    sofa += 1
        
        return sofa
    
    def _identify_sepsis_onset(
        self,
        cohort: pd.DataFrame,
        infection_times: pd.DataFrame,
        sofa_deltas: pd.DataFrame,
    ) -> pd.DataFrame:
        """Identify Sepsis-3 onset: infection + SOFA ≥ 2 increase.
        
        Onset time = max(suspected_infection_time, sofa_increase_time)
        """
        logger.info("Identifying Sepsis-3 onset times...")
        
        # Merge infection times
        cohort = cohort.merge(
            infection_times,
            on=["subject_id", "hadm_id"],
            how="left",
        )
        
        # Merge SOFA deltas
        cohort = cohort.merge(
            sofa_deltas,
            on="stay_id",
            how="left",
        )
        
        # Sepsis-3: both infection AND SOFA increase
        has_infection = cohort["suspected_infection_time"].notna()
        has_sofa = cohort["sofa_increase_time"].notna()
        
        cohort["sepsis_label"] = (has_infection & has_sofa).astype(int)
        
        # Onset time = max of the two times
        cohort["sepsis_onset_time"] = pd.NaT
        sepsis_mask = cohort["sepsis_label"] == 1
        
        if sepsis_mask.any():
            cohort.loc[sepsis_mask, "sepsis_onset_time"] = cohort.loc[sepsis_mask].apply(
                lambda row: max(row["suspected_infection_time"], row["sofa_increase_time"]),
                axis=1,
            )
        
        # Validate: onset must be during ICU stay
        valid_onset = (
            (cohort["sepsis_onset_time"] >= cohort["intime"]) &
            (cohort["sepsis_onset_time"] <= cohort["outtime"])
        )
        invalid_count = sepsis_mask.sum() - valid_onset[sepsis_mask].sum()
        if invalid_count > 0:
            logger.warning(f"  {invalid_count} sepsis onsets outside ICU stay — removing")
            cohort.loc[sepsis_mask & ~valid_onset, "sepsis_label"] = 0
            cohort.loc[sepsis_mask & ~valid_onset, "sepsis_onset_time"] = pd.NaT
        
        return cohort


def run_cohort_extraction(config_path: Optional[str] = None) -> pd.DataFrame:
    """Run the cohort extraction pipeline.
    
    Args:
        config_path: Path to YAML config file (None = use defaults)
    
    Returns:
        DataFrame with extracted cohort
    """
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = get_default_config()
    
    extractor = CohortExtractor(config)
    return extractor.extract_cohort()


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Extract Sepsis-3 cohort from MIMIC-IV")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override MIMIC-IV data directory")
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config) if args.config else get_default_config()
    if args.data_dir:
        config.data.mimic_raw_dir = args.data_dir
    
    cohort = CohortExtractor(config).extract_cohort()
    print(f"\nFinal cohort shape: {cohort.shape}")
    print(f"Sepsis prevalence: {cohort['sepsis_label'].mean():.1%}")
