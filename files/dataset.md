# MIMIC-IV v3.1 Dataset — Complete Reference Guide

> **QuantumSepsis Shield Data Pipeline Reference**

---

## 1. Dataset Overview

| Property               | Value                                    |
|------------------------|------------------------------------------|
| **Full Name**          | Medical Information Mart for Intensive Care IV |
| **Version**            | 3.1                                      |
| **Source**             | PhysioNet (physionet.org/content/mimiciv/3.1/) |
| **Access Level**       | Credentialed (requires CITI training + signed DUA) |
| **Data Period**        | 2008–2022 (deidentified → shifted to 2100–2200) |
| **Institution**        | Beth Israel Deaconess Medical Center (BIDMC) |
| **EHR System**         | MetaVision (ICU CIS)                     |
| **Total Size**         | ~9.9 GB uncompressed                     |
| **License**            | PhysioNet Credentialed Health Data License 1.5.0 |

---

## 2. Key Scale Metrics

| Metric                    | Count      |
|---------------------------|------------|
| Unique patients           | 364,627    |
| Hospitalizations          | 546,028    |
| ICU stays                 | 94,458     |
| Unique ICU patients       | 65,366     |
| ICU care units            | Multiple (MICU, SICU, CCU, CVICU, TSICU, etc.) |

---

## 3. Module Structure

MIMIC-IV v3.1 is organized into **two main modules**:

### 3.1 `hosp` Module — Hospital-Wide EHR Data

This module contains data recorded during the entire hospital stay (not just ICU).

| Table                | Records (approx.) | Key Columns                          | Our Use                              |
|----------------------|-------------------|--------------------------------------|--------------------------------------|
| `patients`           | 364,627           | subject_id, gender, anchor_year, anchor_year_group, anchor_age, dod | Demographics, mortality labels       |
| `admissions`         | 546,028           | subject_id, hadm_id, admittime, dischtime, deathtime, insurance, race | Admission events, death timestamps   |
| `labevents`          | ~124M             | subject_id, hadm_id, specimen_id, itemid, charttime, value, valuenum, valueuom | Lactate, WBC, Creatinine, Bilirubin, Platelets, PCT |
| `d_labitems`         | ~1,600            | itemid, label, fluid, category, loinc_code | Lab item dictionary/lookup           |
| `microbiologyevents` | ~600K             | subject_id, hadm_id, charttime, spec_type_desc, org_name, ab_name, interpretation | Suspected infection detection         |
| `d_micro`            | ~100              | spec_itemid, test_itemid, org_itemid, ab_itemid + labels | Microbiology dictionary              |
| `prescriptions`      | ~17M              | subject_id, hadm_id, starttime, stoptime, drug, route, dose_val_rx | Antibiotic prescriptions (→ suspected infection) |
| `pharmacy`           | ~14M              | subject_id, hadm_id, medication, route, frequency, doses_per_24_hrs | Pharmacy dispensing records          |
| `diagnoses_icd`      | ~5M               | subject_id, hadm_id, seq_num, icd_code, icd_version | ICD-9/10 sepsis codes for validation |
| `d_icd_diagnoses`    | ~100K             | icd_code, icd_version, long_title    | ICD code dictionary                  |
| `procedures_icd`     | ~1M               | subject_id, hadm_id, seq_num, icd_code, icd_version | Procedure tracking                   |
| `emar`               | ~27M              | subject_id, hadm_id, charttime, medication, event_txt | Medication administration events     |
| `emar_detail`        | ~57M              | subject_id, parent_field_ordinal, dose_due, dose_given | Medication dose details              |
| `omr`                | ~7M               | subject_id, chartdate, result_name, result_value | Height, weight, BMI, BP baseline     |
| `transfers`          | ~2M               | subject_id, hadm_id, transfer_id, eventtype, careunit, intime, outtime | ICU transfer events                  |
| `services`           | ~540K             | subject_id, hadm_id, transfertime, prev_service, curr_service | Service type tracking                |
| `poe`                | ~43M              | subject_id, hadm_id, ordertime, order_type, order_subtype | Physician order entry                |
| `poe_detail`         | ~3M               | poe_id, field_name, field_value      | Order details                        |

### 3.2 `icu` Module — ICU-Specific Data (MetaVision CIS)

This module contains high-frequency charted data from ICU stays only.

| Table               | Records (approx.) | Key Columns                          | Our Use                              |
|---------------------|-------------------|--------------------------------------|--------------------------------------|
| `icustays`          | 94,458            | subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime, los | ICU stay identification              |
| `chartevents`       | ~330M             | subject_id, hadm_id, stay_id, charttime, itemid, value, valuenum, valueuom | **PRIMARY**: HR, BP, SpO2, Temp, RR, GCS |
| `inputevents`       | ~9M               | subject_id, hadm_id, stay_id, starttime, endtime, itemid, amount, amountuom, rate, rateuom | Vasopressor use (SOFA cardiovascular) |
| `outputevents`      | ~4M               | subject_id, hadm_id, stay_id, charttime, itemid, value, valueuom | Urine output (SOFA renal)            |
| `procedureevents`   | ~700K             | subject_id, hadm_id, stay_id, starttime, endtime, itemid, value | Mechanical ventilation tracking      |
| `ingredientevents`  | ~12M              | subject_id, hadm_id, stay_id, starttime, endtime, itemid, amount | IV fluid ingredients                 |
| `datetimeevents`    | ~7M               | subject_id, hadm_id, stay_id, charttime, itemid, value | Timestamped documentation            |
| `d_items`           | ~4,000            | itemid, label, abbreviation, linksto, category, unitname | Item dictionary for all charted data |
| `caregiver`         | ~16K              | caregiver_id                         | Deidentified caregiver IDs           |

---

## 4. Critical Item IDs for Feature Extraction

### 4.1 Vital Signs from `chartevents`

| Variable           | Item IDs                              | Notes                          |
|--------------------|---------------------------------------|--------------------------------|
| **Heart Rate**     | 211, 220045                           | CareVue (211), MetaVision (220045) |
| **SBP**            | 51, 442, 455, 6701, 220179, 220050    | Arterial + non-invasive        |
| **DBP**            | 8368, 8440, 8441, 8555, 220180, 220051| Arterial + non-invasive        |
| **MAP**            | 52, 456, 6702, 220052, 220181         | Mean arterial pressure         |
| **Temperature (°C)**| 223762, 226329                       | MetaVision Celsius IDs         |
| **Respiratory Rate**| 615, 618, 220210, 224690             | Multiple charting sources      |
| **SpO2**           | 646, 220277                           | Pulse oximetry                 |
| **GCS - Total**    | 198, 226755, 227013                   | Glasgow Coma Scale total       |
| **GCS - Eye**      | 220739                                | For SOFA CNS component         |
| **GCS - Motor**    | 223900                                | For SOFA CNS component         |
| **GCS - Verbal**   | 223901                                | For SOFA CNS component         |

### 4.2 Lab Values from `labevents`

| Variable          | Item ID  | Unit     | Normal Range  | Clinical Significance         |
|-------------------|----------|----------|---------------|-------------------------------|
| **Lactate**       | 50813    | mmol/L   | 0.5–2.0       | Tissue hypoperfusion marker   |
| **WBC**           | 51301    | K/uL     | 4.5–11.0      | Infection/inflammation        |
| **Creatinine**    | 50912    | mg/dL    | 0.7–1.3       | Renal dysfunction (SOFA)      |
| **Bilirubin**     | 50885    | mg/dL    | 0.1–1.2       | Liver dysfunction (SOFA)      |
| **Platelets**     | 51265    | K/uL     | 150–400       | Coagulation (SOFA)            |
| **Procalcitonin** | *search* | ng/mL    | < 0.1         | Bacterial infection biomarker |
| **PaO2**          | 50821    | mmHg     | 80–100        | Respiratory (SOFA PaO2/FiO2) |
| **FiO2**          | 50816    | —        | 0.21          | Respiratory (SOFA PaO2/FiO2) |

> **Note on Procalcitonin:** Item ID not fixed — search `d_labitems` with `label ILIKE '%procalcitonin%'` to find the correct itemid in your MIMIC-IV instance.

### 4.3 Vasopressor Item IDs from `inputevents`

| Drug              | Item ID  | Notes                         |
|-------------------|----------|-------------------------------|
| Norepinephrine    | 221906   | First-line vasopressor        |
| Epinephrine       | 221289   | Second-line                   |
| Dopamine          | 221662   | Legacy vasopressor            |
| Dobutamine        | 221653   | Inotrope                     |
| Vasopressin       | 222315   | Adjunct vasopressor           |
| Phenylephrine     | 221749   | Alpha-agonist                 |

### 4.4 FiO2 from `chartevents`

| Source            | Item IDs         |
|-------------------|-----------------|
| FiO2 (charted)    | 223835, 3420    |
| FiO2 (ventilator) | 190, 3420       |

---

## 5. Sepsis-3 Cohort Derivation

### 5.1 Sepsis-3 Definition (Singer et al., JAMA 2016)

Sepsis-3 requires **both**:
1. **Suspected infection** — co-occurrence of antibiotic order AND blood/body fluid culture within ±24 hours
2. **Organ dysfunction** — SOFA score increase ≥ 2 points from baseline

### 5.2 SOFA Score Computation

| Component       | Source                          | Score 0  | Score 1    | Score 2    | Score 3    | Score 4        |
|-----------------|--------------------------------|----------|------------|------------|------------|----------------|
| **Respiratory** | PaO2/FiO2                      | ≥ 400   | < 400      | < 300      | < 200 w/vent| < 100 w/vent  |
| **Coagulation** | Platelets (K/uL)               | ≥ 150   | < 150      | < 100      | < 50       | < 20           |
| **Liver**       | Bilirubin (mg/dL)              | < 1.2   | 1.2–1.9    | 2.0–5.9    | 6.0–11.9   | > 12.0         |
| **Cardiovascular** | MAP / Vasopressors          | MAP ≥ 70 | MAP < 70  | Dopa ≤ 5   | Dopa > 5   | Dopa > 15      |
| **CNS**         | GCS                            | 15      | 13–14      | 10–12      | 6–9        | < 6            |
| **Renal**       | Creatinine (mg/dL) / UO (mL/d)| < 1.2   | 1.2–1.9    | 2.0–3.4    | 3.5–4.9    | > 5.0          |

### 5.3 Suspected Infection Detection SQL Pattern

```sql
-- Find suspected infection: antibiotic + culture within ±24 hours
WITH antibiotics AS (
    SELECT subject_id, hadm_id, starttime AS abx_time
    FROM mimiciv_hosp.prescriptions
    WHERE LOWER(drug) SIMILAR TO '%(vancomycin|piperacillin|meropenem|ceftriaxone|levofloxacin|ciprofloxacin|metronidazole|azithromycin|ampicillin|cefepime|cefazolin|gentamicin|tobramycin|amikacin|clindamycin|doxycycline|trimethoprim|linezolid|daptomycin|ertapenem|ceftazidime|amoxicillin|nafcillin|oxacillin|cephalexin)%'
    AND route IN ('IV', 'PO', 'ORAL', 'PO/NG', 'IV DRIP')
),
cultures AS (
    SELECT subject_id, hadm_id, charttime AS culture_time
    FROM mimiciv_hosp.microbiologyevents
    WHERE spec_type_desc IN ('BLOOD CULTURE', 'URINE', 'SPUTUM', 'BRONCHOALVEOLAR LAVAGE',
                              'CATHETER TIP-IV', 'CSF;SPINAL FLUID', 'PERITONEAL FLUID',
                              'ABSCESS', 'WOUND')
),
suspected_infection AS (
    SELECT a.subject_id, a.hadm_id,
           LEAST(a.abx_time, c.culture_time) AS suspected_infection_time
    FROM antibiotics a
    INNER JOIN cultures c
        ON a.subject_id = c.subject_id
        AND a.hadm_id = c.hadm_id
        AND ABS(EXTRACT(EPOCH FROM (a.abx_time - c.culture_time))) <= 86400  -- within 24 hours
)
SELECT subject_id, hadm_id, MIN(suspected_infection_time) AS infection_onset
FROM suspected_infection
GROUP BY subject_id, hadm_id;
```

### 5.4 Sepsis Onset Time

```
onset_time = MAX(suspected_infection_time, sofa_increase_time)
```

Where `sofa_increase_time` = the first time SOFA increases by ≥ 2 from the admission baseline (computed using the first 24h minimum SOFA as baseline).

### 5.5 Labeling Strategy

| Label | Condition                                          |
|-------|---------------------------------------------------|
| **1** | Sepsis-3 onset occurs during ICU stay              |
| **0** | No Sepsis-3 criteria met during entire ICU stay    |

**Prediction window:** Generate labels for 3–4 hours BEFORE onset time:
- For sepsis patients: label applies to windows ending 3–4 hours before onset
- For non-sepsis patients: randomly sample equivalent windows from their ICU stay

---

## 6. Date Handling — CRITICAL

> ⚠️ **All dates in MIMIC-IV are shifted into the 2100–2200 range for deidentification.**

### Rules:
1. **Intra-patient time differences are preserved** — a single date shift is applied per `subject_id`
2. **DO NOT compare timestamps across different patients** — the shift differs by patient
3. Use `anchor_year` + `anchor_year_group` to recover approximate real-world era:
   - `anchor_year_group = "2008 - 2010"` means patient was admitted ~2008-2010
   - `anchor_year` is the deidentified year corresponding to the real year
4. For temporal train/test split:
   - `anchor_year_group ≤ "2017 - 2019"` → Training set
   - `anchor_year_group = "2020 - 2022"` → Test set (temporal validation)

---

## 7. Data Pipeline Output Specification

### 7.1 Cohort CSV
```
subject_id, hadm_id, stay_id, intime, outtime, los_hours,
sepsis_label, sepsis_onset_time, anchor_year_group,
gender, anchor_age, icu_type, mortality_hospital
```

### 7.2 Feature Tensor
```
Shape: (N_windows, 6, 12)
  - N_windows: total number of 6-hour sliding windows across all patients
  - 6: hourly time steps in the window
  - 12: feature variables

Storage: HDF5 file with datasets:
  - 'X_train': (N_train, 6, 12) float32
  - 'y_train': (N_train,) int8
  - 'X_val': (N_val, 6, 12) float32
  - 'y_val': (N_val,) int8
  - 'X_test': (N_test, 6, 12) float32
  - 'y_test': (N_test,) int8
  - 'metadata': {stay_id, window_end_time, hours_before_onset}
```

### 7.3 Normalization Statistics
```json
{
  "feature_names": ["HR", "SBP", "DBP", "MAP", "Temp", "RR", "SpO2", "GCS",
                     "Lactate", "WBC", "Creatinine", "Platelets"],
  "train_mean": [85.2, 120.5, ...],
  "train_std": [15.3, 22.1, ...],
  "train_median": [84.0, 118.0, ...]
}
```

---

## 8. Expected Class Distribution

Based on published MIMIC-IV sepsis prevalence studies:

| Metric                    | Estimate           |
|---------------------------|-------------------|
| Total ICU stays           | ~94,458           |
| Sepsis-3 positive stays   | ~22,000–28,000    |
| Sepsis prevalence in ICU  | ~23–30%           |
| Training windows (est.)   | ~500,000–800,000  |
| Sepsis windows            | ~150,000–250,000  |
| Non-sepsis windows        | ~350,000–550,000  |
| Class ratio               | ~1:2 to 1:3       |

**Imbalance handling:**
- Not severely imbalanced (unlike general hospital admission data)
- Asymmetric focal loss (FN=10×FP weight) handles remaining imbalance
- No SMOTE needed — sufficient positive samples exist

---

## 9. BigQuery Access Schema

If using Google BigQuery for MIMIC-IV access:

```
Project: physionet-data
Dataset (hosp): mimiciv_v3_1_hosp
Dataset (icu):  mimiciv_v3_1_icu

Example query:
SELECT * FROM `physionet-data.mimiciv_v3_1_icu.icustays` LIMIT 10;
SELECT * FROM `physionet-data.mimiciv_v3_1_hosp.labevents` WHERE itemid = 50813 LIMIT 10;
```

---

## 10. Data Download

### Local Download Command
```bash
wget -r -N -c -np --user nityankulkarni28 --ask-password \
  https://physionet.org/files/mimiciv/3.1/
```

### File Structure After Download
```
mimiciv/3.1/
├── hosp/
│   ├── admissions.csv.gz
│   ├── patients.csv.gz
│   ├── labevents.csv.gz       (~4.5 GB compressed)
│   ├── d_labitems.csv.gz
│   ├── microbiologyevents.csv.gz
│   ├── prescriptions.csv.gz
│   ├── diagnoses_icd.csv.gz
│   ├── d_icd_diagnoses.csv.gz
│   ├── ...
│   └── transfers.csv.gz
├── icu/
│   ├── icustays.csv.gz
│   ├── chartevents.csv.gz     (~6 GB compressed, largest file)
│   ├── inputevents.csv.gz
│   ├── outputevents.csv.gz
│   ├── procedureevents.csv.gz
│   ├── d_items.csv.gz
│   └── ...
└── LICENSE.txt
```

---

## 11. Ethical & Legal Compliance

1. **PhysioNet DUA signed** — all team members must complete CITI training
2. **No re-identification attempts** — deidentified data must remain deidentified
3. **No data sharing** — MIMIC-IV cannot be shared outside credentialed users
4. **Date shifts** — acknowledge in all publications that dates are shifted
5. **IRB** — MIMIC-IV has blanket IRB approval; no additional IRB needed for retrospective analysis
6. **Citation required:** Johnson, A., Bulgarelli, L., Pollard, T., et al. (2023). MIMIC-IV. PhysioNet. https://doi.org/10.13026/6mm1-ek67
