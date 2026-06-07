# Fever Etiology Classification Methodology

## Why NOT "FUO Detection"?

### Classical FUO Definition (Petersdorf & Beeson, 1961)
- Fever persisting **>3 weeks**
- Temperature **>38.3°C** on multiple occasions
- Remains **undiagnosed after 1-week hospital workup**

### MIMIC-IV Reality
- ICU stays average **2-5 days** (not weeks)
- Patients are **actively being diagnosed** (not "unknown origin")
- Discharge diagnoses are **available** (not unknown)

**Conclusion:** MIMIC-IV cannot support classical FUO research.

---

## Our Adaptation: "Early Fever Etiology Classification in ICU"

### Problem Statement
**Given:** A febrile ICU patient (temp >38.3°C) in first 48 hours of admission  
**Predict:** Underlying etiology category to guide targeted workup

### Clinical Value
- **Faster diagnosis:** Hierarchical classification narrows differential faster than flat multi-class
- **Resource optimization:** Bacterial vs viral distinction avoids unnecessary antibiotics
- **Early intervention:** Identifying neoplastic/autoimmune causes triggers appropriate consults

---

## Hierarchical Etiology Structure

Inspired by FUO paper's classification (Wang et al., IEEE JBHI 2023):

```
Root
├── Infectious Disease (Level 1)
│   ├── Bacterial Infection (Level 2)
│   ├── Viral Infection (Level 2)
│   └── Fungal Infection (Level 2)
└── Noninfectious Disease (Level 1)
    ├── Neoplastic Disease (Level 2)
    ├── Autoimmune/Inflammatory (Level 2)
    └── Drug-Induced Fever (Level 2)
```

### ICD-10 Mapping

| Etiology | ICD-10 Codes | Examples |
|---|---|---|
| **Bacterial** | A00-A49, J13-J16, N10-N39 | Sepsis, pneumonia, UTI, tuberculosis |
| **Viral** | A80-A89, B00-B34, J09-J12 | Influenza, HIV, hepatitis, CMV, EBV |
| **Fungal** | B35-B49, J17.2 | Candidiasis, aspergillosis, histoplasmosis |
| **Neoplastic** | C00-C95 | Lymphoma, leukemia, solid tumors |
| **Autoimmune** | M05-M35, K50-K51, D86 | Rheumatoid arthritis, lupus, IBD, sarcoidosis |
| **Drug Fever** | T36-T50 | Antibiotic hypersensitivity, drug reactions |

---

## Cohort Extraction Pipeline

### Step 1: Identify Febrile Patients
```sql
SELECT stay_id, subject_id, hadm_id
FROM chartevents
WHERE itemid IN (223762, 226329)  -- Temperature
  AND valuenum > 38.3  -- Celsius
  AND hours_since_admission <= 48
```
**Result:** ~X,XXX febrile ICU stays

### Step 2: Exclude Obvious Sepsis
Patients meeting Sepsis-3 criteria are handled by the existing sepsis pipeline:
- Suspected infection (antibiotics + cultures)
- SOFA score increase ≥2

**Rationale:** Sepsis is a specific syndrome requiring different management. We focus on non-septic fever.

### Step 3: Assign Etiology Labels
- Load `diagnoses_icd` table (discharge diagnoses)
- Map ICD-10 codes to etiology hierarchy
- Exclude "unknown" etiologies (can't train on these)

### Step 4: Add Demographics & Clinical Data
- Age, gender from `patients` table
- Admission type, location from `admissions` table
- Lab values, vitals from existing feature extraction pipeline

---

## Technical Approach (Adapted from FUO Paper)

### 1. Hierarchical Classification Framework
**Paper's Td-HRF (Top-down Hierarchical Reasoning Framework):**
- One local classifier per parent node
- Training: Each classifier trained independently
- Inference: Top-down traversal with "consensus" algorithm

**Our Implementation:**
```python
# Level 1: Infectious vs Noninfectious
classifier_root = La-MNN(input_dim=12, output_dim=2)

# Level 2a: Bacterial vs Viral vs Fungal (if infectious)
classifier_infectious = La-MNN(input_dim=12, output_dim=3)

# Level 2b: Neoplastic vs Autoimmune vs Drug (if noninfectious)
classifier_noninfectious = La-MNN(input_dim=12, output_dim=3)
```

### 2. Multimodal Neural Network (La-MNN)
**Paper's Architecture:**
- **Static data:** Demographics + lab tests → DNN encoder
- **Time series:** Vital signs → Attention-based GRU-D
- **Clinical notes:** Symptoms → Text encoder (we'll use BioBERT)
- **Fusion:** Concatenate embeddings → Classification head

**Our Reuse:**
- ✅ Already have `SepsisLSTM` with temporal attention
- ✅ Already have 12-feature vital/lab pipeline
- 🆕 Add: Clinical note extraction from `noteevents` table
- 🆕 Add: Hierarchical classification heads

### 3. Quantum Kernel Enhancement (Our Novelty)
**Paper uses:** Classical SVM at each hierarchy node  
**We add:** Quantum kernel SVM (already implemented for sepsis)

```python
# At each hierarchy node:
lstm_embeddings = model.extract_embeddings(X)  # (N, 16)
quantum_kernel = QuantumKernelSepsis(n_qubits=8)
quantum_kernel.fit(lstm_embeddings, y)
predictions = quantum_kernel.predict_scores(X_test)
```

### 4. Interpretability (Paper's LRP + Attention)
**Layer-wise Relevance Propagation (LRP):**
- Backpropagate relevance scores from output to input
- Identifies which lab values/vitals contributed most to decision

**Attention Weights:**
- Temporal attention shows which hours in 48h window were critical
- Spatial attention shows which features (HR, temp, lactate) were important

**Clinical Value:** Explains WHY the model predicted bacterial vs viral, helping clinicians validate or override.

---

## Comparison: FUO Paper vs Our Adaptation

| Aspect | FUO Paper (Wang et al.) | Our Adaptation |
|---|---|---|
| **Dataset** | Chinese hospital EHR, 30,794 FUO patients | MIMIC-IV, ~X,XXX febrile ICU patients |
| **Time window** | First 48h after admission ✅ | First 48h after admission ✅ |
| **Cohort definition** | Fever >3 weeks, undiagnosed | Fever >38.3°C in ICU, non-septic |
| **Etiology hierarchy** | 4 coarse + 12 fine categories | 2 coarse + 6 fine categories (simplified) |
| **Multimodal data** | Tabular + time series + notes ✅ | Tabular + time series + notes ✅ |
| **Architecture** | Td-HRF + La-MNN ✅ | Td-HRF + La-MNN ✅ |
| **Interpretability** | LRP + attention ✅ | LRP + attention ✅ |
| **Novel contribution** | Hierarchical + multimodal | **+ Quantum kernel + Conformal prediction + Safety agents** |

---

## Naming Convention

### ❌ Avoid: "FUO Detection"
- Clinically inaccurate (not true FUO)
- Misleading to reviewers/clinicians
- Violates Petersdorf-Beeson criteria

### ✅ Use: "Early Fever Etiology Classification in ICU"
- Accurate description of task
- Acknowledges ICU setting limitations
- Emphasizes "early" (48h) prediction window

### ✅ Alternative: "Hierarchical Fever Etiology Prediction Using Quantum-Enhanced Multimodal Learning"
- Highlights technical contributions
- Avoids FUO terminology
- Emphasizes quantum + multimodal novelty

---

## Expected Results

### Baseline (XGBoost on flattened features)
- AUROC: ~0.75-0.80 (based on sepsis results)

### Our Hierarchical Model
- **Level 1 (Infectious vs Noninfectious):** AUROC ~0.80-0.85
- **Level 2 (Fine-grained):** AUROC ~0.75-0.80
- **With quantum kernel:** +2-5% AUROC improvement
- **With conformal prediction:** 90% coverage guarantee

### Clinical Metrics
- **Sensitivity @ 95% specificity:** Target >0.80
- **Time to correct diagnosis:** Reduce by X hours (compare to standard workup)
- **Unnecessary antibiotic use:** Reduce by Y% (viral vs bacterial distinction)

---

## References

1. **FUO Paper:** Wang et al., "Integrating Medical Domain Knowledge for Early Diagnosis of Fever of Unknown Origin: An Interpretable Hierarchical Multimodal Neural Network Approach," IEEE JBHI, 2023.

2. **Sepsis-3 Criteria:** Singer et al., "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)," JAMA, 2016.

3. **MIMIC-IV:** Johnson et al., "MIMIC-IV, a freely accessible electronic health record dataset," Scientific Data, 2023.

---

## Next Steps

1. **Run cohort extraction:** `python -m src.data.fever_cohort_extraction --data-dir <path>`
2. **Analyze cohort balance:** Check infectious vs noninfectious distribution
3. **Extract clinical notes:** Implement NLP pipeline for symptom extraction
4. **Adapt LSTM architecture:** Add hierarchical classification heads
5. **Train hierarchical model:** One classifier per parent node
6. **Integrate quantum kernel:** Replace classical SVM with quantum SVM
7. **Validate interpretability:** Generate LRP heatmaps + attention visualizations

---

**Bottom Line:** This is a rigorous adaptation of the FUO paper's methodology to a MIMIC-IV-compatible problem. We maintain their technical innovations (hierarchical classification, multimodal fusion, interpretability) while adding our quantum + conformal + safety contributions. The naming accurately reflects the clinical reality.
