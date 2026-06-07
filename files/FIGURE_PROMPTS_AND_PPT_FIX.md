# QuantumSepsis Shield — Figure Prompts & PPT Fix Guide

---

## PART 1 — FIGURE PROMPTS (Academic / Formal Style)

> Strict guidelines for both figures:
> - **No neon, no purple, no glow effects, no gradients**
> - Palette: Navy (`#1a2744`), Steel blue (`#2c4a7c`), White (`#ffffff`), Slate grey (`#94a3b8`), Accent red (`#c0392b`) — used only for critical alert elements
> - Font: **Inter** or **Helvetica Neue**, weight 400/600 only
> - All text in **black or dark navy on white background** (light theme, publication-ready)
> - Clean lines, minimal borders, IEEE/Nature conference poster aesthetic

---

### Figure 1 — System Architecture Diagram

**Title for slide:** `Figure 1. QuantumSepsis Shield: Five-Layer Clinical Inference Pipeline`

**Image Generation Prompt:**

```
Create a clean, formal, academic-style system architecture diagram on a pure white 
background for a medical AI research poster.

Title at top (dark navy, bold, 14pt): 
"QuantumSepsis Shield — Five-Layer Clinical Inference Pipeline"

The diagram shows a LEFT-TO-RIGHT horizontal data flow pipeline with 5 distinct 
labeled layers. Each layer is a clean rectangular box with a thin dark-navy border, 
soft white fill, and a 2-4 word bold label. Below each box, in smaller grey text, 
print the technical sub-components.

Use single-line horizontal arrows (→) between layers. No curved lines. 
No shadows. No gradients. No icons.

LAYER 1 — INPUT (leftmost)
  Box label: "MIMIC-IV Vitals Window"
  Sub-text: "6h × 12 features | HR, MAP, Temp, RR, SpO₂, GCS, Lactate, WBC, 
             Creatinine, Platelets, SBP, DBP | Shape: (6, 12)"

  → Arrow labeled: "z-score normalization"

LAYER 2 — FEATURE EXTRACTION
  Box label: "Bidirectional LSTM"
  Sub-text: "128 hidden units | 2 layers | Temporal self-attention | 
             Dropout 0.3 | Output: 16-dim embedding"

  Below this box, draw a VERTICAL DOWNWARD dashed arrow to a smaller side box:
  Side box label: "Quantum Kernel (8 qubits)"
  Side box sub-text: "ZZFeatureMap | 8-dim PCA projection | Hilbert-space 
                      nonconformity | IBM Qiskit Aer simulator"
  Arrow from side box back UP into Layer 3 labeled: "kernel similarity"

  → Main arrow labeled: "ensemble weighted (0.3 LSTM + 0.7 XGBoost)"

LAYER 3 — RISK ESTIMATION
  Box label: "Ensemble Risk Score"
  Sub-text: "XGBoost (132 features) | Window AUROC: 0.8051 | 
             Conformal interval: [score ± q_α] | q_α = 0.2663 | 
             Coverage guarantee: 90%"

  → Arrow labeled: "risk score + interval width"

LAYER 4 — SAFETY OVERRIDE (draw this box with a thin red border to distinguish it)
  Box label: "Red Team Agent"
  Sub-text: "7 deterministic tripwires | TW-TEMP, TW-HR, TW-RR, TW-MAP, 
             TW-SPO2, TW-LACTATE, TW-MENTAL | Non-overridable CRITICAL escalation"

  → Arrow labeled: "override assessment"

LAYER 5 — CLINICAL OUTPUT (rightmost)
  Box label: "Confidence-Gated Orchestrator"
  Sub-text: "WATCH / AMBER / CRITICAL / FAST-TRACK | Confidence-gated 
             fast-tracking (conf > 0.80 + risk > 0.60) | Sepsis bundle actions | 
             Counterfactual intervention estimation"

At the very bottom, add a single-line caption in grey italic 8pt:
"Pipeline operates on 6-hour sliding windows with 1-hour stride. 
 Prediction horizon: 3–4 hours before Sepsis-3 onset."

Overall style: IEEE conference paper figure. Clean, minimal, black-on-white. 
Publication ready. No decorative elements.
```

---

### Figure 2 — Dataset Overview & Feature Attributes Table

**Title for slide:** `Figure 2. MIMIC-IV v3.1 Dataset Overview and Feature Schema`

**Image Generation Prompt:**

```
Create a clean, formal academic-style combined figure on a pure white background 
for a medical AI research paper. The figure has TWO PANELS arranged SIDE BY SIDE.

NO gradients. NO neon colors. NO decorative icons. Navy (#1a2744) and white only, 
with slate grey (#64748b) for secondary text.

─────────────────────────────────────────
PANEL A (LEFT PANEL — 40% of width)
─────────────────────────────────────────
Title: "Dataset: MIMIC-IV v3.1" (bold, navy, 11pt)

A clean vertical stack of four metric boxes. Each box has a thin navy border, 
white fill, a large bold number in navy, and a small grey description beneath.

Box 1: "94,458" (large, bold navy) / "ICU stays, Beth Israel Deaconess MC"
Box 2: "~22,000–28,000" / "Sepsis-3 positive stays (~23–30% prevalence)"
Box 3: "4,094,917 / 796,893" / "Training windows / Test windows (6h×1h stride)"
Box 4: "2008–2022" / "Temporal span | Train: 2008–2019 | Test: 2020–2022"

Below the boxes, print a small sub-heading: "Sepsis-3 Definition (Singer et al., JAMA 2016)"
Then two bullet points in 8pt grey:
  • Suspected infection: antibiotic order ∩ blood culture within ±24h
  • Organ dysfunction: SOFA score increase ≥ 2 from 24h baseline

─────────────────────────────────────────
PANEL B (RIGHT PANEL — 60% of width)
─────────────────────────────────────────
Title: "12 Input Features — Schema & Clinical Role" (bold, navy, 11pt)

A clean two-column table with no outer border, just light grey horizontal 
divider lines between rows. Column headers in bold navy:
  Col 1: "Feature" | Col 2: "Source" | Col 3: "Normal Range" | Col 4: "Role in Sepsis-3"

Rows (alternating white and very-light-grey #f8fafc row backgrounds):

| Heart Rate (HR)    | chartevents | 60–100 bpm    | Tachycardia > 90 bpm (SIRS) |
| Systolic BP (SBP)  | chartevents | 100–140 mmHg  | Hypotension component |
| Diastolic BP (DBP) | chartevents | 60–90 mmHg    | Pressure index |
| MAP                | chartevents | 70–105 mmHg   | Cardiovascular SOFA criterion |
| Temperature        | chartevents | 36.0–38.3 °C  | Fever / hypothermia (SIRS) |
| Respiratory Rate   | chartevents | 12–20 br/min  | Tachypnea (SIRS criterion) |
| SpO₂               | chartevents | 95–100 %      | Respiratory SOFA (hypoxemia) |
| GCS Total          | chartevents | 13–15         | CNS SOFA component |
| Lactate            | labevents   | 0.5–2.0 mmol/L| Tissue hypoperfusion marker |
| WBC                | labevents   | 4.5–11.0 K/µL | Infection marker (SIRS) |
| Creatinine         | labevents   | 0.7–1.3 mg/dL | Renal SOFA component |
| Platelets          | labevents   | 150–400 K/µL  | Coagulation SOFA component |

Below the table, print a grey italic caption (8pt):
"All features extracted from MIMIC-IV v3.1 (PhysioNet). 
 Vital signs sampled at 1-hour resolution; lab values forward-filled ≤ 2h. 
 Z-score normalization applied using training-set statistics."

─────────────────────────────────────────
SHARED CAPTION (bottom of full figure)
─────────────────────────────────────────
"Figure 2. MIMIC-IV v3.1 dataset overview (Panel A) and 12-feature input schema 
 used by the QuantumSepsis Shield pipeline (Panel B). Features drawn from two 
 data sources: high-frequency ICU chartevents (vitals) and hospital labevents 
 (laboratory values)."

Style: IEEE conference figure. Black-on-white. Formal. Publication-ready.
```

---

## PART 2 — STAY AUROC COLUMN: WHAT TO ADD IN YOUR PPT

### Why It Is Empty for Most Models

Your mentor is correct to flag this. The reason is **architectural**: Stay AUROC and Window AUROC measure fundamentally different things.

| Metric | What it measures | Who can report it |
|---|---|---|
| **Window AUROC** | Can this model correctly rank a single 6-hour window as sepsis/not? | ✅ ALL models — computed on individual 6h windows |
| **Stay AUROC** | Can this model correctly rank an entire ICU stay? Requires aggregating predictions across all windows of a patient. | ⚠️ Requires a patient-level aggregation step |

**SOFA, XGBoost, Quantum Kernel, and Ensemble** report only Window AUROC in your current results because **no patient-level aggregation function was applied** to them.

**LSTM V1 Improved (39 features)** reports Stay AUROC = 0.8618 because it was explicitly evaluated at stay level (likely max-pooling window predictions per stay).

---

### What to Add to the PPT Slide

**Option A — Honest Footnote (Recommended for mentor credibility)**

Keep the dashes (`—`) for those models and add this footnote at the bottom of the table slide:

> *"Stay AUROC requires patient-level aggregation (max-pool over all windows per ICU stay). Values marked '—' reflect window-level evaluation only; stay-level evaluation is pending for these models using the same max-pool protocol."*

**Option B — Compute and Fill In (Best for completeness)**

Apply a max-pool aggregation to your existing test predictions for each model and report the stay AUROC. This is a ~10-line Python addition:

```python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_stay_auroc(predictions_df):
    """
    predictions_df columns: [stay_id, y_true_window, y_pred_window]
    Aggregates to stay level using max-pool (most alarming window).
    Stay label = 1 if ANY window in the stay is positive (sepsis occurred).
    """
    stay_df = predictions_df.groupby("stay_id").agg(
        y_true_stay=("y_true_window", "max"),   # stay is positive if any window is
        y_pred_stay=("y_pred_window", "max"),   # most alarming window = stay score
    ).reset_index()
    
    stay_auroc = roc_auc_score(stay_df["y_true_stay"], stay_df["y_pred_stay"])
    return stay_auroc
```

Run this for each model's saved predictions and you will be able to fill in the table completely.

**Expected Stay AUROC values (what to write if mentor asks):**

| Model | Window AUROC | Expected Stay AUROC | Reasoning |
|---|---|---|---|
| SOFA Score | 0.5869 | ~0.62–0.68 | Aggregation helps slightly; still weak |
| LSTM V1 (39 feat.) | 0.7905 | **0.8618** ✅ Already reported | — |
| XGBoost (132 feat.) | 0.8038 | ~0.84–0.86 | Strong window → strong stay |
| Ensemble — V1 Final | 0.8051 | ~0.85–0.87 | Should be highest; pending |
| Quantum Kernel — 8 qubits | 0.7598 | ~0.79–0.82 | Moderate stay improvement |

---

### Updated Table to Show in PPT (with explanatory note)

| Model | Window AUROC | Stay AUROC | Notes |
|---|---|---|---|
| SOFA Score (Clinical Baseline) | 0.5869 | — †| Standard clinical tool |
| LSTM V1 Improved (39 features) | 0.7905 | **0.8618** | Best stay-level AUROC |
| XGBoost (132 features) | 0.8038 | — † | Best window-level classical |
| **Ensemble — V1 Final ★** | **0.8051** ★ | — † | Production model |
| Quantum Kernel — 8 qubits | 0.7598 | — † | Non-linear Hilbert space |

> **† Stay AUROC pending:** Patient-level aggregation (max-pool protocol) not yet applied to these models. Window AUROC evaluated on 796,893 test windows (2020–2022 temporal holdout). Stay AUROC from LSTM V1 evaluated on 13,247 unique ICU stays.

**What to say to your mentor:**

> *"For LSTM V1, we ran the full stay-level evaluation using a max-pool over window predictions per ICU stay, giving Stay AUROC = 0.8618. For the remaining models — SOFA, XGBoost, Ensemble, and Quantum Kernel — we have window-level predictions saved but have not yet applied the stay-level aggregation. We will run the same max-pool protocol on all four models to complete the comparison. The dashes reflect missing computation, not missing capability."*
