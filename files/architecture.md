# QuantumSepsis Shield — Technical Architecture Specification

> **Logline:** *"An adversarially-safe quantum-classical sentinel that detects sepsis 4 hours early — and never stays silent when a patient is dying."*

---

## 1. System Overview

QuantumSepsis Shield is a **5-layer agentic pipeline** that runs in 15-minute cycles from ICU admission, combining classical temporal modeling, quantum kernel methods, conformal prediction, and adversarial safety mechanisms for early sepsis detection.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QuantumSepsis Shield Pipeline                    │
│                                                                     │
│  ┌───────────┐   ┌───────────┐   ┌───────────────┐   ┌──────────┐ │
│  │  Layer 1   │──▶│  Layer 2   │──▶│   Layer 3     │──▶│ Layer 4a │ │
│  │ Monitoring │   │   LSTM     │   │ Quantum Kernel│   │Conformal │ │
│  │  Agents    │   │  Encoder   │   │   Module      │   │Prediction│ │
│  └───────────┘   └───────────┘   └───────────────┘   └──────────┘ │
│       │                                                      │      │
│       │          ┌───────────┐                          ┌────▼────┐ │
│       └─────────▶│  Layer 4b  │─────────────────────────▶│ Layer 5 │ │
│                  │ Red Team   │                          │Orchestr.│ │
│                  │  Agent     │                          └─────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer-by-Layer Technical Specification

### 2.1 Layer 1 — Monitoring Agents (Per-Patient)

**Purpose:** Real-time data ingestion, alignment, and windowing for each ICU patient.

**Input Variables (12 features):**

| Index | Variable       | Source Table   | Item IDs / Method                     | Unit      |
|-------|---------------|---------------|---------------------------------------|-----------|
| 0     | Heart Rate    | chartevents   | 211, 220045                           | bpm       |
| 1     | SBP           | chartevents   | 51, 442, 455, 6701, 220179, 220050    | mmHg      |
| 2     | DBP           | chartevents   | 8368, 8440, 8441, 8555, 220180, 220051| mmHg      |
| 3     | MAP           | chartevents   | 52, 456, 6702, 220052, 220181         | mmHg      |
| 4     | Temperature   | chartevents   | 223762, 226329                        | °C        |
| 5     | Resp. Rate    | chartevents   | 615, 618, 220210, 224690              | br/min    |
| 6     | SpO2          | chartevents   | 646, 220277                           | %         |
| 7     | GCS Total     | chartevents   | 198, 226755, 227013                   | score     |
| 8     | Lactate       | labevents     | 50813                                 | mmol/L    |
| 9     | WBC           | labevents     | 51301                                 | K/uL      |
| 10    | Creatinine    | labevents     | 50912                                 | mg/dL     |
| 11    | Platelets     | labevents     | 51265                                 | K/uL      |

**Data Alignment:**
- Bin size: 1 hour
- Strategy: Within each 1-hour bin, take the **median** of all available measurements
- Forward fill: Up to 2 consecutive hours maximum
- Fallback: Per-variable **training-set median** imputation
- No backward fill (prevents data leakage)

**Output:** Sliding window tensor of shape `(6, 12)` — 6 hourly time steps × 12 features

**Normalization:**
- Per-variable z-score normalization: `x_norm = (x - μ_train) / σ_train`
- μ and σ computed **only** from training set
- Stored and applied identically to validation and test sets

---

### 2.2 Layer 2 — Classical LSTM (Temporal Encoder)

**Purpose:** Encode the multivariate 6-hour vital sign window into a fixed-dimensional latent embedding that captures temporal dynamics and inter-variable interactions.

**Architecture:**

```
Input: (batch_size, 6, 12)
         │
    ┌────▼────┐
    │ LayerNorm│  ← Pre-normalization for training stability
    │ (6, 12) │
    └────┬────┘
         │
    ┌────▼─────────────┐
    │ Bidirectional LSTM│
    │ layers=2          │
    │ hidden_dim=128    │
    │ dropout=0.3       │
    │ input_size=12     │
    └────┬─────────────┘
         │
    Output: (batch, 6, 256)  ← 128*2 for bidirectional
         │
    ┌────▼────┐
    │ Attention│  ← Temporal attention over 6 time steps
    │ Pooling  │
    └────┬────┘
         │
    (batch, 256)
         │
    ┌────▼────┐
    │ FC: 256→64│
    │ ReLU     │
    │ Dropout  │
    └────┬────┘
         │
    ┌────▼────┐
    │ FC: 64→16│  ← Latent embedding dimension = quantum qubits
    │ Tanh     │  ← Bounded [-1,1] for quantum encoding
    └────┬────┘
         │
    Output: (batch, 16)  ← Latent embedding vector
         │
    ┌────▼────┐
    │ FC: 16→1 │  ← Classical prediction head (for standalone + pre-training)
    │ Sigmoid  │
    └─────────┘
```

**Key Design Decisions:**
- **Bidirectional LSTM:** Captures both forward temporal patterns (deterioration trends) and backward context (baseline deviations)
- **Temporal Attention:** Learnable attention weights over the 6 time steps — allows the model to focus on critical deterioration windows rather than averaging
- **Embedding dim = 16:** Matched to quantum circuit qubit count for seamless encoding
- **Tanh activation on embedding:** Bounds values to [-1, 1] for stable angle encoding into quantum circuits
- **Dual output:** The classical head produces standalone predictions during pre-training; the 16-dim embedding feeds the quantum kernel module

**Hyperparameters:**

| Parameter          | Value                  |
|--------------------|------------------------|
| Input shape        | (batch, 6, 12)         |
| LSTM layers        | 2                      |
| LSTM hidden dim    | 128                    |
| Bidirectional      | True                   |
| LSTM dropout       | 0.3                    |
| Attention dim      | 64                     |
| FC1 dim            | 256 → 64               |
| FC2 dim            | 64 → 16                |
| Output activation  | Tanh (embedding), Sigmoid (classification) |
| Total parameters   | ~420K                  |

**Loss Function — Asymmetric Focal Loss:**

```
L(p, y) = -α_t · (1 - p_t)^γ · log(p_t)

Where:
  - For y=1 (sepsis):     α_t = 0.9,  γ = 2.0   (FN penalty = 10×)
  - For y=0 (non-sepsis): α_t = 0.1,  γ = 2.0
  - p_t = p if y=1, else (1-p)
```

**Training Configuration:**

| Parameter              | Value                     |
|------------------------|---------------------------|
| Optimizer              | AdamW (weight_decay=1e-4) |
| Initial LR             | 1e-3                      |
| LR Scheduler           | Cosine Annealing (T_max=50)|
| Batch size             | 256                       |
| Max epochs             | 100                       |
| Early stopping         | Patience=10 on val AUROC  |
| Gradient clipping      | max_norm=1.0              |
| Random seed            | 42                        |

---

### 2.3 Layer 3 — Quantum Kernel Module (IBM Qiskit)

**Purpose:** Map LSTM embeddings into an exponentially large quantum Hilbert space where sepsis deterioration patterns form geometrically separated clusters, enabling superior classification in small-data regimes.

**Quantum Circuit Design:**

```
|0⟩ ─ H ─ Rz(x₁) ─ ●──── ─ Rz(x₁·x₂) ── ●──── ─ Rz(x₁) ─
|0⟩ ─ H ─ Rz(x₂) ─ ┼──●─ ─ Rz(x₂·x₃) ── ┼──●─ ─ Rz(x₂) ─
|0⟩ ─ H ─ Rz(x₃) ─ ┼──┼─ ─ Rz(x₃·x₄) ── ┼──┼─ ─ Rz(x₃) ─
...                  (ZZ entanglement)      (reps=2)
|0⟩ ─ H ─ Rz(x₈) ─ ───●─ ─ Rz(x₇·x₈) ── ───●─ ─ Rz(x₈) ─
```

**Circuit Specification:**

| Parameter            | Value                        |
|----------------------|------------------------------|
| Qubits               | 8 (PCA-reduced from 16-dim)  |
| Feature map          | ZZFeatureMap                 |
| Encoding             | Angle encoding via Rz gates  |
| Entanglement         | Linear ZZ entanglement       |
| Repetitions (reps)   | 2                            |
| Circuit depth        | ~40 gates                    |
| Kernel function      | K(x,y) = |⟨Φ(x)|Φ(y)⟩|²    |

**Why 8 Qubits (Not 16):**
- 16-dim LSTM embedding → PCA to 8 principal components (retaining >95% variance)
- 8 qubits sufficient for ZZFeatureMap with reps=2 to generate 2^8 = 256 basis states
- Deeper circuits (16q) suffer from barren plateaus and noise on NISQ hardware
- 8 qubits achievable on IBM Quantum Eagle processors (127 qubits available)

**Quantum Advantage Argument (Small-Data Regime):**

The quantum kernel provides advantage when:
1. The quantum feature map Φ generates a feature space that is **classically intractable** to simulate (Havlíček et al., Nature 2019)
2. The **geometric margin** in quantum Hilbert space is larger than in any efficiently computable classical kernel
3. For n < 500 training samples, the quantum kernel's implicit regularisation through the structured feature map prevents overfitting — unlike RBF/polynomial kernels that overfit small datasets

Formal argument (Schuld & Killoran, PRL 2019): The ZZFeatureMap generates features equivalent to an exponential number of Fourier components. Classical simulation requires O(2^n) terms, while the quantum circuit evaluates this in O(poly(n)) depth.

**Implementation:**
```python
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

feature_map = ZZFeatureMap(feature_dimension=8, reps=2, entanglement='linear')
kernel = FidelityQuantumKernel(feature_map=feature_map)
K_train = kernel.evaluate(X_train_pca)  # (n, n) kernel matrix
```

**Output:** Continuous risk score [0, 1] from QSVM, plus calibrated probability via Platt scaling.

---

### 2.4 Layer 4a — Conformal Prediction Layer

**Purpose:** Wrap the quantum risk score with statistically guaranteed confidence intervals, enabling uncertainty-aware clinical decision-making.

**Method:** Split Conformal Prediction (Vovk et al., 2005)

**Algorithm:**

```
1. Hold out calibration set C from training data (20% of positives)
2. Compute nonconformity scores on C:
     s_i = 1 - f(x_i)   for y_i = 1 (sepsis)
     s_i = f(x_i)       for y_i = 0 (non-sepsis)
3. For new sample x_test:
     - Compute f(x_test) = risk_score
     - Prediction set at level α:
       Ŷ(x_test) = {y : s(x_test, y) ≤ quantile(1-α, {s_i})}
4. Confidence interval:
     lower = max(0, risk_score - q_α)
     upper = min(1, risk_score + q_α)
     where q_α = (1-α)-quantile of calibration scores
```

**Configuration:**

| Parameter            | Value                     |
|----------------------|---------------------------|
| Coverage guarantee   | 1 - α = 0.90 (90%)       |
| Calibration set size | 20% of training positives |
| Method               | Split conformal           |
| Library              | MAPIE 1.0+                |

**Escalation Rule:**
- If `(upper_bound - lower_bound) > 0.4`: Uncertainty too high → escalate alert tier regardless of point estimate
- This prevents overconfident misses and forces human review on ambiguous cases

---

### 2.5 Layer 4b — Red Team Agent (Adversarial Safety)

**Purpose:** Independent, non-overridable safety layer that detects sepsis indicators through deterministic clinical rules. Cannot be suppressed by ML model output.

**Tripwire Definitions:**

| Tripwire ID | Metric           | Condition                              | Clinical Basis          |
|-------------|-----------------|----------------------------------------|-------------------------|
| TW-TEMP-LO  | Temperature     | < 36.0°C                              | Hypothermia (septic)    |
| TW-TEMP-HI  | Temperature     | > 38.3°C                              | Fever (SIRS criterion)  |
| TW-HR       | Heart Rate      | > 90 bpm AND ∆HR/∆t > 5 bpm/hr       | Tachycardia + trend     |
| TW-RR       | Resp. Rate      | > 20 breaths/min                       | Tachypnea (SIRS)        |
| TW-MAP      | MAP             | < 70 mmHg                             | Hypotension (Sepsis-3)  |
| TW-MENTAL   | GCS / Mental    | GCS < 14 OR nurse flag                 | Altered mental status   |

**Escalation Logic (Pseudocode):**

```python
def evaluate(vitals_window: np.ndarray) -> Tuple[bool, List[str], str]:
    active_tripwires = []

    latest = vitals_window[-1]  # Most recent readings
    hr_trend = compute_trend(vitals_window[:, HR_IDX])

    if latest[TEMP_IDX] < 36.0 or latest[TEMP_IDX] > 38.3:
        active_tripwires.append("TW-TEMP")
    if latest[HR_IDX] > 90 and hr_trend > 5.0:
        active_tripwires.append("TW-HR")
    if latest[RR_IDX] > 20:
        active_tripwires.append("TW-RR")
    if latest[MAP_IDX] < 70:
        active_tripwires.append("TW-MAP")
    if latest[GCS_IDX] < 14:
        active_tripwires.append("TW-MENTAL")

    n_active = len(active_tripwires)

    if n_active >= 2:
        return (True, active_tripwires, "CRITICAL")
    elif n_active == 1:
        return (True, active_tripwires, "AMBER")
    else:
        return (False, active_tripwires, "WATCH")
```

**Key Property:** The Red Team Agent operates **independently** of the ML pipeline. Even if the quantum kernel outputs risk = 0.0, two simultaneous tripwires force a CRITICAL alert.

---

### 2.6 Layer 5 — Intervention Orchestrator + Outcome Learning Agent

**Orchestrator Decision Matrix:**

| Quantum Risk | Conformal Width | Tripwires | Confidence | → Action Level |
|-------------|----------------|-----------|------------|----------------|
| < 0.3       | < 0.4          | 0         | Any        | WATCH          |
| 0.3 – 0.6  | Any            | 0-1       | 0.5–0.8    | AMBER          |
| > 0.6       | Any            | Any       | > 0.8      | CRITICAL       |
| Any         | > 0.4          | Any       | < 0.5      | AMBER + Review |
| Any         | Any            | ≥ 2       | Any        | CRITICAL       |

**Action Protocols:**

| Level    | Actions Triggered                                          |
|----------|------------------------------------------------------------|
| WATCH    | Dashboard refresh, reassess in 15 min                      |
| AMBER    | Concurrent lactate + PCT + blood culture orders; physician notify |
| CRITICAL | Immediate attending page + full sepsis bundle protocol      |

**Confidence-Gated Clinical Fast-Track (Novel):**

| Confidence | Risk   | Action                                              |
|------------|--------|-----------------------------------------------------|
| > 0.80     | > 0.6  | SKIP preliminary CBC → directly order PCT + culture + vasopressor protocol |
| 0.50–0.80  | > 0.3  | Standard AMBER parallel ordering                     |
| < 0.50     | Any    | Flag for manual clinician review + uncertainty report |

**Outcome Learning Agent:**
- Post-case analysis: 72 hours after alert, check actual sepsis outcome
- Adaptive threshold tuning: Bayesian update of risk thresholds per hospital unit
- Update frequency: Weekly, with minimum 20 resolved cases per unit
- Update rule: `threshold_new = threshold_old + η * (target_sensitivity - observed_sensitivity)`
  where η = 0.05 (conservative learning rate)

---

## 3. Novel Architectural Contributions ("Beyond Hybrid")

### 3.1 Novelty 1: Quantum-Calibrated Conformal Prediction (QCCP)

**Innovation:** Instead of applying standard conformal prediction on classical model scores, we compute nonconformity scores **directly in the quantum kernel space**. The nonconformity measure becomes:

```
s(x_new) = 1 - max_j K(x_new, c_j)
```

where c_j are learned sepsis centroid states in Hilbert space. This yields tighter prediction sets because the quantum kernel captures non-linear structure that inflates classical conformal widths.

**Why novel:** No existing work combines split conformal prediction with quantum kernel nonconformity scores. Classical conformal methods use Euclidean or model-score-based nonconformity; our approach uses Hilbert-space distance.

### 3.2 Novelty 2: Adversarial Tripwire-Gated Asymmetric Safety

**Innovation:** The Red Team Agent doesn't just override — it **modifies the loss function during online learning**. When a tripwire fires and the model predicted low risk, the outcome learning agent doubles the asymmetric penalty for that patient profile, creating an **adversarially adaptive loss surface** that hardcodes clinical safety constraints into the model's gradient updates.

### 3.3 Novelty 3: Confidence-Gated Diagnostic Fast-Tracking

**Innovation:** Using the conformal prediction interval width as a proxy for model confidence to **dynamically skip low-value diagnostic steps** and immediately trigger expensive but definitive tests. This eliminates the 4–6 hour sequential diagnostic delay described in clinical workflow studies.

**Complementarity:** These three innovations work as an integrated system:
- Quantum kernels provide the **feature space** (Novelty 1 leverages this for calibrated uncertainty)
- Calibrated uncertainty drives **clinical decision routing** (Novelty 3 uses conformal widths)
- Safety violations update the **model's learning objective** (Novelty 2 closes the feedback loop)

---

## 4. Tensor Shape Flow (End-to-End)

```
Raw vitals/labs (per patient, irregular timestamps)
    ↓ [Layer 1: Monitoring Agent]
(6, 12) — 6 hours × 12 features, z-normalized float32
    ↓ [Layer 2: LSTM Encoder]
(16,) — Tanh-bounded latent embedding
    ↓ [PCA reduction]
(8,) — 8 principal components
    ↓ [Layer 3: Quantum Kernel + QSVM]
(1,) — risk_score ∈ [0, 1]
    ↓ [Layer 4a: Conformal Prediction]
(3,) — (risk_score, lower_bound, upper_bound)
    ↓ [Layer 4b: Red Team Agent]    ← runs in PARALLEL
(bool, List[str], str) — (triggered, active_tripwires, override_level)
    ↓ [Layer 5: Orchestrator]
{alert_level: str, actions: List[str], confidence: float, explain: dict}
```

---

## 5. Technology Stack

| Component           | Library/Tool                    | Version  |
|---------------------|---------------------------------|----------|
| Language            | Python                          | 3.10+    |
| Deep Learning       | PyTorch                         | 2.x      |
| Quantum Computing   | Qiskit                          | 1.x      |
| Quantum ML          | qiskit-machine-learning         | 0.7+     |
| Classical ML        | scikit-learn                    | 1.3+     |
| Conformal Pred.     | MAPIE                           | 1.0+     |
| Data Processing     | pandas, numpy                   | latest   |
| Database            | PostgreSQL / BigQuery           | —        |
| Experiment Tracking | Weights & Biases (wandb)        | latest   |
| Visualization       | matplotlib, seaborn             | latest   |
| GPU Compute         | CUDA 11.8+                      | —        |
| Quantum Hardware    | IBM Quantum (Eagle processors)  | —        |

---

## 6. Reproducibility Requirements

- All random seeds fixed at 42 (numpy, torch, qiskit)
- All experiments logged to W&B with full hyperparameter capture
- Data splits deterministic and saved as CSV manifests
- Model checkpoints saved at best validation AUROC
- Quantum circuit transpilation seed fixed for reproducibility
