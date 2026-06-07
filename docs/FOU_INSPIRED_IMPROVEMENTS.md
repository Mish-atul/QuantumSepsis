# FOU-Inspired Improvements for QuantumSepsis

## Overview

This document describes the enhancements made to the QuantumSepsis project based on the methodology from:

**"Integrating Medical Domain Knowledge for Early Diagnosis of Fever of Unknown Origin: An Interpretable Hierarchical Multimodal Neural Network Approach"**  
*Wang et al., IEEE Journal of Biomedical and Health Informatics, Vol. 27, No. 11, November 2023*

**Paper Results:** AUROC 0.7809-0.9035 across 5 hierarchical tasks  
**Our Baseline:** AUROC 0.7891 (original LSTM), 0.8038 (XGBoost)  
**Target:** Beat both baselines and approach paper's performance

---

## Key Innovations from FOU Paper

### 1. Hierarchical Classification Framework

**FOU Paper Approach:**
- Decomposed FUO diagnosis into 5 local classification tasks
- One local classifier per parent node in disease hierarchy
- Top-down reasoning framework (Td-HRF)
- Each task more balanced than flat multi-class

**Our Adaptation for Sepsis:**
```
Level 1 (Root): Sepsis vs No Sepsis
    ├─ Level 2: Severe Sepsis vs Non-Severe
    │   └─ Level 3: Septic Shock vs Non-Shock
```

**Benefits:**
- Better class balance at each level
- Specialized classifiers per severity stage
- Clinically interpretable decision path
- Reduced error propagation vs flat classification

### 2. Spatial Attention Mechanism

**FOU Paper Implementation:**
- Attention over time series features (not just time steps)
- Learns which clinical variables are most informative
- Formula: `ε_t = softmax(w_t · x_t)` where x_t are feature values at time t

**Our Implementation:**
```python
class SpatialAttention(nn.Module):
    def __init__(self, n_features: int, attention_dim: int = 32):
        self.attention = nn.Sequential(
            nn.Linear(n_features, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, n_features),
        )
    
    def forward(self, x):
        scores = self.attention(x)  # (batch, seq_len, n_features)
        weights = F.softmax(scores, dim=-1)
        return x * weights, weights
```

**Clinical Value:**
- Identifies which vitals (HR, BP, lactate) drive predictions
- Time-varying feature importance (e.g., lactate more important at hour 4)
- Interpretable for clinicians

### 3. Multimodal Fusion Architecture

**FOU Paper:**
- Tabular data (demographics, labs) → DNN
- Clinical notes (text) → NLP extraction + one-hot
- Time series (vitals) → GRU-D with attention
- Late fusion via concatenation

**Our Architecture:**
```
Static Pathway:  (batch, static_dim) → FC(128) → FC(64)
                                                    ↓
Temporal Pathway: (batch, 6, 12) → SpatialAttn → BiLSTM → TemporalAttn
                                                    ↓
                                    Fusion: Concat → FC(64) → Embedding(16)
                                                    ↓
                            Hierarchical Heads: L1, L2, L3
```

**Improvements Over Baseline:**
- Baseline: Only temporal features
- FOU-inspired: Static + temporal fusion
- Better representation learning
- Captures both snapshot (labs) and trajectory (vitals)

### 4. GRU-D for Irregular Time Series

**FOU Paper:**
- Used GRU-D (Gated Recurrent Unit with Decay)
- Handles missing values via masking and time intervals
- Captures informative missingness patterns

**Our Enhancement:**
- Already using BiLSTM (similar capacity)
- Added spatial attention (FOU innovation)
- Could further enhance with GRU-D decay mechanism

### 5. Layer-wise Relevance Propagation (LRP)

**FOU Paper:**
- Used LRP for static feature attribution
- Attention weights for temporal features
- Population-level and instance-level explanations

**Our Implementation:**
- Spatial attention → feature importance per time step
- Temporal attention → time step importance
- Can add LRP for static pathway (future work)

---

## Architecture Comparison

### Original SepsisLSTM
```
Input (6, 12) → LayerNorm → BiLSTM → TemporalAttn → FC → Embedding(16) → Classifier
Parameters: ~420K
AUROC: 0.7891
```

### FOU-Inspired HierarchicalSepsisLSTM
```
Temporal: (6, 12) → LayerNorm → SpatialAttn → BiLSTM → TemporalAttn → (256)
                                                                         ↓
Static: (static_dim) → FC(128) → FC(64) ────────────────────────────────┘
                                                                         ↓
                                            Fusion → FC(64) → Embedding(16)
                                                                         ↓
                                    Hierarchical Heads: L1, L2, L3 (3 classifiers)
Parameters: ~450K (with static_dim=20)
Expected AUROC: 0.82-0.85+ (target)
```

**Key Differences:**
1. ✅ Spatial attention on features
2. ✅ Static feature pathway
3. ✅ Hierarchical classification (3 levels)
4. ✅ Multi-task learning
5. ✅ Enhanced interpretability

---

## Training Strategy

### FOU Paper Training
- Independent training of each local classifier
- Hyperparameter tuning via Optuna (100 trials)
- Early stopping on validation AUROC
- Cosine annealing scheduler

### Our Training
```python
# Multi-task loss with level weights
loss = 1.0 * loss_L1 + 0.5 * loss_L2 + 0.25 * loss_L3

# Asymmetric focal loss at each level
focal_loss = -α_pos × (1-p)^γ × log(p)  [for positives]
           + -α_neg × p^γ × log(1-p)    [for negatives]

# α_pos=0.9, α_neg=0.1 → 9:1 FN:FP penalty ratio
```

**Improvements:**
- Joint training across levels (vs independent)
- Shared embedding layer (transfer learning)
- Level-weighted loss (prioritize L1)
- Same focal loss as baseline (proven effective)

---

## Quantum Kernel Integration

### FOU Paper: Classical Only
- No quantum components
- Standard SVM with RBF kernel for some experiments

### Our Quantum Enhancement
```
LSTM Embeddings (N, 16) → PCA (16→8) → Quantum Kernel → QSVM
                                         ↓
                            ZZFeatureMap (8 qubits, 2 reps)
                            Fidelity: K(x,y) = |⟨φ(x)|φ(y)⟩|²
```

**Balanced Subsampling:**
- Full dataset: 4M windows (99%+ negative)
- Subsample: 2000 balanced (1000 pos + 1000 neg)
- Tractable kernel matrix: 2000×2000
- Inference: Support vectors only (~1700)

**Expected Improvement:**
- Quantum kernel captures non-linear structure
- Better separation in Hilbert space
- Target: +0.01-0.02 AUROC over classical

---

## Expected Performance Gains

### Baseline Performance
| Model | AUROC | AUPRC | Notes |
|-------|-------|-------|-------|
| SOFA Threshold | 0.5869 | - | Clinical baseline |
| Original LSTM | 0.7891 | 0.0519 | Our baseline |
| XGBoost | 0.8038 | - | Best baseline |
| FOU Paper Best | 0.9035 | - | Target benchmark |

### Expected FOU-Inspired Performance
| Component | Expected AUROC | Improvement | Rationale |
|-----------|---------------|-------------|-----------|
| Hierarchical LSTM L1 | 0.82-0.84 | +0.03-0.05 | Better class balance, specialized classifiers |
| + Spatial Attention | +0.01-0.02 | - | Feature-level importance learning |
| + Static Features | +0.01-0.02 | - | Multimodal fusion (if static_dim > 0) |
| Quantum Kernel | 0.83-0.85 | +0.01-0.02 | Non-linear kernel in Hilbert space |
| **Total Expected** | **0.84-0.86** | **+0.05-0.07** | Combined improvements |

### Why We Might Not Reach 0.9035
1. **Different domain:** FUO vs Sepsis (different data characteristics)
2. **Data quality:** MIMIC-IV vs their Chinese hospital data
3. **Class imbalance:** Our window-level imbalance is more severe
4. **Feature richness:** FOU paper had clinical notes (text), we don't
5. **Observation window:** FOU used 48h, we use 6h windows

### Realistic Target
- **Conservative:** Beat XGBoost (0.8038) → AUROC 0.81-0.82
- **Optimistic:** Approach 0.85 with all enhancements
- **Stretch:** 0.87+ if static features + quantum kernel synergize well

---

## Implementation Details

### File Structure
```
src/models/
├── hierarchical_lstm.py          # FOU-inspired architecture
├── lstm.py                        # Original baseline
└── quantum_kernel.py              # Phase 2 (unchanged)

src/training/
├── train_hierarchical.py          # New training pipeline
└── train_lstm.py                  # Original training

scripts/
└── run_fou_inspired_pipeline.py   # End-to-end runner
```

### Key Classes

**HierarchicalSepsisLSTM:**
- `SpatialAttention`: Feature-level attention
- `TemporalAttention`: Time-level attention (original)
- `head_level1/2/3`: Hierarchical classifiers
- `forward(level=1/2/3)`: Multi-level inference

**HierarchicalLoss:**
- Multi-task loss combiner
- Separate focal loss per level
- Configurable level weights

**HierarchicalTrainer:**
- Joint training across levels
- Multi-metric validation
- Embedding extraction for quantum kernel

### Hyperparameters

**Unchanged from Baseline:**
- LSTM hidden_dim: 128
- LSTM layers: 2
- Embedding dim: 16 (for quantum kernel)
- Dropout: 0.3
- Learning rate: 0.001
- Focal α_pos: 0.9, α_neg: 0.1, γ: 2.0

**New Hyperparameters:**
- Spatial attention dim: 32
- Static encoder: [static_dim → 128 → 64]
- Level weights: [1.0, 0.5, 0.25]
- Hierarchy levels: 3

---

## Running the Pipeline

### Quick Start (No Quantum)
```bash
# Train hierarchical LSTM only (fastest)
python scripts/run_fou_inspired_pipeline.py \
    --data data/processed/features.h5 \
    --skip-quantum

# Expected time: ~3 hours on A100
# Expected AUROC: 0.82-0.84
```

### Full Pipeline (With Quantum)
```bash
# Full pipeline with RBF kernel (recommended)
python scripts/run_fou_inspired_pipeline.py \
    --data data/processed/features.h5 \
    --quantum-samples 2000

# Expected time: ~4 hours on A100
# Expected AUROC: 0.83-0.85
```

### With Qiskit Quantum Kernel (Very Slow)
```bash
# Use actual quantum kernel (research only)
python scripts/run_fou_inspired_pipeline.py \
    --data data/processed/features.h5 \
    --use-qiskit \
    --quantum-samples 1000

# Expected time: ~6-8 hours (quantum simulation is slow)
# Expected AUROC: 0.83-0.85 (similar to RBF)
```

### Single Level Only (Baseline Comparison)
```bash
# Train only level 1 for fair comparison
python scripts/run_fou_inspired_pipeline.py \
    --data data/processed/features.h5 \
    --no-hierarchy \
    --skip-quantum

# Expected time: ~2 hours
# Expected AUROC: 0.81-0.83 (spatial attention benefit only)
```

---

## Evaluation Metrics

### Primary Metrics
- **AUROC** (Area Under ROC Curve): Main metric for comparison
- **AUPRC** (Area Under Precision-Recall): Important for imbalanced data
- **Sensitivity @ 95% Specificity**: Clinical operating point

### Hierarchical Metrics
- **AUROC L1**: Sepsis detection (most important)
- **AUROC L2**: Severe sepsis classification
- **AUROC L3**: Septic shock prediction

### Interpretability Metrics
- **Spatial attention weights**: Feature importance per time step
- **Temporal attention weights**: Time step importance
- **Support vector ratio**: Quantum kernel efficiency

---

## Limitations and Future Work

### Current Limitations
1. **No clinical notes:** FOU paper used text, we don't have it
2. **Synthetic hierarchy:** Real sepsis labels aren't hierarchical (yet)
3. **Static features:** Currently set to 0 (temporal only)
4. **Quantum kernel:** Limited to 2000 samples due to compute

### Future Enhancements
1. **Add static features:** Demographics, admission labs
2. **True hierarchical labels:** SOFA-based severity staging
3. **GRU-D integration:** Replace LSTM with GRU-D for better missing value handling
4. **LRP for static pathway:** Complete interpretability
5. **Larger quantum kernel:** 5000-10000 samples if compute allows
6. **Ensemble:** Combine hierarchical LSTM + quantum kernel predictions

---

## References

1. **FOU Paper:**  
   Wang, Z., Liu, J., Tian, Y., Zhou, T., Liu, Q., Qiu, Y., & Li, J. (2023).  
   *Integrating Medical Domain Knowledge for Early Diagnosis of Fever of Unknown Origin: An Interpretable Hierarchical Multimodal Neural Network Approach.*  
   IEEE Journal of Biomedical and Health Informatics, 27(11), 5237-5248.

2. **GRU-D:**  
   Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018).  
   *Recurrent Neural Networks for Multivariate Time Series with Missing Values.*  
   Scientific Reports, 8(1), 6085.

3. **Asymmetric Focal Loss:**  
   Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).  
   *Focal Loss for Dense Object Detection.*  
   ICCV 2017.

4. **Quantum Kernel Methods:**  
   Havlíček, V., Córcoles, A. D., Temme, K., et al. (2019).  
   *Supervised learning with quantum-enhanced feature spaces.*  
   Nature, 567(7747), 209-212.

---

## Contact

For questions about this implementation:
- See `AGENTS.md` for full project documentation
- Check `docs/FINAL_RESULTS.md` for baseline results
- Review `src/models/hierarchical_lstm.py` for architecture details

**Last Updated:** May 12, 2026
