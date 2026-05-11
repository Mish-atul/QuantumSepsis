# QuantumSepsis Shield — Demo Guide for Mentors

This guide provides exactly what numbers to enter into the **"Custom Input"** sliders (on both the Vercel React UI and the AWS Streamlit dashboard) to trigger specific, impressive reactions from the AI pipeline during a live presentation.

Both the React and Python simulators use the exact same clinical rules, so these values will work perfectly on both!

---

## 🟢 Scenario 1: The "Healthy/Stable" Baseline
*Use this to show the mentor what a low-risk, stable patient looks like. The Risk Gauge should be green and near 0-10%.*

*   **Heart Rate:** 75 bpm
*   **MAP:** 85 mmHg
*   **Temperature:** 36.8 °C
*   **Resp Rate:** 14 br/min
*   **SpO2:** 98 %
*   **GCS Total:** 15
*   **Lactate:** 1.0 mmol/L
*   **WBC:** 8.0 K/uL
*   **Creatinine:** 0.9 mg/dL
*   **Platelets:** 220 K/uL

**What the AI does:** Shows **NORMAL / LOW RISK**. All red team tripwires remain inactive.

---

## 🟡 Scenario 2: The "Early Warning" Catch
*Use this to demonstrate the model catching subtle, early signs of infection before a human doctor might notice. This highlights the predictive power of the LSTM/XGBoost ensemble.*

*   **Heart Rate:** 105 bpm *(Slightly elevated)*
*   **MAP:** 70 mmHg *(Borderline low)*
*   **Temperature:** 38.2 °C *(Mild fever)*
*   **Resp Rate:** 22 br/min *(Slightly elevated)*
*   **SpO2:** 94 %
*   **GCS Total:** 14 *(Mild confusion)*
*   **Lactate:** 2.2 mmol/L *(Slightly elevated)*
*   **WBC:** 12.5 K/uL *(Elevated)*
*   **Creatinine:** 1.2 mg/dL
*   **Platelets:** 150 K/uL

**What the AI does:** The gauge jumps to **AMBER / WATCH**. The Red Team agent might trigger 1 or 2 minor tripwires.

---

## 🔴 Scenario 3: The "Septic Shock" Fast-Track
*Use this to demonstrate the Red Team Agent taking autonomous control and bypassing the ML model for immediate clinical safety.*

*   **Heart Rate:** 135 bpm
*   **MAP:** 55 mmHg *(Severe hypotension)*
*   **Temperature:** 39.5 °C *(High fever)*
*   **Resp Rate:** 32 br/min
*   **SpO2:** 88 %
*   **GCS Total:** 11
*   **Lactate:** 5.5 mmol/L *(Critical)*
*   **WBC:** 22.0 K/uL
*   **Creatinine:** 2.5 mg/dL
*   **Platelets:** 80 K/uL

**What the AI does:** The gauge spikes to **CRITICAL**. The Orchestrator will show **FAST-TRACKED: Yes**, demonstrating that the deterministic Red Team override correctly identified severe shock and bypassed the ML latency.

---

## 🔵 Scenario 4: The "Hypothermic Sepsis" Edge Case
*Use this to show your mentor that the AI handles complex, non-standard presentations (where temperature drops instead of rising).*

*   **Heart Rate:** 55 bpm *(Bradycardia)*
*   **MAP:** 60 mmHg
*   **Temperature:** 34.5 °C *(Severe Hypothermia)*
*   **Resp Rate:** 26 br/min
*   **SpO2:** 90 %
*   **GCS Total:** 12
*   **Lactate:** 4.2 mmol/L
*   **WBC:** 3.5 K/uL *(Leukopenia - very low)*
*   **Creatinine:** 2.0 mg/dL
*   **Platelets:** 95 K/uL

**What the AI does:** The AI correctly identifies high risk despite the absence of a fever, proving the model learned complex multivariate relationships beyond basic SIRS criteria.

---

### 💡 Pro-Tips for the Demo Presentation:
1. **Start with "Simulated 15-Min Cycle" Mode:** Put the sliders on the "Early Warning" preset, and turn on the 15-min cycle on a fast speed (like 3 sec). Let the mentor watch the risk timeline chart grow dynamically.
2. **Switch to "Custom Input":** Pause the simulation, switch to the "Custom Input" scenario, and say: *"Let's test the AI's safety rails by manually inputting an extreme shock event."* Then rapidly drag the Lactate and MAP sliders to critical levels to show the Red Team tripwires lighting up instantly.
