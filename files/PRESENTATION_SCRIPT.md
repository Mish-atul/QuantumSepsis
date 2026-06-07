# QuantumSepsis Shield — Presentation Speaking Script
### Slides 10 Onwards | ~2 Minutes Per Slide

---

> **How to use this script:**
> Read it naturally — don't memorize word for word. The goal is to sound like you understand it deeply, not like you're reciting. Pause after each key point. Make eye contact. Use "we" throughout since this is a team project you led.

---

## SLIDE 10 —  System Architecture

*[This slide shows the 5-layer pipeline diagram]*

**Script (≈2 min):**

"So let me walk you through how we actually built this system. The core idea is a five-layer pipeline — and each layer has a very specific job.

It starts with raw ICU data. We take a **6-hour sliding window** of 12 clinical features — things like heart rate, mean arterial pressure, temperature, respiratory rate, SpO2, GCS, and key lab values like lactate and creatinine. All of this comes directly from MIMIC-IV's chartevents and labevents tables. Every window is z-score normalized using training-set statistics so the model never sees leakage from the test set.

That window goes into our **Bidirectional LSTM**. We chose a BiLSTM specifically because sepsis progression is temporal — the model needs to see how vitals are *trending*, not just what they are at any single moment. The LSTM has 128 hidden units across 2 layers with a temporal self-attention mechanism that learns which of the 6 hours in the window is most clinically significant. The output is a 16-dimensional embedding — a compact representation of the patient's physiological state.

This embedding feeds into our **ensemble**. We combine the LSTM with an XGBoost model trained on 132 hand-crafted features — statistical moments, rolling trends, interaction terms. The ensemble gives us a risk score, and critically, a **conformal prediction interval** — a guaranteed uncertainty band around that score.

From there, the risk score and interval go to our **Red Team Agent** — an independent safety layer — and finally to the **Confidence-Gated Orchestrator** which makes the final clinical decision. I'll go deeper into each of these on the next slides."

---

## SLIDE 11 —MIMIC-IV Dataset

*[This slide shows the dataset overview — MIMIC-IV scale, cohort derivation, feature schema]*

**Script (≈2 min):**

"Let me talk about the data foundation, because this is what makes the results credible.

We used **MIMIC-IV version 3.1** — the Medical Information Mart for Intensive Care, maintained by MIT and Beth Israel Deaconess Medical Center in Boston. It covers over **94,000 ICU stays** across multiple care units — medical ICU, surgical ICU, cardiac ICU — spanning data from 2008 to 2022.

For our cohort, we used the **Sepsis-3 definition from Singer et al., JAMA 2016**. This requires two simultaneous conditions: first, suspected infection — which we detect as the co-occurrence of an antibiotic prescription and a blood or body fluid culture within 24 hours of each other. Second, organ dysfunction — a SOFA score increase of at least 2 points from the patient's 24-hour baseline. We computed SOFA across all six components: respiratory, coagulation, liver, cardiovascular, CNS, and renal.

The result is approximately **22,000 to 28,000 Sepsis-3 positive ICU stays** — around a 23 to 30 percent prevalence within ICU admissions.

Our sliding window approach generated about **4.1 million training windows** and 797,000 test windows, each of shape 6 by 12 — 6 hourly time steps, 12 features. The positive rate at window level is around 1.4 percent, which is important context for our AUPRC numbers.

For the train-test split, we did a **temporal split** — data from 2008 to 2019 for training, and 2020 to 2022 strictly held out as the test set. This simulates real deployment where the model is trained on historical data and must generalize to future patients. We never touched the test set during development."

---

## SLIDE 12 — Model Architecture Deep Dive (LSTM + Quantum Kernel)

*[This slide shows the BiLSTM architecture and Quantum Kernel ZZFeatureMap diagram]*

**Script (≈2 min):**

"Now let me go into the technical architecture in more detail.

Our **BiLSTM** processes the 6-by-12 input window — bidirectional because early hours and late hours of the window are both informative. After the LSTM layers, we apply a **temporal self-attention mechanism** — this produces 6 attention weights, one per hour, and tells us which timestep drove the prediction. This is important for clinical interpretability: a clinician can see 'the model flagged this patient primarily because of what happened in hours 4 and 5 of the window.'

The LSTM outputs a **16-dimensional embedding** — this is the compressed physiological state vector. For the quantum component, we apply PCA to reduce this to 8 dimensions, which then feeds into our **quantum kernel**.

The quantum kernel uses Qiskit's **ZZFeatureMap** with 8 qubits and 2 repetitions. This encodes the 8-dimensional embedding into a quantum circuit, computes the fidelity-based kernel — which is essentially the inner product in quantum Hilbert space — and uses this as the similarity measure for a quantum support vector machine.

Why quantum? For large datasets classical kernels are fine. But for small ICU datasets — say, 200 to 500 patients from a single hospital — the quantum kernel maps into a **256-dimensional Hilbert space** through 2 to the power of 8 dimensions. This gives implicit regularization and finds geometric structure that classical RBF or polynomial kernels cannot. We validated this on a subset of the test data and found the quantum kernel AUROC at 0.7598 — competitive with classical LSTM — while contributing meaningfully to the ensemble.

The final ensemble weight is **30% LSTM and 70% XGBoost**, calibrated on the validation set using a grid search over the simplex."

---

## SLIDE 13 — Conformal Prediction & Uncertainty Quantification

*[This slide shows the conformal prediction interval diagram and the QCCP novelty]*

**Script (≈2 min):**

"One of the things that makes our system different from every existing sepsis alert system is that we don't just give a risk score — we give a **statistically guaranteed uncertainty interval** around that score.

This is called **conformal prediction**, and it's the first of our three core novelties. Here's the key idea: a standard model tells you 'this patient has a 78% risk of sepsis.' But it can't tell you how confident it is in that 78%. Is it 78% ± 5%? Or 78% ± 40%? These are completely different clinical situations.

Conformal prediction gives you a mathematically guaranteed coverage bound. Our setup gives **90% coverage** — meaning that for 90% of patients, the true risk falls within the predicted interval. No assumptions about the data distribution. This is a distribution-free statistical guarantee.

The key technical contribution — our **Novelty 1** — is that we compute the nonconformity score not in classical space, but in **quantum kernel Hilbert space**. The nonconformity measure is: one minus the maximum quantum kernel similarity between the test patient and the learned sepsis centroid states. This produces tighter intervals than classical conformal methods because the quantum kernel finds richer structure in the feature space.

In practice, we calibrate the interval width using a held-out calibration set of 20% of training positives. The calibration quantile q-alpha is **0.2663** — and this is baked into the real-time inference pipeline.

Clinically, the interval width is our confidence proxy. A narrow interval — high confidence — enables fast-tracking. A wide interval — low confidence — triggers conservative escalation. The width itself carries actionable information, not just the point estimate."

---

## SLIDE 14 — Red Team Agent & Safety Architecture

*[This slide shows the 7 tripwires, escalation logic WATCH/AMBER/CRITICAL/FAST-TRACK]*

**Script (≈2 min):**

"Now I want to talk about something that I think is genuinely important from a clinical safety perspective — our **Red Team Agent**.

The core problem with pure ML systems in healthcare is this: the model can be wrong. And when an ML model misses a sepsis case — a false negative — the consequence is that a patient deteriorates and nobody gets alerted. In a commercial sepsis alert like Epic's Sepsis Model, if the model gives a low risk score, the system stays silent. Even if the patient's blood pressure is crashing.

Our approach is fundamentally different. The Red Team Agent is a **completely independent layer** that operates in parallel with the ML pipeline. It has seven deterministic clinical tripwires — and if any of these fire, it can escalate the alert regardless of what the ML model says. These tripwires are: temperature below 36°C or above 38.3°C, heart rate above 90 bpm with an increasing trend, respiratory rate above 20 breaths per minute, MAP below 70 mmHg, SpO2 below 92%, lactate above 2 mmol per litre, and GCS below 12.

The escalation logic is: **one tripwire fires → AMBER alert. Two or more tripwires fire → CRITICAL, non-overridable.** Any single extreme value — MAP below 55, SpO2 below 88, lactate above 4 — also triggers CRITICAL immediately.

This is **Novelty 2** — and the key distinction from every other system is that these tripwires are architecturally non-overridable. The ML model cannot suppress them. Even if the ensemble gives a risk score of 0.05, two simultaneous tripwire activations will generate a CRITICAL alert. This bounds our false negative rate at the architectural level, not the statistical level.

The second part of this novelty is the feedback loop: when the Red Team catches a case the ML missed, it doubles the focal loss penalty for that physiological profile in subsequent training. The model progressively hardens against its own failure modes."

---

## SLIDE 15 — Confidence-Gated Orchestrator & Fast-Tracking

*[This slide shows the Orchestrator decision tree and the 4-6 hour delay elimination diagram]*

**Script (≈2 min):**

"The final layer is what we call the **Confidence-Gated Orchestrator** — and this is where the clinical workflow impact becomes tangible.

Standard ICU workflows for sepsis are **strictly sequential**. A patient comes in, the nurse orders a CBC, waits 45 minutes for results, if abnormal orders a lactate, waits again, if elevated orders blood cultures, waits again, eventually calls the physician, who reviews everything and initiates antibiotics. This sequential chain takes **4 to 6 hours**. And we know from Kumar et al. 2006 that every hour of antibiotic delay increases mortality by 7.6%.

Our orchestrator breaks this sequential chain. The logic is: if the **ensemble risk score is above 0.6 AND the conformal confidence is above 80%** — meaning we have a high-risk prediction with a narrow uncertainty interval — we skip the preliminary diagnostic steps. We don't wait for the CBC. We immediately trigger procalcitonin, blood cultures, and begin the sepsis bundle in parallel.

This is **Novelty 3** — confidence-gated diagnostic fast-tracking. No existing system uses model confidence to dynamically route the clinical workflow. The confidence interval width isn't just a number we report — it's an operational input that changes what actions the system recommends.

The output of the orchestrator is one of four alert levels: **WATCH, AMBER, CRITICAL, or FAST-TRACK**. Each level comes with a specific action list: WATCH means reassess in 15 minutes; AMBER means order CBC, CMP, lactate; CRITICAL means initiate the full sepsis bundle within one hour; FAST-TRACK means skip preliminary steps and act immediately.

The system also estimates **how many hours before onset** the alert was generated — our target lead time is 3 to 4 hours. This lead time is what separates proactive detection from reactive confirmation."

---

## SLIDE 16 — Clinical Validation Quote (Tata Main Hospital)

*[This slide shows the Chief Consultant quote from Critical Care, Tata Main Hospital]*

**Script (≈2 min):**

"This slide shows something I'm particularly proud of — external clinical validation.

We shared our system design and preliminary results with the **Chief Consultant and Head of the Critical Care Unit at Tata Main Hospital** — a practicing intensivist with extensive ICU experience. This is not a theoretical endorsement — this is a domain expert who manages sepsis patients every day, reviewing our architecture and giving their professional assessment.

The key statement from them was: *'What impressed me about QuantumHealth Shield AI is that it goes beyond prediction and provides confidence-based clinical decision support that aligns with how intensivists actually make decisions in the ICU.'*

This is significant because the biggest criticism of clinical AI systems is that they don't match clinical intuition — they produce a number without context. The comment about confidence-based decision support aligning with how *intensivists actually think* validates our Novelty 3 directly.

They also called out the integration of early warning, uncertainty quantification, and safety tripwires as — quote — 'both innovative and clinically meaningful.' That's a practicing clinician validating all three of our core novelties simultaneously.

The note about 'reducing diagnostic delays and supporting faster interventions' directly references the 4 to 6 hour delay problem we're solving. A system that alerts several hours early *while telling the clinician how reliable the prediction is* — that's the confidence interval doing clinical work.

We're using this feedback to inform our next phase of evaluation, which includes integration with the hardware vital sign sensor developed by CHTR-RVCE for bedside validation."

---

## SLIDE 17 — Clinical Case Study (62-Year-Old ICU Patient)

*[This slide shows the before/after comparison: old sequential workflow vs QuantumSepsis alert]*

**Script (≈2 min):**

"Let me make this concrete with a clinical case study — because numbers alone don't tell the full story.

The patient is a 62-year-old male admitted to the ICU with pneumonia. His presenting vitals: fever of 38.7 degrees, heart rate 112, MAP 68 mmHg, respiratory rate 26, SpO2 91%, and mild confusion.

In the **traditional workflow** shown on the left — the nurse observes these vitals at the 15-minute monitoring cycle. She suspects something is off, so she orders sequential labs. A CBC goes out. A lactate is ordered. Blood cultures are drawn. The SOFA score is computed manually by the physician. All of this takes 4 to 6 hours before a formal sepsis diagnosis is made and treatment begins. Every one of those hours costs 7.6% additional mortality risk.

Now look at the right side — **with QuantumSepsis Shield**.

The system's 6-hour sliding window has been tracking this patient since admission. The LSTM identifies a rising trend in lactate and declining MAP over the past 4 hours. The Red Team Agent fires three tripwires simultaneously: TW-TEMP for fever, TW-MAP for MAP below 70, and TW-LACTATE for lactate above 2. The conformal prediction interval is narrow — high confidence. The ensemble risk score comes out at **87%**.

The orchestrator immediately outputs: **Alert CRITICAL — FAST-TRACK**. The action list auto-generates: antibiotics within one hour, lactate and blood cultures ordered simultaneously, 30 mL/kg fluid resuscitation initiated, ICU attending auto-notified.

All of this happens in real time — **4 to 6 hours before the traditional workflow would have reached the same conclusion**. That lead time is the difference between a patient who survives and one who progresses to septic shock."

---

## SLIDE 18 — Model Comparison Table

*[This slide shows Window AUROC and Stay AUROC comparison table]*

**Script (≈2 min):**

"Now let's look at the quantitative results against our baselines.

We evaluated on the temporal test set — anchor year group 2020 to 2022 — which is completely held out and represents data the model has never seen during any phase of development.

Starting with the **SOFA Score** — this is the current clinical gold standard, what every ICU uses today. Window AUROC of **0.5869** — essentially a coin flip at the window level. Stay AUROC of **0.68**. This is the performance of the tool currently protecting patients in ICUs worldwide.

Our **LSTM V1 Improved** — trained on 39 curated features — achieves **0.7905 window AUROC and 0.8618 stay AUROC**. The stay AUROC being notably higher than the window AUROC shows that aggregating predictions across a full ICU stay surfaces the model's predictive signal much more clearly than any single 6-hour window.

**XGBoost on 132 features** achieves **0.8038 window AUROC** — the strongest classical baseline. This is the model that most research papers would stop at. We didn't.

Our **Ensemble V1 Final** — combining LSTM and XGBoost with 30/70 weighting — achieves **0.8051 window AUROC**, our current production model. The ensemble improves over either component alone by capturing complementary signals: the LSTM captures temporal trajectories, XGBoost captures statistical extremes.

The **Quantum Kernel at 8 qubits** achieves **0.7598** — competitive given it operates on an 8-dimensional PCA projection of the LSTM embedding, which is a significant compression. More importantly, its contribution to the ensemble's uncertainty quantification through the QCCP conformal scores adds value beyond the raw AUROC number.

Our target with the full quantum-integrated pipeline is to exceed **0.85 AUROC**, and we're currently finalizing those results."

---

## SLIDE 19 — Live Demo / System Dashboard

*[This slide likely shows the Streamlit dashboard or the Vercel React UI]*

**Script (≈2 min):**

"What you're seeing here is our live system dashboard — this isn't a mockup, this is the actual deployed interface.

We have two deployment environments running simultaneously. The **AWS EC2 backend** runs the full ensemble pipeline — LSTM, XGBoost, conformal prediction, Red Team Agent, and Orchestrator — all in production on a t3 instance. The backend exposes a FastAPI endpoint at port 8000 that accepts a JSON payload of vital signs and returns the full prediction response in real time.

The **Vercel frontend** — built in React with TypeScript — connects to the AWS backend through a proxy route. Clinicians or researchers can select a patient scenario from presets — Stable, Early Warning, Developing Sepsis, Septic Shock, Hypothermic Sepsis — or use the Custom Input mode to enter any combination of vitals manually.

What the dashboard shows in real time: the **Risk Gauge** — a 0 to 100% sepsis probability with color coding. Green for WATCH, amber for AMBER, red for CRITICAL. The **conformal interval** displayed as a bracket — [lower bound, upper bound] — with width indicating confidence. The **Red Team tripwire panel** showing all 7 tripwires with their current values in real time. And the **Recommended Actions** — the orchestrator's clinical action list, populated automatically based on the alert level.

The Risk Timeline on the right builds as you change inputs — you can simulate a patient deteriorating over time by adjusting the sliders and watching the risk score climb and the alert level escalate from WATCH to AMBER to CRITICAL.

This is a working clinical simulation tool — and we're currently extending it with a hardware integration layer using an ESP32 and MAX30102 sensor array for real-time bedside vital capture."

---

## SLIDE 20 — Impact & Clinical Value

*[This slide shows the 3-column impact: global scale, system advantages, outcomes]*

**Script (≈2 min):**

"Let me contextualize the impact of what we've built.

Sepsis kills approximately **11 million people globally every year** — that's one death every 2.8 seconds. It's the leading cause of ICU mortality worldwide. And critically, the majority of these deaths are **preventable** — they happen not because medicine doesn't know how to treat sepsis, but because treatment is initiated too late.

The 7% mortality increase per hour of delay — that's not our number, that's from Kumar et al. 2006 in Critical Care Medicine. It's one of the most-cited statistics in emergency medicine. Our 3 to 4 hour lead time advantage, if it translates to real clinical practice, would correspond to a **21 to 28% mortality reduction** for those patients.

From a system design perspective: we run **automated monitoring every 15 minutes** — matching the natural ICU vital sign charting cycle. The confidence-based alert prioritization means clinicians don't get flooded with every marginal elevation — they get actionable, high-confidence alerts. This directly addresses **alert fatigue**, which is one of the most documented reasons existing sepsis alert systems are ignored.

In terms of scalability: the system integrates with any EHR that surfaces chartevents and labevents in a compatible format. MIMIC-IV's structure mirrors what MetaVision and Epic produce. The same pipeline is **extensible to stroke, acute cardiac events, and septic shock in cardiac ICU** — anywhere temporal physiological monitoring with SOFA-like criteria applies.

The downstream outcomes: lower ICU length of stay, fewer complications from delayed treatment, reduced readmission rates, and reduced resource utilization from earlier targeted intervention versus broad-spectrum emergency response."

---

## SLIDE 21 — Future Work & Extensions

*[This slide shows FUO/SIRS extension, hardware integration, agentic architecture]*

**Script (≈2 min):**

"Looking ahead, we have three concrete directions for extending this work.

The first is **FUO and SIRS extension**. FUO — Fever or Pyrexia of Unknown Origin — and SIRS — Systemic Inflammatory Response Syndrome — share almost identical physiological signals with early sepsis. They use the same features: heart rate, blood pressure, temperature, respiratory rate, SpO2, lactate. The core difference is the clinical label — in SIRS, there may be no confirmed infection; in FUO, the fever source is unidentified.

This makes both conditions ideal candidates for our uncertainty-aware framework. In fact, they're harder diagnostic problems than sepsis precisely because the ground truth is less certain — which is where conformal prediction and wide uncertainty intervals are most valuable. We already have the MIMIC-IV cohort extraction code to pull FUO and SIRS stays. The same pipeline, minimal modification.

The second direction is **hardware integration**. We're collaborating with CHTR — the Centre for Healthcare Technology Research at RVCE — which has built a thumb-clip sensor array that captures heart rate, blood pressure, SpO2, respiratory rate, and MAP in real time using EEG-grade sensors. We plan to connect this hardware directly to our AWS backend — the sensor streams vitals via WiFi, the FastAPI endpoint processes them, and the dashboard displays live predictions at the bedside.

The third is our **Agentic AI Architecture** extension. Currently we have four specialized agents: the monitoring agent, the prediction ensemble, the Red Team safety agent, and the orchestration agent. Future work includes adding an **Outcome Learning Agent** that adjusts alert thresholds per patient based on historical near-miss events — a closed-loop adaptive system. We're also exploring federated learning across multiple ICU units to train a shared model without sharing patient data — the first combination of federated learning with quantum kernels for clinical prediction."

---

## SLIDE 22 — Team Contributions

*[This slide shows team member contributions]*

**Script (≈1.5 min — shorter, transition slide):**

"I want to briefly acknowledge the team before we move to the conclusion.

I led the overall project architecture, built the BiLSTM, XGBoost, and ensemble models, implemented the uncertainty quantification pipeline, managed experimentation, and coordinated everything you've seen today.

Our quantum specialist implemented the Qiskit-based quantum kernel using the ZZFeatureMap, designed the fidelity kernel workflow, and developed the Quantum-Calibrated Conformal Prediction methodology — Novelty 1. They also ran all quantum benchmarking and QSVM experiments.

Our clinical and safety engineer conducted the literature survey that informed our research gap analysis, developed the seven clinical safety tripwires, designed the WATCH, AMBER, CRITICAL escalation workflows, and supported dataset preparation.

We also had a team member who assisted with testing, documentation, and slide preparation throughout.

The combination of deep learning expertise, quantum computing knowledge, and clinical safety design is what allowed us to build a system that addresses all three dimensions — predictive accuracy, uncertainty quantification, and patient safety — simultaneously."

---

> **Closing transition line (into Q&A or final slide):**
> 
> *"So in summary — QuantumSepsis Shield is not just another sepsis prediction model. It's a clinically safe, uncertainty-aware, agentic AI system that provides 3 to 4 hours of early warning, with architectural guarantees against dangerous false negatives, and a real-time confidence signal that changes clinical workflow. We're happy to take any questions."*
