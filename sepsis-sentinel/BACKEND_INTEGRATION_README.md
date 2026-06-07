# QuantumSepsis Shield — Backend Integration Guide

**Version:** 1.0  
**Last Updated:** May 6, 2026  
**Purpose:** Complete API specification and data contracts for frontend integration.

> This document is the **source of truth** for how the React frontend in this repo talks to the QuantumSepsis Shield backend. The frontend ships with a typed adapter at `src/lib/api/quantumSepsis.ts` and a realtime hook at `src/hooks/useRealtimeAlerts.ts` that implement this contract. Lovable Cloud (Supabase) is currently used for auth, demo data, and HITL persistence; the QuantumSepsis REST/WebSocket API listed here is the target production backend and is consumed via the adapter (toggle with `VITE_USE_QS_API=true`).

---

## Table of Contents
1. System Architecture Overview
2. API Endpoints
3. Data Models & Schemas
4. WebSocket Real-Time Updates
5. Authentication & Authorization
6. Error Handling
7. Frontend Integration Examples
8. Performance Considerations
9. Testing & Development
10. Security Considerations
11. Deployment Architecture
12. Monitoring & Observability
13. Migration Guide
14. FAQ
15. Frontend Component Map
16. Appendix A — OpenAPI Spec

---

## 1. System Architecture Overview

```
Frontend (React + Vite + TS)
    ↓ HTTPS REST API (axios via src/lib/api/quantumSepsis.ts)
Backend API Server (FastAPI / Flask)
    ↓
┌─────────────────────────────────────────┐
│  QuantumSepsis Shield Core Pipeline     │
│  1. LSTM Model (risk_score)             │
│  2. Quantum Kernel (optional)           │
│  3. Conformal Predictor (intervals)     │
│  4. Red Team Agent (tripwires)          │
│  5. Orchestrator (final decision)       │
└─────────────────────────────────────────┘
    ↓
PostgreSQL + Redis
    ↓ WebSocket (src/hooks/useRealtimeAlerts.ts)
Frontend (live alerts, vitals, system status)
```

### Recommended Stack
- **Backend API:** FastAPI (async, OpenAPI), or Flask
- **Database:** PostgreSQL
- **Cache / Pub-Sub:** Redis
- **Message Queue:** Redis Pub/Sub or RabbitMQ
- **Deployment:** Docker + Kubernetes, NGINX reverse proxy with SSL

### Architectural Principles (enforced in the frontend)
- **Bidirectional Communication** — persistent WebSocket connection (`useRealtimeAlerts`, `useRealtimeConnection`).
- **Shared State** — `HospitalContext` provides hospital scope, connection status, and the live pipeline event stream to every page.
- **Observability** — `ConnectionStatusIndicator` and `PipelineActivityFeed` surface intermediate agent steps (predictions, tripwires, overrides).
- **Human-in-the-Loop (HITL)** — `HITLActionPanel`, `LogVitalsDrawer`, `LogLabsDrawer`, and acknowledge/resolve flows require explicit clinician approval.
- **Generative UI** — `AgentDataRenderer` renders structured `AgentUIBlock` payloads (charts, tables, forms, alerts) returned by agents.

---

## 2. API Endpoints

**Base URL**
- Production: `https://api.quantumsepsis.hospital.org/v1`
- Development: `http://localhost:8000/v1`
- Frontend reads: `import.meta.env.VITE_QS_API_BASE_URL` (defaults to `http://localhost:8000/v1`)

### 2.1 Patient Monitoring

#### `GET /patients`
List active ICU patients.

**Query params:** `icu_unit`, `alert_level`, `page` (default `1`), `limit` (default `50`).

**Response:**
```json
{
  "patients": [
    {
      "patient_id": "P123456",
      "stay_id": "ICU_2026_001234",
      "icu_unit": "MICU",
      "bed_number": "12A",
      "admission_time": "2026-05-05T14:30:00Z",
      "current_alert": {
        "level": "AMBER",
        "risk_score": 0.42,
        "confidence": 0.73,
        "last_updated": "2026-05-06T08:15:00Z"
      },
      "demographics": { "age": 67, "gender": "M" }
    }
  ],
  "pagination": { "total": 127, "page": 1, "limit": 50, "total_pages": 3 }
}
```

#### `GET /patients/{patient_id}`
Patient detail including `current_vitals`, `current_assessment` (alert level, risk, conformal interval, red-team status, recommended actions, reasoning), and `alert_history`.

#### `GET /patients/{patient_id}/vitals/history?hours=24&resolution=1`
Time-series vitals + labs for chart rendering.

#### `GET /patients/{patient_id}/risk/timeline?hours=24`
Risk score evolution with confidence intervals and trend analysis (`direction`, `slope`, `acceleration`).

### 2.2 Real-Time Prediction

#### `POST /predict`
Run prediction on a vitals window (testing/simulation).

**Request:**
```json
{
  "patient_id": "P123456",
  "vitals_window": [
    { "timestamp": "2026-05-06T08:00:00Z", "heart_rate": 94, "sbp": 118, "dbp": 72,
      "map": 87, "temperature": 38.1, "resp_rate": 22, "spo2": 96, "gcs_total": 15,
      "lactate": 2.1, "wbc": 14.2, "creatinine": 1.3, "platelets": 185 }
  ]
}
```

Returns a `PredictionResult` (see §3) including `red_team_assessment`, `recommended_actions`, `model_version`, and `processing_time_ms`.

### 2.3 Alert Management

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/alerts` | List alerts. Filters: `icu_unit`, `level`, `status` (`active`/`acknowledged`/`resolved`). |
| `POST` | `/alerts/{alert_id}/acknowledge` | HITL: clinician acknowledges alert. Body: `{ clinician_id, notes }`. |
| `POST` | `/alerts/{alert_id}/resolve` | HITL: clinician closes alert. Body: `{ clinician_id, outcome, notes, actions_taken[] }`. `outcome` ∈ `true_positive` / `false_positive` / `false_negative` / `near_miss`. |

### 2.4 Dashboard Statistics

- `GET /dashboard/summary` — patient counts, alert distribution, system component health, performance metrics.
- `GET /dashboard/metrics` — model perf (AUROC, AUPRC, sensitivity), conformal coverage, red-team stats, outcome learning.

### 2.5 Configuration & Admin

- `GET /config/thresholds` — global + per-ICU thresholds.
- `PUT /config/thresholds/{icu_unit}` — admin only. Body: `{ watch_threshold, amber_threshold, updated_by, reason }`.

### 2.6 Health & Metrics

- `GET /health` — component health (`database`, `redis`, `lstm_model`, `quantum_kernel`).
- `GET /metrics` — Prometheus exposition.

---

## 3. Data Models & Schemas

All types are mirrored in `src/lib/api/quantumSepsis.ts` and re-exported from `src/types/database.ts` where they overlap with the existing Lovable Cloud schema.

```ts
export type AlertLevel = "WATCH" | "AMBER" | "CRITICAL" | "FAST-TRACK";
export type TripwireCode = "TW-TEMP" | "TW-HR" | "TW-RR" | "TW-MAP" | "TW-MENTAL";
export type AlertStatus = "active" | "acknowledged" | "resolved";
export type Outcome = "true_positive" | "false_positive" | "false_negative" | "near_miss";

export interface Patient {
  patient_id: string;
  stay_id: string;
  icu_unit: "MICU" | "SICU" | "CVICU" | "NICU" | "OTHER";
  bed_number: string;
  admission_time: string;
  demographics: { age: number; gender: "M" | "F" | "O"; weight_kg?: number };
  current_alert?: AlertSummary;
}

export interface VitalSigns {
  timestamp: string;
  heart_rate: number; sbp: number; dbp: number; map: number;
  temperature: number; resp_rate: number; spo2: number; gcs_total: number;
  lactate?: number; wbc?: number; creatinine?: number; platelets?: number;
}

export interface RedTeamAssessment {
  triggered: boolean;
  active_tripwires: TripwireCode[];
  override_level: AlertLevel | null;
  n_active: number;
  details?: Record<string, string>;
}

export interface PredictionResult {
  alert_level: AlertLevel;
  risk_score: number;        // 0–1
  confidence: number;        // 0–1
  conformal_interval: [number, number];
  fast_tracked: boolean;
  red_team_assessment: RedTeamAssessment;
  recommended_actions: string[];
  reasoning: string;
  timestamp: string;
  model_version: string;
  processing_time_ms: number;
}

export interface Alert {
  alert_id: string;
  patient_id: string;
  stay_id: string;
  level: AlertLevel;
  risk_score: number;
  confidence: number;
  triggered_at: string;
  status: AlertStatus;
  acknowledged_by?: string;
  acknowledged_at?: string;
  resolved_by?: string;
  resolved_at?: string;
  outcome?: Outcome;
  red_team_override: boolean;
  active_tripwires: TripwireCode[];
  bed_number: string;
  icu_unit: string;
}
```

---

## 4. WebSocket Real-Time Updates

**Endpoint:** `wss://api.quantumsepsis.hospital.org/v1/ws`  
Frontend hook: `src/hooks/useRealtimeAlerts.ts`.

### Handshake
```ts
ws.send(JSON.stringify({ type: "auth", token: jwt }));
ws.send(JSON.stringify({ type: "subscribe", channel: "icu_unit", icu_unit: "MICU" }));
```

### Message Types
- `new_alert` → `{ alert_id, patient_id, level, risk_score, bed_number, icu_unit }`
- `alert_level_change` → `{ alert_id, patient_id, old_level, new_level, risk_score }`
- `vitals_update` → `{ patient_id, vitals: Partial<VitalSigns> }`
- `system_status` → `{ status, active_patients, active_alerts }`

Every inbound message is forwarded to `HospitalContext.addEvent()` so the **PipelineActivityFeed** observability panel shows a live stream of agent activity, and to React Query for cache invalidation.

---

## 5. Authentication & Authorization

JWT in `Authorization: Bearer <token>`. Tokens are issued by Lovable Cloud auth and exchanged for a backend JWT (or used directly when the backend trusts the same JWKS).

```json
{
  "sub": "user_id_12345",
  "role": "clinician",
  "icu_units": ["MICU", "SICU"],
  "permissions": ["view_patients", "acknowledge_alerts", "resolve_alerts"],
  "exp": 1746518400, "iat": 1746432000
}
```

| Role | Permissions |
|---|---|
| `viewer` (nurse view-only) | View patients, view alerts |
| `clinician` (nurse / attending) | + log vitals, acknowledge alerts, resolve alerts |
| `admin` | + update thresholds, view metrics, manage staff |
| `researcher` | View anonymized data, export metrics |

Frontend role mapping lives in `src/hooks/useAuth.tsx` and is enforced in `HITLActionPanel` and `/admin`.

**Required headers**
```
Authorization: Bearer <jwt>
Content-Type: application/json
X-Request-ID: <uuid>   // optional, auto-added by adapter
```

---

## 6. Error Handling

```json
{
  "error": {
    "code": "PATIENT_NOT_FOUND",
    "message": "Patient with ID P123456 not found",
    "details": { "patient_id": "P123456" },
    "timestamp": "2026-05-06T08:15:00Z",
    "request_id": "req_abc123"
  }
}
```

| HTTP | Meaning | Frontend behaviour |
|---|---|---|
| 200 | OK | render |
| 201 | Created | toast success |
| 400 | Bad Request | toast destructive |
| 401 | Unauthorized | force re-login |
| 403 | Forbidden | hide action / toast |
| 404 | Not Found | empty state |
| 422 | Invalid vitals | inline form error |
| 429 | Rate limited | retry with backoff |
| 500 / 503 | Server error | toast + retry |

Error codes: `PATIENT_NOT_FOUND`, `INVALID_VITALS`, `MODEL_INFERENCE_ERROR`, `INSUFFICIENT_DATA`, `RATE_LIMIT_EXCEEDED`, `UNAUTHORIZED`, `FORBIDDEN`.

The adapter normalises every error to `QSApiError { code, message, status, details }` so callers can `try/catch` uniformly.

---

## 7. Frontend Integration Examples

### 7.1 Using the typed adapter
```ts
import { qsApi } from "@/lib/api/quantumSepsis";

const { patients, pagination } = await qsApi.listPatients({ icu_unit: "MICU" });
const detail = await qsApi.getPatient("P123456");
const prediction = await qsApi.predict({ patient_id: "P123456", vitals_window: [...] });
await qsApi.acknowledgeAlert(alertId, { clinician_id: profile.id, notes: "Reviewed" });
```

### 7.2 Realtime alerts hook
```tsx
import { useRealtimeAlerts } from "@/hooks/useRealtimeAlerts";

const { alerts, status, lastMessage } = useRealtimeAlerts({ icu_unit: "MICU" });
```

The hook plays a sound for `CRITICAL` `new_alert`, mutates the React-Query alert cache for `alert_level_change`, and updates `HospitalContext` so the **PipelineActivityFeed** reflects every event.

### 7.3 HITL acknowledge
```ts
await qsApi.acknowledgeAlert(alertId, {
  clinician_id: profile.id,
  notes: "Vitals reviewed, blood cultures ordered",
});
```

### 7.4 Risk timeline chart
Use `qsApi.getRiskTimeline(patientId, { hours: 24 })` and feed `timeline[].risk_score` plus `conformal_interval` upper/lower bounds into Recharts (the existing `VitalsChart` component is already structured to accept these series).

---

## 8. Performance Considerations

- **Caching (Redis):** `patient:{id}:current` (30s), `vitals:24h` (5m), `risk:timeline` (1m), `dashboard:summary` (10s), `alerts:active:{unit}` (15s).
- **Rate limits:** `/patients` 60/min, `/patients/{id}` 120/min, `/predict` 10/min, WebSocket ≤5 concurrent/user. Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.
- **Pagination:** cursor-based preferred (`?cursor=…&limit=50`).
- **Batch:** `POST /patients/batch` with `{ patient_ids: [...] }`.
- **Compression:** `Accept-Encoding: gzip, deflate` (axios sets automatically).

Frontend uses React Query with `staleTime` aligned to these TTLs to avoid hammering the API.

---

## 9. Testing & Development

### Mock server (Express)
```js
const express = require("express");
const app = express();
app.get("/v1/patients/:id", (req, res) => res.json({
  patient_id: req.params.id,
  current_assessment: { alert_level: "AMBER", risk_score: 0.42, confidence: 0.73,
                        conformal_interval: [0.28, 0.56] },
  current_vitals: { heart_rate: 94, temperature: 38.1 },
}));
app.listen(8000);
```

Point the frontend at it with `VITE_QS_API_BASE_URL=http://localhost:8000/v1`.

### Postman
Import the spec from §16; set `{{base_url}}` and `{{jwt_token}}` collection variables.

---

## 10. Security Considerations

- **HIPAA headers (server-side):** `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Content-Security-Policy: default-src 'self'`.
- **Audit log:** every API call records user, action, patient_id, timestamp, IP, request_id.
- **Anonymisation:** `/research/*` returns hashed patient IDs and date-shifted timestamps.
- **Vitals validation ranges:**
  ```ts
  const VITAL_RANGES = {
    heart_rate: [20, 250], sbp: [50, 250], dbp: [20, 150],
    temperature: [32, 42], resp_rate: [4, 60], spo2: [50, 100], gcs_total: [3, 15],
  };
  ```
  Mirrored in `LogVitalsDrawer` zod schema.

---

## 11. Deployment Architecture

```
NGINX (SSL) → API Server pool (FastAPI) → PostgreSQL primary + replica
                                        ↘ Redis (cache + pub/sub)
```

**Production env**
```
DATABASE_URL=postgresql://user:pass@db:5432/quantumsepsis
REDIS_URL=redis://redis:6379/0
JWT_SECRET=...
MODEL_PATH=/models/lstm_v3_phase2_best.pt
QUANTUM_ENABLED=true
CORS_ORIGINS=https://dashboard.quantumsepsis.hospital.org
```

---

## 12. Monitoring & Observability

- `GET /health` returns component status + uptime + RPS + avg response time.
- `GET /metrics` exposes Prometheus counters (`api_requests_total`) and histograms (`prediction_duration_seconds`).
- Frontend surfaces these via `ConnectionStatusIndicator` (live/degraded/down) and the `/admin` system panel.

---

## 13. Migration Guide

1. Export legacy patient data → CSV.
2. Transform column names (`pt_id`→`patient_id`, `hr`→`heart_rate`, …).
3. Bulk insert into PostgreSQL.
4. Run `POST /predict` over all active patients to seed `risk_assessments`.
5. Verify per-ICU thresholds via `GET /config/thresholds`.

---

## 14. FAQ

**Q: How often are predictions updated?** Every new vitals push (1–5 min).  
**Q: What happens if the model fails?** Red Team Agent continues independently; alerts come from tripwires only.  
**Q: Per-patient thresholds?** v2.0 — currently per-ICU.  
**Q: Quantum vs classical kernel?** ~2–3% AUROC improvement, tighter CIs.  
**Q: Latency?** Avg 134 ms, P95 287 ms.  
**Q: Reporting a false negative?** `POST /alerts/{id}/resolve` with `outcome: "false_negative"` — feeds the Outcome Learning Agent.

---

## 15. Frontend Component Map

| Concern | Backend surface | Frontend module |
|---|---|---|
| Patient list | `GET /patients` | `usePatients` → `Index.tsx`, `PatientCard` |
| Patient detail | `GET /patients/{id}` | `PatientDetail.tsx`, `PatientTopBar` |
| Vitals history | `GET /patients/{id}/vitals/history` | `usePatientVitals`, `VitalsChart` |
| Risk timeline | `GET /patients/{id}/risk/timeline` | `useRiskAssessments`, `RiskGauge`, `ConfidenceInterval` |
| Predict (sim) | `POST /predict` | `qsApi.predict()` (admin sandbox) |
| Alerts | `GET /alerts`, ack/resolve | `useTripwireAlerts`, `HITLActionPanel`, `TripwirePanel` |
| Dashboard stats | `GET /dashboard/summary` | `Index.tsx` status bar, `/admin` |
| Thresholds | `GET/PUT /config/thresholds` | `Admin.tsx` |
| Realtime stream | `WSS /ws` | `useRealtimeAlerts`, `useRealtimeConnection`, `HospitalContext`, `PipelineActivityFeed`, `ConnectionStatusIndicator` |
| HITL data entry | (drives `POST /predict` server-side) | `LogVitalsDrawer`, `LogLabsDrawer`, `AdmitPatientModal`, `DischargePatientDialog` |
| Generative UI | Agent `AgentUIBlock` payloads | `AgentDataRenderer` |

---

## 16. Appendix A — OpenAPI Spec

```yaml
openapi: 3.0.0
info:
  title: QuantumSepsis Shield API
  version: 1.0.0
  description: Early sepsis detection system API
servers:
  - url: https://api.quantumsepsis.hospital.org/v1
    description: Production
  - url: http://localhost:8000/v1
    description: Development
security:
  - bearerAuth: []
paths:
  /patients:
    get:
      summary: List all patients
      parameters:
        - { name: icu_unit,    in: query, schema: { type: string, enum: [MICU, SICU, CVICU, NICU] } }
        - { name: alert_level, in: query, schema: { type: string, enum: [WATCH, AMBER, CRITICAL, FAST-TRACK] } }
        - { name: page,        in: query, schema: { type: integer, default: 1 } }
        - { name: limit,       in: query, schema: { type: integer, default: 50 } }
      responses:
        '200': { description: Success, content: { application/json: { schema: { $ref: '#/components/schemas/PatientList' } } } }
  /patients/{patient_id}:
    get:
      summary: Get patient details
      parameters: [ { name: patient_id, in: path, required: true, schema: { type: string } } ]
      responses:
        '200': { description: Success, content: { application/json: { schema: { $ref: '#/components/schemas/PatientDetail' } } } }
        '404': { description: Patient not found }
  /predict:
    post:
      summary: Run prediction
      requestBody:
        required: true
        content: { application/json: { schema: { $ref: '#/components/schemas/PredictionRequest' } } }
      responses:
        '200': { description: Success, content: { application/json: { schema: { $ref: '#/components/schemas/PredictionResult' } } } }
components:
  securitySchemes:
    bearerAuth: { type: http, scheme: bearer, bearerFormat: JWT }
  schemas:
    PatientList:
      type: object
      properties:
        patients: { type: array, items: { $ref: '#/components/schemas/Patient' } }
        pagination: { $ref: '#/components/schemas/Pagination' }
    Patient:
      type: object
      properties:
        patient_id:    { type: string }
        stay_id:       { type: string }
        icu_unit:      { type: string, enum: [MICU, SICU, CVICU, NICU] }
        bed_number:    { type: string }
        admission_time:{ type: string, format: date-time }
    PatientDetail:
      allOf:
        - $ref: '#/components/schemas/Patient'
        - type: object
          properties:
            current_vitals:     { $ref: '#/components/schemas/VitalSigns' }
            current_assessment: { $ref: '#/components/schemas/PredictionResult' }
    VitalSigns:
      type: object
      properties:
        timestamp:   { type: string, format: date-time }
        heart_rate:  { type: number }
        sbp:         { type: number }
        dbp:         { type: number }
        map:         { type: number }
        temperature: { type: number }
        resp_rate:   { type: number }
        spo2:        { type: number }
        gcs_total:   { type: integer }
        lactate:     { type: number }
        wbc:         { type: number }
        creatinine:  { type: number }
        platelets:   { type: number }
    PredictionResult:
      type: object
      properties:
        alert_level:        { type: string, enum: [WATCH, AMBER, CRITICAL, FAST-TRACK] }
        risk_score:         { type: number, minimum: 0, maximum: 1 }
        confidence:         { type: number, minimum: 0, maximum: 1 }
        conformal_interval: { type: array, items: { type: number }, minItems: 2, maxItems: 2 }
        fast_tracked:       { type: boolean }
        recommended_actions:{ type: array, items: { type: string } }
        reasoning:          { type: string }
        timestamp:          { type: string, format: date-time }
    Pagination:
      type: object
      properties:
        total:       { type: integer }
        page:        { type: integer }
        limit:       { type: integer }
        total_pages: { type: integer }
```

---

*End of Backend Integration Guide. Frontend adapter: `src/lib/api/quantumSepsis.ts`. Realtime hook: `src/hooks/useRealtimeAlerts.ts`. Toggle external backend with `VITE_USE_QS_API=true` and `VITE_QS_API_BASE_URL=…`.*
