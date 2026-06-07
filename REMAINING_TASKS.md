# QuantumSepsis Shield — Remaining Tasks for Phase 2 Completion

> **Last updated**: 7 June 2026  
> **Status**: Backend (AWS) ✅ working, Frontend ✅ working locally, Vercel ❌ broken (env vars)  
> **Share this doc** with your teammate along with the `.pem` file

---

## 🏗️ Architecture Overview (read this first)

```
┌─────────────────┐       ┌──────────────────────────┐       ┌──────────────────┐
│   Frontend      │──────▶│   AWS EC2 Backend         │       │   Supabase       │
│  (Vite + React) │ REST  │   54.242.66.27:8000       │       │  (Postgres + Auth│
│                 │       │   api_server.py (FastAPI)  │       │   + Realtime)    │
│  Vercel deploy: │       │   LSTM + XGBoost + CXR    │       │                  │
│  testing-quant  │       │   + Ollama LLM            │       │  Project ID:     │
│  .vercel.app    │       │   + Red Team Tripwires    │       │  hntfeivuhmzdq.. │
└────────┬────────┘       └──────────────────────────┘       └────────┬─────────┘
         │                                                            │
         └────────────────── Supabase JS Client ─────────────────────┘
              Auth, DB reads/writes, Realtime WebSocket
```

### Two Git Repos
| Repo | URL | What |
|------|-----|------|
| **Main** (backend + all code) | https://github.com/Mish-atul/QuantumSepsis | api_server.py, models, agents |
| **Vercel** (frontend only) | https://github.com/Mish-atul/testing-quant | sepsis-sentinel/ (React app) |

### Key Paths (local)
```
QuantumSepsis/
├── api_server.py               ← FastAPI backend (runs on AWS)
├── src/models/cxr_encoder.py   ← CXR analysis (DenseNet121)
├── src/agents/clinical_reasoning.py  ← LLM narrative agent
├── src/agents/red_team.py      ← Tripwire evaluation
├── quantum-key.pem             ← AWS SSH key (NEVER commit)
└── sepsis-sentinel/            ← Frontend React app
    ├── src/pages/              ← All page components
    ├── src/hooks/              ← Auth, data hooks
    ├── src/integrations/supabase/  ← Supabase client + types
    ├── .env                    ← Supabase keys (committed)
    ├── vite.config.ts          ← Dev proxy → AWS
    └── vercel.json             ← Prod proxy → AWS
```

---

## 🔴 PRIORITY 1: Fix Vercel Deployment (BROKEN NOW)

### Problem
The Vercel deployment at `testing-quant.vercel.app` has **wrong environment variables** in the Vercel dashboard. The Supabase API key is set to an old EC2 IP address instead of the real key. This causes:
- `WebSocket connection to 'wss://...supabase.co/...?apikey=34.224.69.251'` errors
- Auth completely broken on production

### Fix
1. Go to **Vercel Dashboard** → `testing-quant` project → **Settings** → **Environment Variables**
2. Delete any existing `VITE_SUPABASE_*` variables
3. Add these three (for ALL environments: Production, Preview, Development):

| Variable | Value |
|----------|-------|
| `VITE_SUPABASE_URL` | `https://hntfeivuhmzdqlhtwxcn.supabase.co` |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhudGZlaXZ1aG16ZHFsaHR3eGNuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI5NzY0OTAsImV4cCI6MjA4ODU1MjQ5MH0.kHpRKIjT7V4XsSW4atMRlh2k6SqB8ElOo5WH8-FNUBo` |
| `VITE_SUPABASE_PROJECT_ID` | `hntfeivuhmzdqlhtwxcn` |

4. **Redeploy** from Vercel dashboard (or push any commit to trigger)
5. Verify: console should have **zero** WebSocket errors

---

## 🟡 PRIORITY 2: Supabase Database Setup

### Current State
The Supabase project (`hntfeivuhmzdqlhtwxcn`) already has these tables defined in the **TypeScript types** (`src/integrations/supabase/types.ts`):

| Table | Purpose | Status |
|-------|---------|--------|
| `hospitals` | Multi-tenant hospital registry | ⚠️ Need to verify tables exist in Supabase |
| `patients` | ICU patient records (name, MRN, bed, status) | ⚠️ Need to verify |
| `vitals` | Vital signs (HR, MAP, Temp, RR, SpO2, etc.) | ⚠️ Need to verify |
| `labs` | Lab results (WBC, Lactate, Creatinine, etc.) | ⚠️ Need to verify |
| `risk_assessments` | AI risk scores + tier + confidence intervals | ⚠️ Need to verify |
| `tripwire_alerts` | Red Team alerts (metric, threshold, active) | ⚠️ Need to verify |
| `profiles` | User profiles (role, hospital_id, department) | ⚠️ Need to verify |

### What To Do
1. **Login to Supabase Dashboard**: https://supabase.com/dashboard → project `hntfeivuhmzdqlhtwxcn`
2. Go to **Table Editor** → check if these 7 tables exist
3. If they DON'T exist, create them. Here's the SQL to run in **SQL Editor**:

```sql
-- ============================================================
-- HOSPITALS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.hospitals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  tier TEXT NOT NULL DEFAULT 'standard',
  total_icu_beds INT NOT NULL DEFAULT 20,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- PROFILES (linked to auth.users)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name TEXT,
  role TEXT NOT NULL DEFAULT 'nurse',
  department TEXT,
  employee_id TEXT,
  hospital_id UUID REFERENCES public.hospitals(id),
  avatar_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(user_id)
);

-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (user_id, full_name)
  VALUES (NEW.id, NEW.raw_user_meta_data->>'full_name');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================================
-- PATIENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.patients (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  mrn TEXT NOT NULL,
  name TEXT NOT NULL,
  bed_number TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  hospital_id UUID REFERENCES public.hospitals(id),
  admission_time TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- VITALS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.vitals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  heart_rate NUMERIC,
  blood_pressure_sys NUMERIC,
  blood_pressure_dia NUMERIC,
  map NUMERIC,
  temperature NUMERIC,
  respiratory_rate NUMERIC,
  spo2 NUMERIC,
  mental_status TEXT DEFAULT 'alert',
  is_manual_entry BOOLEAN DEFAULT false,
  hospital_id UUID REFERENCES public.hospitals(id),
  timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- LABS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.labs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  wbc NUMERIC,
  lactate NUMERIC,
  creatinine NUMERIC,
  platelets NUMERIC,
  bilirubin NUMERIC,
  procalcitonin NUMERIC,
  hospital_id UUID REFERENCES public.hospitals(id),
  timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- RISK ASSESSMENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.risk_assessments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  quantum_risk_score NUMERIC NOT NULL,
  tier TEXT NOT NULL,
  confidence_interval_lower NUMERIC NOT NULL,
  confidence_interval_upper NUMERIC NOT NULL,
  hospital_id UUID REFERENCES public.hospitals(id),
  timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- TRIPWIRE ALERTS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.tripwire_alerts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  metric TEXT NOT NULL,
  value NUMERIC NOT NULL,
  threshold_breached TEXT NOT NULL,
  is_active BOOLEAN DEFAULT true,
  hospital_id UUID REFERENCES public.hospitals(id),
  timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- HELPER FUNCTIONS (used by frontend)
-- ============================================================
CREATE OR REPLACE FUNCTION public.get_user_role(_user_id UUID)
RETURNS TEXT AS $$
  SELECT role FROM public.profiles WHERE user_id = _user_id LIMIT 1;
$$ LANGUAGE sql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION public.get_user_hospital_id(_user_id UUID)
RETURNS UUID AS $$
  SELECT hospital_id FROM public.profiles WHERE user_id = _user_id LIMIT 1;
$$ LANGUAGE sql SECURITY DEFINER;
```

4. **Enable Realtime** for these tables (Supabase Dashboard → Database → Replication):
   - `vitals` ✅
   - `risk_assessments` ✅
   - `tripwire_alerts` ✅
   - `patients` ✅
   - `labs` ✅

---

## 🟡 PRIORITY 3: Row Level Security (RLS) Policies

The app is multi-tenant (hospital-scoped). Each user should only see data from their hospital. Run this in **Supabase SQL Editor**:

```sql
-- Enable RLS on all tables
ALTER TABLE public.hospitals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vitals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.labs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.risk_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tripwire_alerts ENABLE ROW LEVEL SECURITY;

-- PROFILES: users can read/update their own profile
CREATE POLICY "Users can view own profile" ON public.profiles
  FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can update own profile" ON public.profiles
  FOR UPDATE USING (auth.uid() = user_id);

-- PROFILES: admins can view all profiles in their hospital
CREATE POLICY "Admins can view hospital profiles" ON public.profiles
  FOR SELECT USING (
    hospital_id = (SELECT hospital_id FROM public.profiles WHERE user_id = auth.uid())
    AND (SELECT role FROM public.profiles WHERE user_id = auth.uid()) = 'admin'
  );

-- PATIENTS: all staff can view patients in their hospital
CREATE POLICY "Staff can view hospital patients" ON public.patients
  FOR SELECT USING (
    hospital_id = (SELECT hospital_id FROM public.profiles WHERE user_id = auth.uid())
  );

-- PATIENTS: nurses + attendings can insert patients
CREATE POLICY "Staff can admit patients" ON public.patients
  FOR INSERT WITH CHECK (
    hospital_id = (SELECT hospital_id FROM public.profiles WHERE user_id = auth.uid())
  );

-- PATIENTS: staff can update patients in their hospital
CREATE POLICY "Staff can update patients" ON public.patients
  FOR UPDATE USING (
    hospital_id = (SELECT hospital_id FROM public.profiles WHERE user_id = auth.uid())
  );

-- VITALS: staff can read/write vitals for their hospital
CREATE POLICY "Staff can view vitals" ON public.vitals
  FOR SELECT USING (
    hospital_id = (SELECT hospital_id FROM public.profiles WHERE user_id = auth.uid())
  );
CREATE POLICY "Staff can log vitals" ON public.vitals
  FOR INSERT WITH CHECK (
    hospital_id = (SELECT hospital_id FROM public.profiles WHERE user_id = auth.uid())
  );

-- LABS: same as vitals
CREATE POLICY "Staff can view labs" ON public.labs
  FOR SELECT USING (
    hospital_id = (SELECT hospital_id FROM public.profiles WHERE user_id = auth.uid())
  );
CREATE POLICY "Staff can log labs" ON public.labs
  FOR INSERT WITH CHECK (
    hospital_id = (SELECT hospital_id FROM public.profiles WHERE user_id = auth.uid())
  );

-- RISK ASSESSMENTS: read only for all staff
CREATE POLICY "Staff can view risk assessments" ON public.risk_assessments
  FOR SELECT USING (
    hospital_id = (SELECT hospital_id FROM public.profiles WHERE user_id = auth.uid())
  );
CREATE POLICY "System can insert risk assessments" ON public.risk_assessments
  FOR INSERT WITH CHECK (true);  -- API server inserts via service key

-- TRIPWIRE ALERTS: read for all staff, write for system
CREATE POLICY "Staff can view tripwire alerts" ON public.tripwire_alerts
  FOR SELECT USING (
    hospital_id = (SELECT hospital_id FROM public.profiles WHERE user_id = auth.uid())
  );
CREATE POLICY "System can manage tripwire alerts" ON public.tripwire_alerts
  FOR ALL USING (true);  -- API server manages via service key

-- HOSPITALS: anyone can read (needed for registration)
CREATE POLICY "Public can view hospitals" ON public.hospitals
  FOR SELECT USING (true);
CREATE POLICY "Anyone can create hospital" ON public.hospitals
  FOR INSERT WITH CHECK (true);
```

---

## 🟡 PRIORITY 4: Role-Based Access Control (Frontend)

### What's Already Built ✅
| Feature | File | Status |
|---------|------|--------|
| Auth context + hook | `src/hooks/useAuth.tsx` | ✅ Working |
| Login page (email + Google SSO) | `src/pages/Login.tsx` | ✅ Built |
| Sign Up form | `src/components/auth/SignUpForm.tsx` | ✅ Built |
| Register Hospital flow | `src/pages/Register.tsx` | ✅ Built |
| Protected routes | `src/components/auth/ProtectedRoute.tsx` | ✅ Built |
| Admin page (staff management) | `src/pages/Admin.tsx` | ✅ Built |
| Password reset | `src/pages/ResetPassword.tsx` | ✅ Built |

### What's Missing / Needs Work ❌

#### 4a. `ProtectedRoute` needs role-based gating
Currently `ProtectedRoute` only checks `user != null`. It should also check roles:

```tsx
// src/components/auth/ProtectedRoute.tsx — add role prop
export function ProtectedRoute({ 
  children, 
  allowedRoles 
}: { 
  children: ReactNode; 
  allowedRoles?: StaffRole[] 
}) {
  const { user, profile, loading } = useAuth();

  if (loading) return <LoadingSpinner />;
  if (!user) return <Navigate to="/login" replace />;
  
  // Role check
  if (allowedRoles && profile && !allowedRoles.includes(profile.role)) {
    return <Navigate to="/dashboard" replace />;
    // Or show "Unauthorized" page
  }

  return <>{children}</>;
}
```

Then in `App.tsx`:
```tsx
// Admin page — only admins
<Route path="/admin" element={
  <ProtectedRoute allowedRoles={["admin"]}>
    <Admin />
  </ProtectedRoute>
} />
```

#### 4b. Nurse vs Doctor visibility rules
| Feature | Nurse | Attending (Doctor) | Admin |
|---------|-------|-------------------|-------|
| View Ward dashboard | ✅ | ✅ | ✅ |
| View patient vitals | ✅ | ✅ | ✅ |
| Log vitals (manual entry) | ✅ | ✅ | ❌ |
| Log lab results | ✅ | ✅ | ❌ |
| Admit/discharge patients | ❌ | ✅ | ✅ |
| View risk scores | ✅ (limited) | ✅ (full) | ✅ |
| Override AI alerts | ❌ | ✅ | ✅ |
| Run Demo Simulator | ✅ | ✅ | ✅ |
| Manage staff | ❌ | ❌ | ✅ |

**Where to implement**: Hide/show buttons conditionally based on `profile.role` from `useAuth()`.

#### 4c. Demo page access
Currently `/demo` is **public** (no auth required). For the hackathon demo this is fine, but for production you may want to protect it or add a guest mode.

---

## 🟡 PRIORITY 5: Connect Ward Dashboard to Live Database

### What's Already Built ✅
| Hook | File | What it does |
|------|------|-------------|
| `usePatients` | `src/hooks/usePatients.ts` | Fetch patients from Supabase |
| `usePatientVitals` | `src/hooks/usePatientVitals.ts` | Fetch vitals for a patient |
| `usePatientLabs` | `src/hooks/usePatientLabs.ts` | Fetch labs for a patient |
| `useRiskAssessments` | `src/hooks/useRiskAssessments.ts` | Fetch risk scores |
| `useTripwireAlerts` | `src/hooks/useTripwireAlerts.ts` | Fetch tripwire alerts |
| `useRealtimeAlerts` | `src/hooks/useRealtimeAlerts.ts` | Realtime push notifications |
| `useRealtimeConnection` | `src/hooks/useRealtimeConnection.ts` | WebSocket connection status |
| `HospitalContext` | `src/contexts/HospitalContext.tsx` | Hospital-scoped realtime subscriptions |

### What Needs Work
The hooks are all implemented, but they need **the database tables to actually exist** (see Priority 2). Once the tables are created:

1. **Seed demo data**: Create a test hospital + some patients via Supabase SQL:
```sql
-- Create demo hospital
INSERT INTO public.hospitals (name, tier, total_icu_beds)
VALUES ('QuantumHealth Demo Hospital', 'premium', 24)
RETURNING id;  -- Note this ID

-- Create demo patients (use the hospital ID from above)
INSERT INTO public.patients (mrn, name, bed_number, hospital_id, status)
VALUES
  ('MRN-001', 'Rajesh Kumar', 'ICU-01', '<hospital_id>', 'active'),
  ('MRN-002', 'Priya Sharma', 'ICU-02', '<hospital_id>', 'active'),
  ('MRN-003', 'Amit Singh', 'ICU-05', '<hospital_id>', 'active');
```

2. **Register an admin user**: Go to `localhost:8080/register`, create a hospital, then update that user's role to `admin` in Supabase:
```sql
UPDATE public.profiles SET role = 'admin' WHERE user_id = '<your-auth-user-id>';
```

---

## 🟢 PRIORITY 6: Polish & Nice-to-Haves

### 6a. Remove Lovable dependency
The file `src/integrations/lovable/index.ts` imports `@lovable.dev/cloud-auth-js` for Google SSO. If this causes build errors on Vercel, either:
- Remove the Google SSO button from Login.tsx, or  
- Replace with native Supabase Google OAuth:
```tsx
const handleGoogleLogin = async () => {
  await supabase.auth.signInWithOAuth({
    provider: 'google',
    options: { redirectTo: window.location.origin }
  });
};
```

### 6b. Wire prediction results to Supabase
After `/predict` returns, the frontend should save the result to Supabase so the Ward dashboard can show historical risk scores:
```ts
// In DemoSimulator or a new hook
const savePrediction = async (patientId: string, result: PredictionResult) => {
  await supabase.from("risk_assessments").insert({
    patient_id: patientId,
    quantum_risk_score: result.risk_score,
    tier: result.alert_level,
    confidence_interval_lower: result.conformal_interval[0],
    confidence_interval_upper: result.conformal_interval[1],
    hospital_id: profile?.hospital_id,
  });
};
```

### 6c. Error badge in nav
The red "Error" badge in the top-right navigation comes from the Supabase Realtime connection failing. Once you fix the env vars (Priority 1), it should show green. If it persists, check `src/components/layout/GlobalNav.tsx` or `useRealtimeConnection.ts`.

---

## 🔧 AWS Access (for your teammate)

### SSH into EC2
```bash
ssh -i quantum-key.pem ubuntu@54.242.66.27
```

### API Server Location
```bash
cd ~/QuantumSepsis_Complete_Backup
source venv/bin/activate
```

### Restart API Server
```bash
tmux kill-session -t api
tmux new -d -s api 'cd ~/QuantumSepsis_Complete_Backup && source venv/bin/activate && python api_server.py 2>&1 | tee /tmp/api.log'
```

### Check Logs
```bash
tail -f /tmp/api.log
```

### Test Prediction
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

### What's Running on AWS
| Service | Port | Status |
|---------|------|--------|
| FastAPI (api_server.py) | 8000 | ✅ Running in tmux |
| Ollama (Llama 3.2 3B) | 11434 | ✅ Installed, CPU mode |
| Models loaded | — | LSTM + XGBoost + Conformal + RedTeam |

---

## 🏃 Local Development Setup

### Frontend
```bash
cd sepsis-sentinel
npm install
npm run dev
# → http://localhost:8080
# → Proxy /api/* → http://54.242.66.27:8000
```

### Test the /predict endpoint locally
Open http://localhost:8080/demo → select "Septic Shock" scenario → you should see:
- Backend badge: `ensemble`
- Risk score and tripwires from AWS
- Clinical AI Assessment card (violet)
- Demographics note at bottom

---

## 📋 Quick Checklist for Your Teammate

```
[ ] Fix Vercel env vars (Priority 1) — 5 min
[ ] Verify/create Supabase tables (Priority 2) — 15 min
[ ] Add RLS policies (Priority 3) — 10 min
[ ] Add role prop to ProtectedRoute (Priority 4a) — 10 min
[ ] Add nurse/doctor visibility rules (Priority 4b) — 30 min
[ ] Seed demo data in Supabase (Priority 5) — 10 min
[ ] Test login → dashboard → patient flow — 15 min
[ ] Test Demo Simulator on Vercel — 5 min
[ ] Remove/fix Lovable dependency if needed (Priority 6a) — 10 min
[ ] Wire predictions to Supabase (Priority 6b) — 20 min
```

**Estimated total: ~2-3 hours**

---

## ⚠️ Important Notes

1. **Never commit `quantum-key.pem`** — it's in `.gitignore`
2. **EC2 costs money** — stop the instance when not testing: AWS Console → EC2 → Stop Instance
3. **Supabase free tier** is sufficient for the hackathon (500MB DB, 50K auth requests)
4. **The `.env` file IS committed** to the Vercel repo — the Supabase keys are **publishable** (anon key), not secret
5. **Pushing to `testing-quant` repo** auto-triggers Vercel deployment
6. **Pushing to `QuantumSepsis` repo** does NOT trigger deployment — you need to SCP files to AWS manually
