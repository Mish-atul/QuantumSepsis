-- ============================================================
-- INITIAL SCHEMA FOR QUANTUMSEPSIS SHIELD
-- ============================================================

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

-- ============================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_profiles_user_id ON public.profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_profiles_hospital_id ON public.profiles(hospital_id);
CREATE INDEX IF NOT EXISTS idx_patients_hospital_id ON public.patients(hospital_id);
CREATE INDEX IF NOT EXISTS idx_patients_status ON public.patients(status);
CREATE INDEX IF NOT EXISTS idx_vitals_patient_id ON public.vitals(patient_id);
CREATE INDEX IF NOT EXISTS idx_vitals_timestamp ON public.vitals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_labs_patient_id ON public.labs(patient_id);
CREATE INDEX IF NOT EXISTS idx_labs_timestamp ON public.labs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_patient_id ON public.risk_assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_timestamp ON public.risk_assessments(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tripwire_alerts_patient_id ON public.tripwire_alerts(patient_id);
CREATE INDEX IF NOT EXISTS idx_tripwire_alerts_is_active ON public.tripwire_alerts(is_active);
