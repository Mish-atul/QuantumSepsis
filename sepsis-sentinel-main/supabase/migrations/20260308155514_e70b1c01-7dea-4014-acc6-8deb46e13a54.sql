
-- Create patients table
CREATE TABLE public.patients (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  name TEXT NOT NULL,
  mrn TEXT NOT NULL UNIQUE,
  bed_number TEXT NOT NULL,
  admission_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'discharged')),
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create vitals table (Layer 1)
CREATE TABLE public.vitals (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  heart_rate NUMERIC,
  blood_pressure_sys NUMERIC,
  blood_pressure_dia NUMERIC,
  map NUMERIC,
  temperature NUMERIC,
  spo2 NUMERIC,
  respiratory_rate NUMERIC,
  mental_status TEXT NOT NULL DEFAULT 'normal' CHECK (mental_status IN ('normal', 'confused', 'agitated', 'reduced_gcs'))
);

-- Create labs table
CREATE TABLE public.labs (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  lactate NUMERIC,
  wbc NUMERIC,
  procalcitonin NUMERIC,
  creatinine NUMERIC,
  bilirubin NUMERIC,
  platelets NUMERIC
);

-- Create risk_assessments table (Layers 3 & 5)
CREATE TABLE public.risk_assessments (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  quantum_risk_score NUMERIC NOT NULL CHECK (quantum_risk_score >= 0 AND quantum_risk_score <= 1),
  confidence_interval_lower NUMERIC NOT NULL,
  confidence_interval_upper NUMERIC NOT NULL,
  tier TEXT NOT NULL CHECK (tier IN ('WATCH', 'AMBER', 'CRITICAL'))
);

-- Create tripwire_alerts table (Layer 4b)
CREATE TABLE public.tripwire_alerts (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  metric TEXT NOT NULL,
  value NUMERIC NOT NULL,
  threshold_breached TEXT NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT true
);

-- Create indexes for performance
CREATE INDEX idx_vitals_patient_time ON public.vitals(patient_id, timestamp DESC);
CREATE INDEX idx_labs_patient_time ON public.labs(patient_id, timestamp DESC);
CREATE INDEX idx_risk_patient_time ON public.risk_assessments(patient_id, timestamp DESC);
CREATE INDEX idx_tripwire_patient_active ON public.tripwire_alerts(patient_id, is_active, timestamp DESC);
CREATE INDEX idx_patients_status ON public.patients(status);

-- Enable RLS on all tables
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vitals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.labs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.risk_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tripwire_alerts ENABLE ROW LEVEL SECURITY;

-- RLS policies: allow all authenticated users to read (ICU staff)
CREATE POLICY "Authenticated users can read patients" ON public.patients FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read vitals" ON public.vitals FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read labs" ON public.labs FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read risk_assessments" ON public.risk_assessments FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read tripwire_alerts" ON public.tripwire_alerts FOR SELECT TO authenticated USING (true);

-- Also allow anon read for demo purposes
CREATE POLICY "Anon users can read patients" ON public.patients FOR SELECT TO anon USING (true);
CREATE POLICY "Anon users can read vitals" ON public.vitals FOR SELECT TO anon USING (true);
CREATE POLICY "Anon users can read labs" ON public.labs FOR SELECT TO anon USING (true);
CREATE POLICY "Anon users can read risk_assessments" ON public.risk_assessments FOR SELECT TO anon USING (true);
CREATE POLICY "Anon users can read tripwire_alerts" ON public.tripwire_alerts FOR SELECT TO anon USING (true);

-- Enable realtime replication
ALTER PUBLICATION supabase_realtime ADD TABLE public.vitals;
ALTER PUBLICATION supabase_realtime ADD TABLE public.risk_assessments;
ALTER PUBLICATION supabase_realtime ADD TABLE public.tripwire_alerts;
ALTER PUBLICATION supabase_realtime ADD TABLE public.patients;
