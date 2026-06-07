-- ============================================================
-- SEED DATA FOR DEMO/TESTING
-- Creates demo hospital, patients, and sample vitals/labs
-- ============================================================

-- Insert demo hospital
INSERT INTO public.hospitals (id, name, tier, total_icu_beds)
VALUES
  ('550e8400-e29b-41d4-a716-446655440000', 'QuantumHealth Demo Hospital', 'premium', 24)
ON CONFLICT (id) DO NOTHING;

-- Insert demo patients (using the hospital ID from above)
INSERT INTO public.patients (id, mrn, name, bed_number, hospital_id, status, admission_time)
VALUES
  ('660e8400-e29b-41d4-a716-446655440001', 'MRN-001', 'Rajesh Kumar', 'ICU-01', '550e8400-e29b-41d4-a716-446655440000', 'active', now() - interval '2 days'),
  ('660e8400-e29b-41d4-a716-446655440002', 'MRN-002', 'Priya Sharma', 'ICU-02', '550e8400-e29b-41d4-a716-446655440000', 'active', now() - interval '1 day'),
  ('660e8400-e29b-41d4-a716-446655440003', 'MRN-003', 'Amit Singh', 'ICU-05', '550e8400-e29b-41d4-a716-446655440000', 'active', now() - interval '6 hours'),
  ('660e8400-e29b-41d4-a716-446655440004', 'MRN-004', 'Lakshmi Patel', 'ICU-08', '550e8400-e29b-41d4-a716-446655440000', 'active', now() - interval '12 hours')
ON CONFLICT (id) DO NOTHING;

-- Insert sample vitals for patients
-- Patient 1: High risk sepsis indicators
INSERT INTO public.vitals (patient_id, heart_rate, blood_pressure_sys, blood_pressure_dia, map, temperature, respiratory_rate, spo2, mental_status, hospital_id, timestamp)
VALUES
  ('660e8400-e29b-41d4-a716-446655440001', 112, 95, 55, 68, 38.8, 24, 92, 'confused', '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes'),
  ('660e8400-e29b-41d4-a716-446655440001', 108, 98, 58, 71, 38.6, 22, 93, 'confused', '550e8400-e29b-41d4-a716-446655440000', now() - interval '1 hour');

-- Patient 2: Moderate risk
INSERT INTO public.vitals (patient_id, heart_rate, blood_pressure_sys, blood_pressure_dia, map, temperature, respiratory_rate, spo2, mental_status, hospital_id, timestamp)
VALUES
  ('660e8400-e29b-41d4-a716-446655440002', 95, 110, 65, 80, 37.8, 19, 96, 'normal', '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes'),
  ('660e8400-e29b-41d4-a716-446655440002', 92, 112, 68, 83, 37.6, 18, 97, 'normal', '550e8400-e29b-41d4-a716-446655440000', now() - interval '1 hour');

-- Patient 3: Low risk
INSERT INTO public.vitals (patient_id, heart_rate, blood_pressure_sys, blood_pressure_dia, map, temperature, respiratory_rate, spo2, mental_status, hospital_id, timestamp)
VALUES
  ('660e8400-e29b-41d4-a716-446655440003', 78, 120, 75, 90, 37.0, 16, 98, 'normal', '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes'),
  ('660e8400-e29b-41d4-a716-446655440003', 75, 118, 72, 87, 36.8, 15, 99, 'normal', '550e8400-e29b-41d4-a716-446655440000', now() - interval '1 hour');

-- Patient 4: Critical tripwire
INSERT INTO public.vitals (patient_id, heart_rate, blood_pressure_sys, blood_pressure_dia, map, temperature, respiratory_rate, spo2, mental_status, hospital_id, timestamp)
VALUES
  ('660e8400-e29b-41d4-a716-446655440004', 125, 85, 45, 58, 39.2, 28, 89, 'agitated', '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes');

-- Insert sample labs
-- Patient 1: Sepsis indicators
INSERT INTO public.labs (patient_id, wbc, lactate, creatinine, platelets, bilirubin, procalcitonin, hospital_id, timestamp)
VALUES
  ('660e8400-e29b-41d4-a716-446655440001', 15.2, 3.8, 1.8, 110, 1.2, 2.5, '550e8400-e29b-41d4-a716-446655440000', now() - interval '2 hours');

-- Patient 2: Moderate elevation
INSERT INTO public.labs (patient_id, wbc, lactate, creatinine, platelets, bilirubin, procalcitonin, hospital_id, timestamp)
VALUES
  ('660e8400-e29b-41d4-a716-446655440002', 11.5, 2.1, 1.2, 180, 0.9, 0.8, '550e8400-e29b-41d4-a716-446655440000', now() - interval '3 hours');

-- Patient 3: Normal values
INSERT INTO public.labs (patient_id, wbc, lactate, creatinine, platelets, bilirubin, procalcitonin, hospital_id, timestamp)
VALUES
  ('660e8400-e29b-41d4-a716-446655440003', 8.2, 1.0, 0.9, 250, 0.7, 0.1, '550e8400-e29b-41d4-a716-446655440000', now() - interval '4 hours');

-- Insert sample risk assessments
INSERT INTO public.risk_assessments (patient_id, quantum_risk_score, tier, confidence_interval_lower, confidence_interval_upper, hospital_id, timestamp)
VALUES
  ('660e8400-e29b-41d4-a716-446655440001', 0.78, 'CRITICAL', 0.65, 0.88, '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes'),
  ('660e8400-e29b-41d4-a716-446655440002', 0.42, 'AMBER', 0.32, 0.55, '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes'),
  ('660e8400-e29b-41d4-a716-446655440003', 0.15, 'WATCH', 0.08, 0.25, '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes'),
  ('660e8400-e29b-41d4-a716-446655440004', 0.52, 'AMBER', 0.38, 0.68, '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes');

-- Insert sample tripwire alerts
-- Patient 1: Multiple tripwires (should escalate to CRITICAL)
INSERT INTO public.tripwire_alerts (patient_id, metric, value, threshold_breached, is_active, hospital_id, timestamp)
VALUES
  ('660e8400-e29b-41d4-a716-446655440001', 'Temperature', 38.8, '> 38.3°C', true, '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes'),
  ('660e8400-e29b-41d4-a716-446655440001', 'Heart Rate', 112, '> 90 bpm', true, '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes'),
  ('660e8400-e29b-41d4-a716-446655440001', 'Respiratory Rate', 24, '> 20/min', true, '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes');

-- Patient 4: Multiple tripwires (critical)
INSERT INTO public.tripwire_alerts (patient_id, metric, value, threshold_breached, is_active, hospital_id, timestamp)
VALUES
  ('660e8400-e29b-41d4-a716-446655440004', 'Temperature', 39.2, '> 38.3°C', true, '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes'),
  ('660e8400-e29b-41d4-a716-446655440004', 'MAP', 58, '< 70 mmHg', true, '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes'),
  ('660e8400-e29b-41d4-a716-446655440004', 'Respiratory Rate', 28, '> 20/min', true, '550e8400-e29b-41d4-a716-446655440000', now() - interval '15 minutes');

-- Note: To link a user to this hospital, run this after user signup:
-- UPDATE public.profiles
-- SET hospital_id = '550e8400-e29b-41d4-a716-446655440000', role = 'admin'
-- WHERE user_id = '<your-auth-user-id>';
