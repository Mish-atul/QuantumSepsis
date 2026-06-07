
-- 1. Create hospitals table
CREATE TABLE public.hospitals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  tier TEXT NOT NULL DEFAULT 'Tier-2',
  total_icu_beds INT NOT NULL DEFAULT 20,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE public.hospitals ENABLE ROW LEVEL SECURITY;

-- 2. Add hospital_id to profiles and employee_id
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS hospital_id UUID REFERENCES public.hospitals(id);
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS employee_id TEXT;

-- 3. Add hospital_id to clinical tables
ALTER TABLE public.patients ADD COLUMN IF NOT EXISTS hospital_id UUID REFERENCES public.hospitals(id);
ALTER TABLE public.vitals ADD COLUMN IF NOT EXISTS hospital_id UUID REFERENCES public.hospitals(id);
ALTER TABLE public.labs ADD COLUMN IF NOT EXISTS hospital_id UUID REFERENCES public.hospitals(id);
ALTER TABLE public.risk_assessments ADD COLUMN IF NOT EXISTS hospital_id UUID REFERENCES public.hospitals(id);
ALTER TABLE public.tripwire_alerts ADD COLUMN IF NOT EXISTS hospital_id UUID REFERENCES public.hospitals(id);

-- 4. Security definer: get user hospital_id
CREATE OR REPLACE FUNCTION public.get_user_hospital_id(_user_id UUID)
RETURNS UUID
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT hospital_id FROM public.profiles WHERE user_id = _user_id LIMIT 1
$$;

-- 5. Security definer: get user role
CREATE OR REPLACE FUNCTION public.get_user_role(_user_id UUID)
RETURNS TEXT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT role FROM public.profiles WHERE user_id = _user_id LIMIT 1
$$;

-- 6. RLS for hospitals - authenticated can read
CREATE POLICY "Authenticated users can read hospitals"
ON public.hospitals FOR SELECT TO authenticated
USING (true);

-- 7. Drop old RLS policies and create hospital-scoped ones

-- patients
DROP POLICY IF EXISTS "Anon users can read patients" ON public.patients;
DROP POLICY IF EXISTS "Authenticated users can read patients" ON public.patients;

CREATE POLICY "Users can read own hospital patients"
ON public.patients FOR SELECT TO authenticated
USING (hospital_id = public.get_user_hospital_id(auth.uid()));

CREATE POLICY "Attending can update patients"
ON public.patients FOR UPDATE TO authenticated
USING (hospital_id = public.get_user_hospital_id(auth.uid()) AND public.get_user_role(auth.uid()) = 'attending')
WITH CHECK (hospital_id = public.get_user_hospital_id(auth.uid()) AND public.get_user_role(auth.uid()) = 'attending');

-- vitals
DROP POLICY IF EXISTS "Anon users can read vitals" ON public.vitals;
DROP POLICY IF EXISTS "Authenticated users can read vitals" ON public.vitals;

CREATE POLICY "Users can read own hospital vitals"
ON public.vitals FOR SELECT TO authenticated
USING (hospital_id = public.get_user_hospital_id(auth.uid()));

-- labs
DROP POLICY IF EXISTS "Anon users can read labs" ON public.labs;
DROP POLICY IF EXISTS "Authenticated users can read labs" ON public.labs;

CREATE POLICY "Users can read own hospital labs"
ON public.labs FOR SELECT TO authenticated
USING (hospital_id = public.get_user_hospital_id(auth.uid()));

-- risk_assessments
DROP POLICY IF EXISTS "Anon users can read risk_assessments" ON public.risk_assessments;
DROP POLICY IF EXISTS "Authenticated users can read risk_assessments" ON public.risk_assessments;

CREATE POLICY "Users can read own hospital risk_assessments"
ON public.risk_assessments FOR SELECT TO authenticated
USING (hospital_id = public.get_user_hospital_id(auth.uid()));

-- tripwire_alerts
DROP POLICY IF EXISTS "Anon users can read tripwire_alerts" ON public.tripwire_alerts;
DROP POLICY IF EXISTS "Authenticated users can read tripwire_alerts" ON public.tripwire_alerts;

CREATE POLICY "Users can read own hospital tripwire_alerts"
ON public.tripwire_alerts FOR SELECT TO authenticated
USING (hospital_id = public.get_user_hospital_id(auth.uid()));

CREATE POLICY "Attending can update tripwire_alerts"
ON public.tripwire_alerts FOR UPDATE TO authenticated
USING (hospital_id = public.get_user_hospital_id(auth.uid()) AND public.get_user_role(auth.uid()) = 'attending')
WITH CHECK (hospital_id = public.get_user_hospital_id(auth.uid()) AND public.get_user_role(auth.uid()) = 'attending');

-- profiles: update policies for hospital isolation
DROP POLICY IF EXISTS "Authenticated users can read profiles" ON public.profiles;

CREATE POLICY "Users can read own hospital profiles"
ON public.profiles FOR SELECT TO authenticated
USING (hospital_id = public.get_user_hospital_id(auth.uid()) OR user_id = auth.uid());

-- Admin can insert/update profiles in their hospital
CREATE POLICY "Admin can insert profiles"
ON public.profiles FOR INSERT TO authenticated
WITH CHECK (public.get_user_role(auth.uid()) = 'admin' AND hospital_id = public.get_user_hospital_id(auth.uid()));

DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;

CREATE POLICY "Users can update own profile"
ON public.profiles FOR UPDATE TO authenticated
USING (user_id = auth.uid())
WITH CHECK (user_id = auth.uid());

CREATE POLICY "Admin can update hospital profiles"
ON public.profiles FOR UPDATE TO authenticated
USING (public.get_user_role(auth.uid()) = 'admin' AND hospital_id = public.get_user_hospital_id(auth.uid()))
WITH CHECK (public.get_user_role(auth.uid()) = 'admin' AND hospital_id = public.get_user_hospital_id(auth.uid()));
