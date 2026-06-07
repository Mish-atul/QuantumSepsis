-- ============================================================
-- ROW LEVEL SECURITY (RLS) POLICIES - FIXED VERSION
-- Multi-tenant hospital-scoped access control
-- ============================================================
-- 
-- CRITICAL FIX: Added SECURITY DEFINER helper functions to prevent
-- infinite recursion when policies query the same table they protect.
-- ============================================================

-- ============================================================
-- HELPER FUNCTIONS (SECURITY DEFINER to bypass RLS)
-- ============================================================

-- Get user's hospital_id without triggering RLS
CREATE OR REPLACE FUNCTION public.get_user_hospital_id(user_uuid UUID)
RETURNS UUID
LANGUAGE sql
SECURITY DEFINER
STABLE
AS $$
  SELECT hospital_id FROM public.profiles WHERE user_id = user_uuid LIMIT 1;
$$;

-- Get user's role without triggering RLS
CREATE OR REPLACE FUNCTION public.get_user_role(user_uuid UUID)
RETURNS TEXT
LANGUAGE sql
SECURITY DEFINER
STABLE
AS $$
  SELECT role FROM public.profiles WHERE user_id = user_uuid LIMIT 1;
$$;

-- Grant execute to authenticated users
GRANT EXECUTE ON FUNCTION public.get_user_hospital_id(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION public.get_user_role(UUID) TO authenticated;

-- ============================================================
-- Enable RLS on all tables
-- ============================================================

ALTER TABLE public.hospitals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vitals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.labs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.risk_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tripwire_alerts ENABLE ROW LEVEL SECURITY;

-- ============================================================
-- PROFILES POLICIES
-- ============================================================

-- Users can view own profile
CREATE POLICY "Users can view own profile" ON public.profiles
  FOR SELECT USING (auth.uid() = user_id);

-- Users can update own profile
CREATE POLICY "Users can update own profile" ON public.profiles
  FOR UPDATE USING (auth.uid() = user_id);

-- ✅ FIXED: Admins can view all profiles in their hospital (using helper functions)
CREATE POLICY "Admins can view hospital profiles" ON public.profiles
  FOR SELECT USING (
    hospital_id = public.get_user_hospital_id(auth.uid())
    AND public.get_user_role(auth.uid()) = 'admin'
  );

-- ============================================================
-- PATIENTS POLICIES
-- ============================================================

-- ✅ FIXED: All staff can view patients in their hospital
CREATE POLICY "Staff can view hospital patients" ON public.patients
  FOR SELECT USING (
    hospital_id = public.get_user_hospital_id(auth.uid())
  );

-- ✅ FIXED: Attendings and admins can admit patients
CREATE POLICY "Attendings can admit patients" ON public.patients
  FOR INSERT WITH CHECK (
    hospital_id = public.get_user_hospital_id(auth.uid())
    AND public.get_user_role(auth.uid()) IN ('attending', 'admin')
  );

-- ✅ FIXED: Staff can update patients in their hospital
CREATE POLICY "Staff can update patients" ON public.patients
  FOR UPDATE USING (
    hospital_id = public.get_user_hospital_id(auth.uid())
  );

-- ============================================================
-- VITALS POLICIES
-- ============================================================

-- ✅ FIXED: Staff can view vitals for their hospital
CREATE POLICY "Staff can view vitals" ON public.vitals
  FOR SELECT USING (
    hospital_id = public.get_user_hospital_id(auth.uid())
  );

-- ✅ FIXED: Nurses and attendings can log vitals
CREATE POLICY "Nurses can log vitals" ON public.vitals
  FOR INSERT WITH CHECK (
    hospital_id = public.get_user_hospital_id(auth.uid())
    AND public.get_user_role(auth.uid()) IN ('nurse', 'attending')
  );

-- ============================================================
-- LABS POLICIES
-- ============================================================

-- ✅ FIXED: Staff can view labs for their hospital
CREATE POLICY "Staff can view labs" ON public.labs
  FOR SELECT USING (
    hospital_id = public.get_user_hospital_id(auth.uid())
  );

-- ✅ FIXED: Nurses and attendings can log labs
CREATE POLICY "Nurses can log labs" ON public.labs
  FOR INSERT WITH CHECK (
    hospital_id = public.get_user_hospital_id(auth.uid())
    AND public.get_user_role(auth.uid()) IN ('nurse', 'attending')
  );

-- ============================================================
-- RISK ASSESSMENTS POLICIES
-- ============================================================

-- ✅ FIXED: Staff can view risk assessments for their hospital (read only)
CREATE POLICY "Staff can view risk assessments" ON public.risk_assessments
  FOR SELECT USING (
    hospital_id = public.get_user_hospital_id(auth.uid())
  );

-- System can insert risk assessments via service key
CREATE POLICY "System can insert risk assessments" ON public.risk_assessments
  FOR INSERT WITH CHECK (true);

-- ============================================================
-- TRIPWIRE ALERTS POLICIES
-- ============================================================

-- ✅ FIXED: Staff can view tripwire alerts for their hospital
CREATE POLICY "Staff can view tripwire alerts" ON public.tripwire_alerts
  FOR SELECT USING (
    hospital_id = public.get_user_hospital_id(auth.uid())
  );

-- System can manage tripwire alerts via service key
CREATE POLICY "System can manage tripwire alerts" ON public.tripwire_alerts
  FOR ALL USING (true);

-- ✅ FIXED: Attendings and admins can dismiss alerts
CREATE POLICY "Attendings can update alerts" ON public.tripwire_alerts
  FOR UPDATE USING (
    hospital_id = public.get_user_hospital_id(auth.uid())
    AND public.get_user_role(auth.uid()) IN ('attending', 'admin')
  );

-- ============================================================
-- HOSPITALS POLICIES
-- ============================================================

-- Anyone can view hospitals (needed for registration)
CREATE POLICY "Public can view hospitals" ON public.hospitals
  FOR SELECT USING (true);

-- Anyone can create hospital (for initial setup)
CREATE POLICY "Anyone can create hospital" ON public.hospitals
  FOR INSERT WITH CHECK (true);

-- ✅ FIXED: Admins can update their own hospital
CREATE POLICY "Admins can update hospital" ON public.hospitals
  FOR UPDATE USING (
    id = public.get_user_hospital_id(auth.uid())
    AND public.get_user_role(auth.uid()) = 'admin'
  );
