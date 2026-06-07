
-- 1. Add is_manual_entry column to vitals
ALTER TABLE public.vitals ADD COLUMN IF NOT EXISTS is_manual_entry boolean NOT NULL DEFAULT false;

-- 2. INSERT policy on patients for attending/admin
CREATE POLICY "Staff can admit patients to own hospital"
ON public.patients FOR INSERT TO authenticated
WITH CHECK (
  hospital_id = get_user_hospital_id(auth.uid())
  AND get_user_role(auth.uid()) IN ('attending', 'admin')
);

-- 3. INSERT policy on vitals for attending/nurse
CREATE POLICY "Staff can log vitals for own hospital patients"
ON public.vitals FOR INSERT TO authenticated
WITH CHECK (
  hospital_id = get_user_hospital_id(auth.uid())
  AND get_user_role(auth.uid()) IN ('attending', 'nurse')
);

-- 4. INSERT policy on hospitals for admin registration flow
CREATE POLICY "Admins can insert hospitals"
ON public.hospitals FOR INSERT TO authenticated
WITH CHECK (get_user_role(auth.uid()) = 'admin');

-- 5. Allow anyone to insert hospitals during registration (service role will handle this via edge function)
-- For now, allow authenticated users to insert (the register flow creates user first, then hospital)
CREATE POLICY "New users can insert hospitals during registration"
ON public.hospitals FOR INSERT TO authenticated
WITH CHECK (true);
