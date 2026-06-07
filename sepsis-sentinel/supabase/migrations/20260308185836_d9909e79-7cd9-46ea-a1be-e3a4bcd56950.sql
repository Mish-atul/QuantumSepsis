
-- Allow attending/nurse to insert labs for their hospital's patients
CREATE POLICY "Staff can log labs for own hospital patients"
ON public.labs FOR INSERT TO authenticated
WITH CHECK (
  hospital_id = get_user_hospital_id(auth.uid())
  AND get_user_role(auth.uid()) IN ('attending', 'nurse')
);
