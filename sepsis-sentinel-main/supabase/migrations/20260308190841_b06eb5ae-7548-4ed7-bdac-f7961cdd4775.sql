
-- Allow admin to also update patients in their hospital
CREATE POLICY "Admin can update patients"
ON public.patients FOR UPDATE TO authenticated
USING (hospital_id = get_user_hospital_id(auth.uid()) AND get_user_role(auth.uid()) = 'admin')
WITH CHECK (hospital_id = get_user_hospital_id(auth.uid()) AND get_user_role(auth.uid()) = 'admin');
