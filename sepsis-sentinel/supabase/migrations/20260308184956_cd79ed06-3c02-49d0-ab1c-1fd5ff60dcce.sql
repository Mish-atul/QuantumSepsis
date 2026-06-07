
-- Remove the overly permissive policy and replace with a more targeted one
DROP POLICY IF EXISTS "Admins can insert hospitals" ON public.hospitals;
DROP POLICY IF EXISTS "New users can insert hospitals during registration" ON public.hospitals;

-- Single policy: any authenticated user can insert a hospital (needed for registration flow)
-- but we scope it tightly - the edge function for registration will handle validation
CREATE POLICY "Authenticated can register hospitals"
ON public.hospitals FOR INSERT TO authenticated
WITH CHECK (
  -- Only allow if the user doesn't already belong to a hospital
  get_user_hospital_id(auth.uid()) IS NULL
  OR get_user_role(auth.uid()) = 'admin'
);
