
-- Fix the role check constraint to include 'attending'
ALTER TABLE public.profiles DROP CONSTRAINT profiles_role_check;
ALTER TABLE public.profiles ADD CONSTRAINT profiles_role_check 
  CHECK (role = ANY (ARRAY['doctor'::text, 'nurse'::text, 'admin'::text, 'technician'::text, 'attending'::text]));
