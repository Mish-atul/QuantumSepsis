
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  INSERT INTO public.profiles (user_id, full_name, department, employee_id, role, hospital_id)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data ->> 'full_name', NEW.raw_user_meta_data ->> 'name', ''),
    COALESCE(NEW.raw_user_meta_data ->> 'department', 'ICU'),
    NEW.raw_user_meta_data ->> 'employee_id',
    COALESCE(NEW.raw_user_meta_data ->> 'role', 'nurse'),
    (SELECT id FROM public.hospitals LIMIT 1)
  );
  RETURN NEW;
END;
$$;
