
DO $$
DECLARE
  hosp_id uuid := 'a0000000-0000-0000-0000-000000000001';
  admin_id uuid := 'd0000000-0000-0000-0000-0000000000a1';
  doc_id   uuid := 'd0000000-0000-0000-0000-0000000000d1';
  nurse_id uuid := 'd0000000-0000-0000-0000-0000000000e1';
  encrypted_pw text := crypt('Demo1234!', gen_salt('bf'));
BEGIN
  DELETE FROM auth.users WHERE id IN (admin_id, doc_id, nurse_id);

  INSERT INTO auth.users (instance_id, id, aud, role, email, encrypted_password, email_confirmed_at, raw_app_meta_data, raw_user_meta_data, created_at, updated_at, confirmation_token, email_change, email_change_token_new, recovery_token)
  VALUES
    ('00000000-0000-0000-0000-000000000000', admin_id, 'authenticated', 'authenticated',
      'admin@demo.hospital', encrypted_pw, now(),
      '{"provider":"email","providers":["email"]}'::jsonb,
      jsonb_build_object('full_name','Dr. Anjali Verma','department','Administration','employee_id','ADM-001','role','admin'),
      now(), now(), '', '', '', ''),
    ('00000000-0000-0000-0000-000000000000', doc_id, 'authenticated', 'authenticated',
      'doctor@demo.hospital', encrypted_pw, now(),
      '{"provider":"email","providers":["email"]}'::jsonb,
      jsonb_build_object('full_name','Dr. Rohan Mehta','department','ICU','employee_id','DOC-001','role','attending'),
      now(), now(), '', '', '', ''),
    ('00000000-0000-0000-0000-000000000000', nurse_id, 'authenticated', 'authenticated',
      'nurse@demo.hospital', encrypted_pw, now(),
      '{"provider":"email","providers":["email"]}'::jsonb,
      jsonb_build_object('full_name','Priya Nair','department','ICU','employee_id','NUR-001','role','nurse'),
      now(), now(), '', '', '', '');

  UPDATE public.profiles SET hospital_id = hosp_id WHERE user_id IN (admin_id, doc_id, nurse_id);
END $$;
