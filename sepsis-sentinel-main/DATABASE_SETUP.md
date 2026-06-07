# Database Setup Guide

This guide walks through setting up the Supabase database for QuantumSepsis Shield.

## Prerequisites

- Supabase account (free tier is sufficient for development)
- Supabase project created at https://supabase.com/dashboard

## Step 1: Get Your Supabase Credentials

1. Go to your Supabase project dashboard
2. Navigate to **Settings** → **API**
3. Copy the following values:
   - **Project URL**: `https://[your-project-ref].supabase.co`
   - **Anon/Public Key**: `eyJhbGc...` (long JWT token)
   - **Project ID**: The subdomain part (e.g., `hntfeivuhmzdqlhtwxcn`)

## Step 2: Set Environment Variables

### Local Development

Create `.env.local` in `sepsis-sentinel-main/`:

```bash
VITE_SUPABASE_URL=https://[your-project-ref].supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=eyJhbGc...
VITE_SUPABASE_PROJECT_ID=[your-project-ref]
```

### Vercel Production

Go to **Vercel Dashboard** → Your Project → **Settings** → **Environment Variables**

Add these three variables for **all environments** (Production, Preview, Development):

| Variable | Value |
|----------|-------|
| `VITE_SUPABASE_URL` | Your Supabase project URL |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Your anon key |
| `VITE_SUPABASE_PROJECT_ID` | Your project ref |

**Important:** After updating environment variables, trigger a new deployment.

## Step 3: Run Database Migrations

### Option A: Using Supabase SQL Editor (Recommended)

1. Go to **Supabase Dashboard** → **SQL Editor**
2. Create a new query
3. Copy and paste the contents of each migration file in order:
   - `supabase/migrations/001_initial_schema.sql`
   - `supabase/migrations/002_rls_policies.sql`
   - `supabase/migrations/003_seed_data.sql` (optional, for demo data)
4. Click **Run** for each migration

### Option B: Using Supabase CLI

```bash
# Install Supabase CLI
npm install -g supabase

# Login
supabase login

# Link to your project
supabase link --project-ref [your-project-ref]

# Apply migrations
supabase db push
```

## Step 4: Enable Realtime

1. Go to **Supabase Dashboard** → **Database** → **Replication**
2. Enable realtime for these tables:
   - ✅ `vitals`
   - ✅ `labs`
   - ✅ `risk_assessments`
   - ✅ `tripwire_alerts`
   - ✅ `patients`

## Step 5: Configure Authentication

### Email/Password Auth

1. Go to **Authentication** → **Providers**
2. Ensure **Email** provider is enabled
3. Configure email templates (optional):
   - Go to **Authentication** → **Email Templates**
   - Customize confirmation and password reset emails

### Google OAuth (Optional)

1. Go to **Authentication** → **Providers** → **Google**
2. Enable the Google provider
3. Add your Google OAuth credentials:
   - **Client ID**: From Google Cloud Console
   - **Client Secret**: From Google Cloud Console
4. Add authorized redirect URIs in Google Cloud Console:
   - `https://[your-project-ref].supabase.co/auth/v1/callback`
   - `http://localhost:8080/auth/callback` (for local dev)

## Step 6: Create First Admin User

### Manual Creation

1. Sign up through the frontend: `http://localhost:8080/register`
2. Get your user ID from **Authentication** → **Users** in Supabase Dashboard
3. Run this SQL in **SQL Editor**:

```sql
-- Link user to demo hospital and make them admin
UPDATE public.profiles 
SET 
  hospital_id = '550e8400-e29b-41d4-a716-446655440000',
  role = 'admin',
  full_name = 'Dr. Admin User',
  department = 'ICU'
WHERE user_id = '<paste-your-user-id-here>';
```

## Step 7: Verify Setup

### Check Tables

Go to **Table Editor** and verify these tables exist:
- `hospitals`
- `profiles`
- `patients`
- `vitals`
- `labs`
- `risk_assessments`
- `tripwire_alerts`

### Check RLS Policies

Go to **Authentication** → **Policies** for each table and verify policies are active.

### Test Frontend

1. Start the dev server: `npm run dev`
2. Login with your admin user
3. Navigate to `/dashboard`
4. You should see the demo patients (if you ran seed data)

## Troubleshooting

### "WebSocket connection failed" error

**Cause:** Wrong environment variables in Vercel or `.env.local`

**Fix:** Double-check that `VITE_SUPABASE_URL` and `VITE_SUPABASE_PUBLISHABLE_KEY` are correct.

### "No active patients" on dashboard

**Cause:** Either no patients exist, or user is not linked to a hospital

**Fix:** 
1. Check your profile: `SELECT * FROM profiles WHERE user_id = auth.uid();`
2. Verify `hospital_id` is set
3. Run seed data migration if needed

### "Access denied" when trying to view patients

**Cause:** RLS policies not applied or user role not set

**Fix:**
1. Verify RLS is enabled: `ALTER TABLE patients ENABLE ROW LEVEL SECURITY;`
2. Check your role: `SELECT role FROM profiles WHERE user_id = auth.uid();`
3. Re-run `002_rls_policies.sql` migration

### Changes not reflecting in Vercel production

**Cause:** Environment variables updated but deployment not triggered

**Fix:** 
1. Go to Vercel Dashboard → Deployments
2. Click **⋯** on latest deployment → **Redeploy**

## Security Notes

- **Never commit** `.env.local` to git
- **Never expose** the `service_role` key in frontend code
- The `anon` key is safe for client-side use (RLS enforces permissions)
- Always use RLS policies for multi-tenancy
- Verify that admins can only access their hospital's data

## Next Steps

- Connect ML backend to write risk scores and tripwire alerts
- Set up monitoring and alerting via Supabase dashboard
- Configure database backups (Supabase Pro feature)
- Add indexes for performance optimization (already in migration)
