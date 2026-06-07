# Vercel Deployment Fix Guide

## Problem

The Vercel deployment at `testing-quant.vercel.app` has incorrect environment variables. The Supabase API key is set to an old EC2 IP address instead of the actual Supabase credentials, causing WebSocket errors and broken authentication.

## Symptoms

- Console error: `WebSocket connection to 'wss://...supabase.co/...?apikey=34.224.69.251'`
- Authentication completely broken on production
- "Error" badge in navigation bar
- Unable to login or access any protected routes

## Fix Steps

### 1. Access Vercel Dashboard

Go to: https://vercel.com/dashboard

Navigate to your project: `testing-quant`

### 2. Delete Old Environment Variables

1. Click **Settings** tab
2. Click **Environment Variables** in sidebar
3. Delete ALL existing `VITE_SUPABASE_*` variables:
   - Delete `VITE_SUPABASE_URL`
   - Delete `VITE_SUPABASE_PUBLISHABLE_KEY`
   - Delete `VITE_SUPABASE_PROJECT_ID`
   - Delete any other Supabase-related variables

### 3. Add Correct Environment Variables

Add these three variables for **ALL environments** (Production, Preview, Development):

Click **Add New** for each variable:

#### Variable 1: VITE_SUPABASE_URL

```
Name: VITE_SUPABASE_URL
Value: https://hntfeivuhmzdqlhtwxcn.supabase.co
Environments: ✅ Production ✅ Preview ✅ Development
```

#### Variable 2: VITE_SUPABASE_PUBLISHABLE_KEY

```
Name: VITE_SUPABASE_PUBLISHABLE_KEY
Value: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhudGZlaXZ1aG16ZHFsaHR3eGNuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI5NzY0OTAsImV4cCI6MjA4ODU1MjQ5MH0.kHpRKIjT7V4XsSW4atMRlh2k6SqB8ElOo5WH8-FNUBo
Environments: ✅ Production ✅ Preview ✅ Development
```

#### Variable 3: VITE_SUPABASE_PROJECT_ID

```
Name: VITE_SUPABASE_PROJECT_ID
Value: hntfeivuhmzdqlhtwxcn
Environments: ✅ Production ✅ Preview ✅ Development
```

### 4. Trigger Redeploy

**Important:** Environment variable changes require a new deployment to take effect.

Two options:

#### Option A: Redeploy from Dashboard (Fast)

1. Click **Deployments** tab
2. Find the latest deployment
3. Click **⋯** (three dots) → **Redeploy**
4. Confirm the redeploy

#### Option B: Push to GitHub (Automatic)

1. Make any commit to your repository
2. Push to main branch
3. Vercel will auto-deploy

```bash
git commit --allow-empty -m "fix: trigger redeploy for env vars"
git push origin main
```

### 5. Verify Fix

Once deployment completes (usually 2-3 minutes):

1. Visit: https://testing-quant.vercel.app
2. Open browser DevTools → Console tab
3. Check for errors:
   - ✅ **No WebSocket errors** = Fixed
   - ❌ **Still see errors** = Recheck env vars
4. Try logging in - should work correctly
5. Check navigation bar - "Error" badge should be gone

## Troubleshooting

### Issue: Still seeing WebSocket errors after redeploy

**Cause:** Browser cached old JavaScript bundle

**Fix:**
1. Hard refresh: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
2. Or clear browser cache and reload

### Issue: Environment variables not showing up

**Cause:** Variables not set for all environments

**Fix:**
1. Go back to Settings → Environment Variables
2. Click on each variable
3. Ensure all three checkboxes are checked:
   - ✅ Production
   - ✅ Preview
   - ✅ Development

### Issue: "Failed to fetch" errors

**Cause:** Supabase project paused or credentials wrong

**Fix:**
1. Verify Supabase project is active: https://supabase.com/dashboard
2. Double-check credentials match exactly (no extra spaces)
3. Verify project ref `hntfeivuhmzdqlhtwxcn` is correct

## Security Note

The `VITE_SUPABASE_PUBLISHABLE_KEY` is the **anon key** - it's safe to expose in client-side code. Supabase Row Level Security (RLS) policies enforce permissions at the database level, so even if someone has the anon key, they can only access data their role permits.

**Never expose:**
- ❌ Service role key
- ❌ Database password
- ❌ JWT secret

## Next Steps After Fix

1. Set up database (see `DATABASE_SETUP.md`)
2. Create admin user and link to hospital
3. Test full login → dashboard → patient flow
4. Configure Google OAuth if needed (optional)

## Quick Verification Checklist

```
[ ] Old env vars deleted in Vercel
[ ] Three new env vars added (all environments)
[ ] Redeployed successfully
[ ] No WebSocket errors in console
[ ] Can login successfully
[ ] Dashboard loads without errors
[ ] Connection status shows "Connected" (green)
```

## Reference: Where These Values Come From

These credentials are from the Supabase project created for QuantumSepsis Shield:

- **Project Name**: QuantumSepsis Shield
- **Project Ref**: `hntfeivuhmzdqlhtwxcn`
- **Region**: US East (Ohio)
- **URL**: https://hntfeivuhmzdqlhtwxcn.supabase.co

To get credentials for a different project:
1. Go to Supabase Dashboard → Your Project
2. Settings → API
3. Copy Project URL and anon public key
