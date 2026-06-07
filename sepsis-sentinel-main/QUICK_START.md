# Quick Start Guide

This guide gets you up and running with QuantumSepsis Shield frontend in 5 minutes.

## Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Supabase account (free tier works)

## Step 1: Clone and Install

```bash
cd sepsis-sentinel-main
npm install
```

## Step 2: Set Up Environment Variables

Create `.env.local` file:

```bash
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your-anon-key-here
VITE_SUPABASE_PROJECT_ID=your-project-ref
```

Get these values from your Supabase project:
- Dashboard → Settings → API

## Step 3: Set Up Database

Run the SQL migrations in order:

1. Open Supabase Dashboard → SQL Editor
2. Run `supabase/migrations/001_initial_schema.sql`
3. Run `supabase/migrations/002_rls_policies.sql`
4. Run `supabase/migrations/003_seed_data.sql` (optional demo data)

See `DATABASE_SETUP.md` for detailed instructions.

## Step 4: Enable Realtime

Supabase Dashboard → Database → Replication

Enable realtime for:
- ✅ `vitals`
- ✅ `labs`
- ✅ `risk_assessments`
- ✅ `tripwire_alerts`
- ✅ `patients`

## Step 5: Start Development Server

```bash
npm run dev
```

Opens at: http://localhost:8080

## Step 6: Create Admin User

1. Click "Register Hospital" on login page
2. Fill in hospital details and admin credentials
3. After signup, get your user ID from Supabase Dashboard → Authentication → Users
4. Link user to demo hospital:

```sql
UPDATE public.profiles 
SET 
  hospital_id = '550e8400-e29b-41d4-a716-446655440000',
  role = 'admin'
WHERE user_id = '<your-user-id>';
```

## Step 7: Test the Dashboard

Login and navigate to `/dashboard`. You should see:

- ✅ Demo patients (if you ran seed data)
- ✅ Live pipeline feed on right sidebar
- ✅ No console errors
- ✅ Green "Connected" status in nav bar

## Project Structure

```
sepsis-sentinel-main/
├── src/
│   ├── components/       # UI components
│   │   ├── auth/        # Authentication
│   │   ├── patient/     # Patient monitoring
│   │   ├── ward/        # Ward management
│   │   └── ui/          # shadcn/ui primitives
│   ├── pages/           # Route pages
│   ├── hooks/           # React Query data hooks
│   ├── contexts/        # Global state (Hospital)
│   └── types/           # TypeScript definitions
├── supabase/
│   └── migrations/      # Database schema & policies
└── public/              # Static assets
```

## Common Commands

```bash
# Development
npm run dev              # Start dev server (port 8080)
npm run build            # Production build
npm run preview          # Preview production build

# Quality
npm run lint             # Check code style
npm test                 # Run tests
npm run test:watch       # Watch mode

# Database
# (Run SQL files in Supabase Dashboard → SQL Editor)
```

## Role-Based Access

The system has three roles:

| Role | Permissions |
|------|-------------|
| **Nurse** | View dashboard, log vitals/labs, view risk scores |
| **Attending** | All nurse permissions + admit/discharge patients, override alerts |
| **Admin** | All permissions + manage staff, hospital settings |

Default role on signup: `nurse`

Change role in Supabase:
```sql
UPDATE profiles SET role = 'admin' WHERE user_id = '<user-id>';
```

## Frontend ↔ Backend Integration

The frontend displays outputs from the Python ML backend:

1. **Backend** (Python) runs predictions → writes to Supabase
2. **Supabase** broadcasts real-time updates via WebSocket
3. **Frontend** (React) subscribes to changes → UI updates automatically

**Example:** When backend writes a new risk score to `risk_assessments` table, the patient card on dashboard updates within 100ms.

## Next Steps

- [ ] Connect ML backend (see main CLAUDE.md)
- [ ] Configure AWS API server proxy (see vite.config.ts)
- [ ] Set up production deployment (see DEPLOYMENT.md)
- [ ] Configure Google OAuth (optional)
- [ ] Customize theme colors (tailwind.config.ts)

## Troubleshooting

### "Failed to connect to Supabase"

- Verify `.env.local` credentials are correct
- Check Supabase project is active (not paused)
- Ensure `VITE_` prefix on all variables

### "No patients showing"

- Run seed data: `003_seed_data.sql`
- Verify user has `hospital_id` set in profile
- Check RLS policies are enabled

### "Access denied"

- Verify RLS policies were applied
- Check user role: `SELECT role FROM profiles WHERE user_id = auth.uid();`
- Ensure user is linked to hospital

### WebSocket errors

- Enable Realtime replication in Supabase
- Check browser console for specific errors
- Try hard refresh (Ctrl+Shift+R)

## Getting Help

- **Database issues**: See `DATABASE_SETUP.md`
- **Deployment issues**: See `DEPLOYMENT.md`
- **Vercel errors**: See `VERCEL_SETUP.md`
- **Architecture**: See `CLAUDE.md` (main project docs)

## Development Tips

- Use React DevTools to inspect component state
- Check Network tab for API call failures
- Enable verbose logging: Add `?debug=true` to URL
- Supabase logs: Dashboard → Logs & Reports

## Production Deployment

See `DEPLOYMENT.md` for complete Vercel deployment guide.

Quick deploy:
```bash
npm run build
# Upload dist/ to Vercel or any static host
```

Remember to set environment variables in production!
