# Deployment Guide

Complete guide for deploying QuantumSepsis Shield to production.

## Deployment Architecture

```
┌─────────────────┐       ┌──────────────────────────┐       ┌──────────────────┐
│   Frontend      │──────▶│   AWS EC2 Backend         │       │   Supabase       │
│  (Vercel)       │ HTTPS │   54.242.66.27:8000       │       │  (Cloud DB)      │
│  React + Vite   │       │   api_server.py (FastAPI)  │       │  Postgres + Auth │
└─────────────────┘       └──────────────────────────┘       └──────────────────┘
```

## Part 1: Supabase Setup

### 1.1 Create Supabase Project

1. Go to https://supabase.com/dashboard
2. Click "New Project"
3. Configure:
   - **Name**: QuantumSepsis Shield
   - **Database Password**: Generate strong password (save it)
   - **Region**: Choose closest to your users
   - **Pricing Plan**: Free (sufficient for development/testing)
4. Wait ~2 minutes for project provisioning

### 1.2 Apply Database Migrations

See `DATABASE_SETUP.md` for detailed instructions.

Quick steps:
1. Dashboard → SQL Editor
2. Run migrations in order:
   - `supabase/migrations/001_initial_schema.sql`
   - `supabase/migrations/002_rls_policies.sql`
   - `supabase/migrations/003_seed_data.sql` (optional)
3. Verify tables exist in Table Editor

### 1.3 Enable Realtime

Dashboard → Database → Replication → Enable for:
- `vitals`, `labs`, `risk_assessments`, `tripwire_alerts`, `patients`

### 1.4 Configure Authentication

**Email Provider:**
- Dashboard → Authentication → Providers → Email
- Enable if not already enabled

**Google OAuth (Optional):**
1. Create OAuth credentials in Google Cloud Console
2. Dashboard → Authentication → Providers → Google
3. Add Client ID and Client Secret
4. Add redirect URI: `https://[project-ref].supabase.co/auth/v1/callback`

### 1.5 Get API Credentials

Dashboard → Settings → API

Copy these values (needed for deployment):
- **Project URL**: `https://[project-ref].supabase.co`
- **Anon Key**: Long JWT token starting with `eyJ...`
- **Project Ref**: The subdomain (e.g., `hntfeivuhmzdqlhtwxcn`)

---

## Part 2: Vercel Deployment

### 2.1 Prerequisites

- GitHub account
- Vercel account (free tier works)
- Frontend code pushed to GitHub repository

### 2.2 Connect Repository

1. Go to https://vercel.com/new
2. Import your GitHub repository: `Mish-atul/testing-quant`
3. Configure project:
   - **Framework Preset**: Vite
   - **Root Directory**: `sepsis-sentinel-main` (if not in repo root)
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

### 2.3 Set Environment Variables

Before deploying, add these environment variables:

Click "Environment Variables" section:

| Variable | Value | Environments |
|----------|-------|--------------|
| `VITE_SUPABASE_URL` | Your Supabase project URL | Production, Preview, Development |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Your anon key | Production, Preview, Development |
| `VITE_SUPABASE_PROJECT_ID` | Your project ref | Production, Preview, Development |

**Important:** Select all three environment checkboxes for each variable.

### 2.4 Deploy

1. Click "Deploy"
2. Wait ~2-3 minutes for build
3. Visit your deployment URL: `https://[project-name].vercel.app`

### 2.5 Configure Custom Domain (Optional)

1. Vercel Dashboard → Your Project → Settings → Domains
2. Add your domain (e.g., `sepsis.yourhospital.com`)
3. Update DNS records as instructed by Vercel
4. SSL certificate provisioned automatically

---

## Part 3: AWS Backend Setup

### 3.1 SSH Access

```bash
ssh -i quantum-key.pem ubuntu@54.242.66.27
```

**Important:** Never commit `quantum-key.pem` to git.

### 3.2 Backend Location

```bash
cd ~/QuantumSepsis_Complete_Backup
source venv/bin/activate
```

### 3.3 Install Dependencies (If Needed)

```bash
pip install fastapi uvicorn supabase-py python-dotenv
```

### 3.4 Configure Backend Environment

Create `.env` file in backend directory:

```bash
SUPABASE_URL=https://[project-ref].supabase.co
SUPABASE_SERVICE_KEY=[service-role-key]  # From Supabase Settings → API
```

**Security Warning:** The service key bypasses RLS. Never expose it in frontend code.

### 3.5 Start API Server

```bash
# Kill existing server
tmux kill-session -t api

# Start new server
tmux new -d -s api 'cd ~/QuantumSepsis_Complete_Backup && source venv/bin/activate && python api_server.py 2>&1 | tee /tmp/api.log'
```

### 3.6 Verify Backend

```bash
# Check server is running
curl http://localhost:8000/health

# Check logs
tail -f /tmp/api.log
```

### 3.7 Configure Security Groups

AWS Console → EC2 → Security Groups → Your Instance

**Inbound Rules:**
- Port 8000: Allow from Vercel IPs or 0.0.0.0/0 (less secure)
- Port 22: Allow from your IP only

---

## Part 4: Connect Frontend to Backend

### 4.1 Update API Proxy (Local Development)

`sepsis-sentinel-main/vite.config.ts`:

```typescript
export default defineConfig({
  server: {
    port: 8080,
    proxy: {
      '/api': {
        target: 'http://54.242.66.27:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
```

### 4.2 Update API Proxy (Production)

`sepsis-sentinel-main/vercel.json`:

```json
{
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "http://54.242.66.27:8000/:path*"
    }
  ]
}
```

**Note:** Vercel may have CORS issues with HTTP backends. Consider:
- Use HTTPS for backend (AWS Certificate Manager + Load Balancer)
- Or use Vercel API routes as proxy

### 4.3 Test Integration

1. Frontend calls `/api/predict` → Vercel proxy → AWS backend
2. Backend writes results to Supabase
3. Frontend receives real-time updates via WebSocket

---

## Part 5: Post-Deployment Checklist

### 5.1 Create Admin User

1. Visit your Vercel URL
2. Click "Register Hospital"
3. Complete signup
4. Get user ID from Supabase Dashboard → Authentication
5. Run SQL:

```sql
UPDATE public.profiles 
SET 
  hospital_id = '550e8400-e29b-41d4-a716-446655440000',
  role = 'admin'
WHERE user_id = '<user-id>';
```

### 5.2 Test Core Flows

- [ ] Login with admin user
- [ ] Navigate to dashboard
- [ ] Admit a test patient (attending/admin only)
- [ ] Log vitals (nurse/attending)
- [ ] Log labs (nurse/attending)
- [ ] Check real-time updates work
- [ ] Verify tripwire alerts display
- [ ] Test discharge patient (attending/admin only)

### 5.3 Verify Security

- [ ] RLS policies enforced (users only see their hospital)
- [ ] Role-based access working (admin page restricted)
- [ ] No service key exposed in frontend
- [ ] HTTPS enforced on production
- [ ] No sensitive data in console logs

### 5.4 Monitor Performance

**Supabase:**
- Dashboard → Reports → Check query performance
- Set up email alerts for downtime

**Vercel:**
- Analytics → Monitor response times
- Functions → Check serverless function logs

**AWS:**
- CloudWatch → Monitor EC2 CPU/memory
- Set up alarms for high load

---

## Part 6: Troubleshooting

### Issue: Vercel build fails

**Cause:** Missing dependencies or wrong build command

**Fix:**
```bash
# Verify locally
cd sepsis-sentinel-main
npm install
npm run build

# Check build logs in Vercel dashboard
```

### Issue: "CORS error" when calling backend

**Cause:** Backend not configured for CORS

**Fix:** Add to `api_server.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-vercel-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: Realtime not working

**Cause:** Replication not enabled or wrong credentials

**Fix:**
1. Supabase → Database → Replication → Enable tables
2. Hard refresh browser (Ctrl+Shift+R)
3. Check console for WebSocket errors

### Issue: "Row Level Security Policy violation"

**Cause:** User not linked to hospital or RLS policies wrong

**Fix:**
```sql
-- Check user's hospital
SELECT * FROM profiles WHERE user_id = auth.uid();

-- Verify RLS enabled
SELECT tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public';
```

---

## Part 7: Scaling Considerations

### For Production Use

**Database:**
- Upgrade to Supabase Pro for better performance
- Enable connection pooling (Supavisor)
- Add database indexes for slow queries
- Set up automated backups

**Backend:**
- Use AWS Auto Scaling Group
- Add Application Load Balancer
- Enable CloudWatch monitoring
- Set up log aggregation

**Frontend:**
- Use Vercel Pro for better analytics
- Enable Edge Middleware for auth
- Implement proper error boundaries
- Add performance monitoring (Sentry)

**Security:**
- Regular security audits
- Keep dependencies updated
- Monitor for anomalous access patterns
- Implement rate limiting

---

## Part 8: Maintenance

### Regular Tasks

**Weekly:**
- Check error logs (Vercel + AWS)
- Monitor database size (Supabase)
- Review user access patterns

**Monthly:**
- Update dependencies: `npm update`
- Review and optimize slow queries
- Check SSL certificate expiration
- Audit user roles and permissions

**Quarterly:**
- Review RLS policies for gaps
- Performance load testing
- Security penetration testing
- Backup restoration testing

---

## Quick Reference

### Useful Commands

```bash
# Frontend
npm run dev                    # Local development
npm run build                  # Production build
vercel --prod                  # Deploy to Vercel

# Backend (on AWS)
tmux attach -t api             # Attach to API server
tail -f /tmp/api.log           # View logs
htop                           # Monitor resources

# Database
supabase db push               # Apply migrations
supabase db reset              # Reset database (DESTRUCTIVE)
```

### Important URLs

- **Supabase Dashboard**: https://supabase.com/dashboard
- **Vercel Dashboard**: https://vercel.com/dashboard
- **AWS Console**: https://console.aws.amazon.com/ec2
- **Production URL**: https://testing-quant.vercel.app
- **Backend API**: http://54.242.66.27:8000

---

## Support

For issues specific to:
- **Frontend**: Check `QUICK_START.md`
- **Database**: Check `DATABASE_SETUP.md`
- **Vercel**: Check `VERCEL_SETUP.md`
- **Architecture**: Check main `CLAUDE.md`
