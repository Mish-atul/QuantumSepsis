# QuantumSepsis Shield - Frontend Dashboard

Real-time ICU monitoring dashboard for the QuantumSepsis Shield ML system. Visualizes quantum-classical AI predictions, tripwire alerts, and patient risk assessments.

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Create environment file (see .env.local.example)
cp .env.local.example .env.local
# Edit .env.local with your Supabase credentials

# Start development server
npm run dev
```

Opens at http://localhost:8080

**First time setup?** See [QUICK_START.md](./QUICK_START.md) for detailed instructions.

## 📚 Documentation

- **[QUICK_START.md](./QUICK_START.md)** - Get running in 5 minutes
- **[DATABASE_SETUP.md](./DATABASE_SETUP.md)** - Complete Supabase setup guide
- **[DEPLOYMENT.md](./DEPLOYMENT.md)** - Production deployment guide
- **[VERCEL_SETUP.md](./VERCEL_SETUP.md)** - Fix Vercel environment variables
- **[RBAC_REFERENCE.md](./RBAC_REFERENCE.md)** - Role-based access control guide

## 🏗️ Architecture

```
Frontend (React + Vite)
    ↓
Supabase (PostgreSQL + Realtime + Auth)
    ↑
Backend ML Pipeline (Python + FastAPI on AWS)
```

## 🎯 Key Features

- **Real-time monitoring** - WebSocket updates for vitals, labs, and risk scores
- **Multi-tenant architecture** - Hospital-scoped data isolation via RLS
- **Role-based access** - Nurse, attending, admin roles with granular permissions
- **Quantum-classical AI** - Visualizes LSTM + Quantum kernel predictions
- **Safety tripwires** - Non-overridable Red Team alerts
- **Human-in-the-loop** - Manual vitals/labs entry by ICU staff

## 🛠️ Tech Stack

- **Framework**: React 18 + TypeScript + Vite
- **UI Library**: shadcn/ui (Radix UI) + Tailwind CSS
- **Backend**: Supabase (PostgreSQL + Auth + Realtime)
- **State**: TanStack Query (React Query) + Context API
- **Routing**: React Router v6
- **Charts**: Recharts
- **Forms**: React Hook Form + Zod

## 📁 Project Structure

```
src/
├── components/         # UI components
│   ├── auth/          # Authentication (ProtectedRoute, SignUpForm)
│   ├── patient/       # Patient monitoring (RiskGauge, VitalsPanel, etc.)
│   ├── ward/          # Ward management (PatientCard, AdmitModal)
│   ├── observability/ # System monitoring (ActivityFeed, ConnectionStatus)
│   └── ui/            # shadcn/ui primitives (50+ components)
├── pages/             # Route pages (Landing, Login, Dashboard, etc.)
├── hooks/             # React Query data hooks (usePatients, useVitals, etc.)
├── contexts/          # Global state (HospitalContext - realtime subscriptions)
├── types/             # TypeScript definitions (database types)
├── lib/               # Utilities
└── integrations/      # External integrations (Supabase client)

supabase/
└── migrations/        # Database schema + RLS policies + seed data
```

## 🔑 Environment Variables

Create `.env.local`:

```bash
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your-anon-key
VITE_SUPABASE_PROJECT_ID=your-project-ref
```

See `.env.local.example` for details.

## 🧪 Development

```bash
# Development server (with hot reload)
npm run dev

# Production build
npm run build

# Preview production build
npm run preview

# Linting
npm run lint

# Tests
npm test
npm run test:watch
```

## 🗄️ Database Setup

1. Create Supabase project at https://supabase.com
2. Run migrations in SQL Editor:
   - `supabase/migrations/001_initial_schema.sql`
   - `supabase/migrations/002_rls_policies.sql`
   - `supabase/migrations/003_seed_data.sql` (optional demo data)
3. Enable Realtime replication for: `vitals`, `labs`, `risk_assessments`, `tripwire_alerts`, `patients`

See [DATABASE_SETUP.md](./DATABASE_SETUP.md) for detailed instructions.

## 👥 Role-Based Access Control

Three roles with distinct permissions:

| Role | Permissions |
|------|-------------|
| **Nurse** | View dashboard, log vitals/labs, view risk scores |
| **Attending** | All nurse permissions + admit/discharge patients |
| **Admin** | All permissions + manage staff, access admin panel |

See [RBAC_REFERENCE.md](./RBAC_REFERENCE.md) for complete permission matrix.

## 🚀 Deployment

### Vercel (Recommended)

1. Push code to GitHub
2. Import repository in Vercel
3. Set environment variables (see VERCEL_SETUP.md)
4. Deploy

```bash
# Or use Vercel CLI
npm install -g vercel
vercel --prod
```

See [DEPLOYMENT.md](./DEPLOYMENT.md) for complete guide.

## 🔐 Security

- **Row Level Security (RLS)** - All database tables enforce hospital-scoped access
- **Multi-tenancy** - Users only see data from their hospital
- **Role-based permissions** - Enforced at database level, not just UI
- **Environment variables** - Never commit secrets to git
- **Anon key safe** - The publishable key is safe for client-side use (RLS enforces security)

**Never expose:**
- ❌ Service role key
- ❌ Database password
- ❌ JWT secret
- ❌ `.env.local` file

## 🐛 Troubleshooting

### "WebSocket connection failed"
**Fix:** Check environment variables in `.env.local` or Vercel dashboard

### "No patients showing"
**Fix:** Run seed data or create test patients, verify user has `hospital_id` set

### "Access denied"
**Fix:** Check RLS policies are enabled, verify user role in database

### "Build failed"
**Fix:** Run `npm install` and `npm run build` locally to test

See documentation files for detailed troubleshooting guides.

## 🔄 Real-time Updates

The dashboard uses Supabase Realtime for live updates:

```typescript
// HospitalContext subscribes to changes
supabase
  .channel(`hospital-${hospitalId}`)
  .on("postgres_changes", {
    event: "INSERT",
    schema: "public",
    table: "risk_assessments",
  }, (payload) => {
    // UI updates automatically via React Query cache invalidation
  })
  .subscribe();
```

## 🧬 ML Integration

**Backend writes → Supabase → Frontend displays**

1. Python ML backend generates prediction
2. Backend writes to Supabase tables:
   - `risk_assessments` - Quantum risk scores + confidence intervals
   - `tripwire_alerts` - Red Team safety alerts
3. Supabase broadcasts INSERT events via WebSocket
4. Frontend React Query cache invalidates
5. UI updates within ~100ms

## 📊 Component Hierarchy

```
App
├── GlobalNav (hospital name, user profile, theme toggle)
├── Index (Ward Dashboard)
│   ├── PatientCard (risk gauge, tier badge, sparkline)
│   │   └── RiskSparkline
│   └── PipelineActivityFeed (live event stream)
└── PatientDetail
    ├── PatientTopBar
    ├── RiskGauge (quantum risk score)
    ├── ConfidenceInterval (conformal prediction)
    ├── TripwirePanel (Red Team alerts)
    ├── ActionPanel (orchestrator decisions)
    ├── VitalsChart (time-series)
    ├── VitalsPanel (current snapshot)
    ├── LabsPanel
    ├── LogVitalsDrawer (HITL)
    ├── LogLabsDrawer (HITL)
    └── DischargePatientDialog
```

## 🎨 Theming

Uses CSS variables for theming. See `src/index.css`:

```css
:root {
  --primary: hsl(217, 91%, 60%);
  --tier-critical: hsl(0, 84%, 60%);
  --tier-amber: hsl(38, 92%, 50%);
  --tier-watch: hsl(142, 71%, 45%);
}
```

Toggle dark/light mode via `ThemeToggle` component.

## 📝 Code Style

- **TypeScript** for type safety
- **Functional components** with hooks
- **Tailwind CSS** for styling (no custom CSS files)
- **React Query** for server state management
- **Context API** for global state (minimal)

## 🧪 Testing

```bash
# Run all tests
npm test

# Watch mode
npm run test:watch

# Coverage report
npm run test:coverage
```

Test files: `src/**/*.test.tsx`

## 🤝 Contributing

1. Read architecture in main [CLAUDE.md](../CLAUDE.md)
2. Check role permissions in [RBAC_REFERENCE.md](./RBAC_REFERENCE.md)
3. Test locally before deploying
4. Ensure RLS policies enforce security

## 📦 Dependencies

Key packages:
- `@supabase/supabase-js` - Supabase client
- `@tanstack/react-query` - Data fetching/caching
- `react-router-dom` - Routing
- `recharts` - Charts
- `zod` + `react-hook-form` - Form validation
- `lucide-react` - Icons
- `tailwindcss` - Styling

See `package.json` for complete list.

## 🔗 Related Repositories

- **Main ML Pipeline**: [Mish-atul/QuantumSepsis](https://github.com/Mish-atul/QuantumSepsis)
- **Vercel Deployment**: [Mish-atul/testing-quant](https://github.com/Mish-atul/testing-quant)

## 📄 License

See main project LICENSE file.

## 🆘 Support

- **Quick issues**: See troubleshooting sections in documentation
- **Database setup**: [DATABASE_SETUP.md](./DATABASE_SETUP.md)
- **Deployment help**: [DEPLOYMENT.md](./DEPLOYMENT.md)
- **Role questions**: [RBAC_REFERENCE.md](./RBAC_REFERENCE.md)

## 🎯 Project Status

- ✅ Frontend: Complete and production-ready
- ✅ Database schema: Defined (needs to be applied)
- ✅ RLS policies: Complete
- ⚠️ Vercel deployment: Needs env var fix (5 min)
- ⏳ Backend integration: Needs Supabase client added

See [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) for detailed status.

---

**Built for:** MIMIC-IV ICU data | 94,458 stays | 12,972 sepsis cases

**ML Stack:** BiLSTM → Quantum Kernel → Conformal Prediction → Red Team Tripwires
