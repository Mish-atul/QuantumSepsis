# QuantumSepsis Quick Fix Reference Card

## 🚨 Critical: RLS Infinite Recursion Fix

### Replace This File:
```
❌ supabase/migrations/002_rls_policies.sql
✅ supabase/migrations/002_rls_policies_FIXED.sql
```

### Test Commands (in Supabase SQL Editor):
```sql
-- Verify helper functions work:
SELECT public.get_user_hospital_id(auth.uid());
SELECT public.get_user_role(auth.uid());
```

---

## 💡 Frontend Saves Predictions (2-Minute Integration)

### Add to any component calling `/predict`:

```typescript
// After getting prediction from API:
const result = await fetch('http://localhost:8000/predict', {...});
const prediction = await result.json();

// ✅ Save to database (frontend does this!)
await supabase.from("risk_assessments").insert({
  patient_id: patientId,
  quantum_risk_score: prediction.risk_score,
  tier: prediction.alert_level, // WATCH/AMBER/CRITICAL/FAST-TRACK
  confidence_interval_lower: prediction.conformal_interval[0],
  confidence_interval_upper: prediction.conformal_interval[1],
  hospital_id: profile.hospital_id,
});
```

---

## ✅ Status Summary

| Task | Status | Notes |
|------|--------|-------|
| RLS Policies | ⚠️ Fix ready | Use `_FIXED.sql` file |
| Backend Integration | 💡 Shortcut available | Frontend saves = faster |
| RBAC Implementation | ✅ Complete | `ProtectedRoute` works |
| Lovable Removal | ✅ Complete | Using Supabase directly |

---

## 📁 Important Files

```
sepsis-sentinel-main/
├── supabase/migrations/
│   └── 002_rls_policies_FIXED.sql     ← Use this!
├── HACKATHON_SHORTCUTS.md              ← Full examples
├── CRITICAL_ISSUES_STATUS.md           ← Detailed analysis
└── src/components/auth/
    └── ProtectedRoute.tsx              ← RBAC verified ✅
```

---

## 🚀 Running System

| Service | Status | URL |
|---------|--------|-----|
| Python API | 🟢 Running | http://localhost:8000 |
| React Frontend | 🟢 Running | http://localhost:8080 |

### Test API Health:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"healthy","model_loaded":true,"backend_mode":"ensemble"}
```

---

**Need Details?** See `CRITICAL_ISSUES_STATUS.md`  
**Need Code Examples?** See `HACKATHON_SHORTCUTS.md`
