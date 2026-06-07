# Role-Based Access Control Reference

Quick reference for implementing and understanding role-based access in QuantumSepsis Shield.

## Role Hierarchy

```
Admin (highest privilege)
  ↓
Attending (clinical lead)
  ↓
Nurse (bedside care)
```

## Role Definitions

### Nurse
**Typical users:** ICU nurses, critical care nurses, charge nurses

**Permissions:**
- ✅ View ward dashboard
- ✅ View all patients in their hospital
- ✅ View patient vitals, labs, risk scores
- ✅ Log vitals (manual entry via HITL)
- ✅ Log labs (manual entry via HITL)
- ✅ View tripwire alerts
- ❌ Cannot admit/discharge patients
- ❌ Cannot override AI alerts
- ❌ Cannot manage staff
- ❌ Cannot access admin panel

### Attending (Doctor)
**Typical users:** ICU attending physicians, intensivists, resident physicians

**Permissions:**
- ✅ All nurse permissions
- ✅ Admit new patients to ICU
- ✅ Discharge patients from ICU
- ✅ Override AI risk assessments (future feature)
- ✅ View full risk score details with confidence intervals
- ❌ Cannot manage staff
- ❌ Cannot access admin panel

### Admin
**Typical users:** ICU directors, hospital administrators, system administrators

**Permissions:**
- ✅ All attending permissions
- ✅ Access admin panel
- ✅ Manage hospital staff (add/remove/change roles)
- ✅ View hospital-wide analytics
- ✅ Configure system settings
- ✅ Manage hospital profile

---

## How to Implement Role-Based Access

### 1. Route-Level Protection

Use `ProtectedRoute` with `allowedRoles` prop:

```tsx
// In App.tsx
<Route
  path="/admin"
  element={
    <ProtectedRoute allowedRoles={["admin"]}>
      <Admin />
    </ProtectedRoute>
  }
/>

<Route
  path="/analytics"
  element={
    <ProtectedRoute allowedRoles={["admin", "attending"]}>
      <Analytics />
    </ProtectedRoute>
  }
/>
```

### 2. Component-Level Visibility

Hide entire components based on role:

```tsx
import { useAuth } from "@/hooks/useAuth";

export function AdmitPatientButton() {
  const { profile } = useAuth();
  
  // Only show to attending and admin
  if (!profile || !["attending", "admin"].includes(profile.role)) {
    return null;
  }
  
  return <Button>Admit Patient</Button>;
}
```

### 3. Conditional UI Elements

Show/hide buttons or features within a component:

```tsx
export function PatientDetail() {
  const { profile } = useAuth();
  const isAttendingOrAdmin = profile?.role === "attending" || profile?.role === "admin";
  
  return (
    <div>
      <PatientInfo />
      <VitalsPanel />
      
      {/* Only attending/admin see discharge button */}
      {isAttendingOrAdmin && (
        <DischargeButton />
      )}
      
      {/* Everyone sees vitals logger */}
      <LogVitalsDrawer />
    </div>
  );
}
```

### 4. Action Restrictions

Disable actions based on role:

```tsx
export function ActionPanel() {
  const { profile } = useAuth();
  const canOverride = ["attending", "admin"].includes(profile?.role || "");
  
  return (
    <Button 
      disabled={!canOverride}
      onClick={handleOverride}
    >
      Override AI Assessment
    </Button>
  );
}
```

---

## Database-Level Security (RLS)

Row Level Security policies enforce permissions at the database level.

### Patients Table

```sql
-- All staff can view patients in their hospital
CREATE POLICY "Staff can view hospital patients" ON patients
  FOR SELECT USING (
    hospital_id = (SELECT hospital_id FROM profiles WHERE user_id = auth.uid())
  );

-- Only attending/admin can admit patients
CREATE POLICY "Attendings can admit patients" ON patients
  FOR INSERT WITH CHECK (
    hospital_id = (SELECT hospital_id FROM profiles WHERE user_id = auth.uid())
    AND (SELECT role FROM profiles WHERE user_id = auth.uid()) IN ('attending', 'admin')
  );
```

### Vitals Table

```sql
-- Nurses and attendings can log vitals
CREATE POLICY "Nurses can log vitals" ON vitals
  FOR INSERT WITH CHECK (
    hospital_id = (SELECT hospital_id FROM profiles WHERE user_id = auth.uid())
    AND (SELECT role FROM profiles WHERE user_id = auth.uid()) IN ('nurse', 'attending')
  );
```

---

## Permission Matrix

| Feature | Nurse | Attending | Admin |
|---------|-------|-----------|-------|
| **Dashboard** |
| View ward dashboard | ✅ | ✅ | ✅ |
| View patient cards | ✅ | ✅ | ✅ |
| View risk scores | ✅ | ✅ | ✅ |
| View confidence intervals | Limited | ✅ | ✅ |
| **Patient Management** |
| Admit patient | ❌ | ✅ | ✅ |
| Discharge patient | ❌ | ✅ | ✅ |
| Update patient info | ❌ | ✅ | ✅ |
| **Clinical Data** |
| View vitals | ✅ | ✅ | ✅ |
| Log vitals (manual) | ✅ | ✅ | ❌ |
| View labs | ✅ | ✅ | ✅ |
| Log labs (manual) | ✅ | ✅ | ❌ |
| **AI System** |
| View risk assessments | ✅ | ✅ | ✅ |
| View tripwire alerts | ✅ | ✅ | ✅ |
| Override AI alerts | ❌ | ✅ | ✅ |
| Dismiss tripwires | ❌ | ✅ | ✅ |
| **Administration** |
| Access admin panel | ❌ | ❌ | ✅ |
| Manage staff | ❌ | ❌ | ✅ |
| View analytics | ❌ | Limited | ✅ |
| Configure settings | ❌ | ❌ | ✅ |

---

## How to Change User Roles

### Method 1: Supabase Dashboard (Manual)

1. Go to Supabase Dashboard → Authentication → Users
2. Find the user and copy their `id`
3. Go to SQL Editor
4. Run:

```sql
UPDATE public.profiles
SET role = 'admin'  -- or 'attending', 'nurse'
WHERE user_id = '<user-id>';
```

### Method 2: Admin Panel (Future Feature)

The admin panel (accessible to admin role only) will have a UI for managing staff:

```tsx
// Future implementation in Admin.tsx
<Select value={user.role} onChange={(role) => updateUserRole(user.id, role)}>
  <SelectItem value="nurse">Nurse</SelectItem>
  <SelectItem value="attending">Attending</SelectItem>
  <SelectItem value="admin">Admin</SelectItem>
</Select>
```

---

## Testing Role-Based Access

### Test Scenario 1: Nurse User

1. Create user and set role to `nurse`
2. Login and navigate to dashboard ✅
3. Try to admit patient → Button should be hidden ❌
4. Log vitals → Should work ✅
5. Try to access `/admin` → Should redirect ❌

### Test Scenario 2: Attending User

1. Create user and set role to `attending`
2. Login and navigate to dashboard ✅
3. Admit patient → Should work ✅
4. Discharge patient → Should work ✅
5. Try to access `/admin` → Should redirect ❌

### Test Scenario 3: Admin User

1. Create user and set role to `admin`
2. Login and navigate to dashboard ✅
3. All patient operations work ✅
4. Access `/admin` → Should work ✅
5. View staff management → Should work ✅

---

## Security Best Practices

### Frontend
- ✅ Always check `profile.role` before rendering sensitive UI
- ✅ Hide buttons/actions for unauthorized roles (return `null`)
- ✅ Use `ProtectedRoute` for role-restricted pages
- ⚠️ Frontend checks are for UX only - not security

### Backend (RLS)
- ✅ Always use RLS policies on all tables
- ✅ Never bypass RLS in application code
- ✅ Test policies with different user roles
- ✅ Use `auth.uid()` to get current user ID in policies

### Never Do This ❌
```tsx
// BAD - Frontend only check, no database enforcement
if (profile?.role === "admin") {
  await supabase.from("users").delete().eq("id", userId);
}
```

### Always Do This ✅
```tsx
// GOOD - RLS policy prevents unauthorized deletes at database level
// Frontend check is just for UX
if (profile?.role === "admin") {
  // This will fail at database level if RLS policy doesn't allow it
  await supabase.from("users").delete().eq("id", userId);
}
```

---

## Common Patterns

### Pattern 1: Show Different UI by Role

```tsx
export function PatientActions() {
  const { profile } = useAuth();
  
  switch (profile?.role) {
    case "admin":
      return <AdminActions />;
    case "attending":
      return <AttendingActions />;
    case "nurse":
      return <NurseActions />;
    default:
      return <BasicActions />;
  }
}
```

### Pattern 2: Check Multiple Roles

```tsx
const canManagePatients = ["attending", "admin"].includes(profile?.role || "");
const canLogData = ["nurse", "attending"].includes(profile?.role || "");
```

### Pattern 3: Conditional Navigation

```tsx
export function GlobalNav() {
  const { profile } = useAuth();
  
  return (
    <nav>
      <Link to="/dashboard">Dashboard</Link>
      {["attending", "admin"].includes(profile?.role || "") && (
        <Link to="/analytics">Analytics</Link>
      )}
      {profile?.role === "admin" && (
        <Link to="/admin">Admin Panel</Link>
      )}
    </nav>
  );
}
```

---

## Troubleshooting

### User can see features they shouldn't

**Check:**
1. Profile role is set correctly in database
2. Component has proper role check
3. RLS policy is enabled on table

### User can't perform allowed actions

**Check:**
1. RLS policy includes their role
2. User is linked to hospital (`hospital_id` not null)
3. User's hospital matches data's hospital

### Role changes not taking effect

**Fix:**
1. Hard refresh browser (Ctrl+Shift+R)
2. Logout and login again
3. Check profile was updated in database

---

## Future Enhancements

- [ ] Role inheritance (admin inherits all other permissions)
- [ ] Custom permissions beyond three roles
- [ ] Temporary elevated access (e.g., nurse becomes temp admin)
- [ ] Audit logging (track who did what with which role)
- [ ] Role-based data filtering (some admins see all hospitals)
