# Hackathon Shortcuts for QuantumSepsis Shield

## 💡 Quick Win: Save Predictions from Frontend (Not Backend)

### Why Frontend-First?
- ✅ Supabase client already initialized
- ✅ User context already available
- ✅ No SSH, no deployment, no server config
- ✅ Takes 2 minutes vs 2 hours

### Implementation Example

Add this to any component that calls the Python `/predict` API:

```typescript
// Example: In src/pages/DemoSimulator.tsx or PatientDetail.tsx

import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";

const DemoSimulator = () => {
  const { profile } = useAuth();
  const [patientId, setPatientId] = useState<string>("");

  const handlePredict = async () => {
    try {
      // 1. Call Python API (AWS EC2)
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          heart_rate: 95,
          map: 82,
          temperature: 38.2,
          resp_rate: 22,
          spo2: 94,
          gcs_total: 14,
          lactate: 2.3,
          wbc: 15.2,
          creatinine: 1.4,
          platelets: 180,
          age: 67,
          gender: "M"
        })
      });

      const result = await response.json();

      // 2. ✅ Save to Supabase immediately (frontend does this!)
      if (result && patientId && profile?.hospital_id) {
        const { data, error } = await supabase
          .from("risk_assessments")
          .insert({
            patient_id: patientId,
            stay_id: `ICU_${Date.now()}`, // Generate or use existing
            quantum_risk_score: result.risk_score,
            lstm_risk_score: result.lstm_score,
            xgboost_risk_score: result.xgb_score,
            tier: result.alert_level, // WATCH/AMBER/CRITICAL/FAST-TRACK
            confidence_score: result.confidence,
            confidence_interval_lower: result.conformal_interval[0],
            confidence_interval_upper: result.conformal_interval[1],
            fast_tracked: result.fast_tracked,
            reasoning: result.reasoning,
            hospital_id: profile.hospital_id,
            assessed_at: new Date().toISOString(),
          });

        if (error) {
          console.error("Failed to save risk assessment:", error);
          toast.error("Prediction succeeded but failed to save to database");
        } else {
          toast.success(`Alert: ${result.alert_level} - Risk: ${(result.risk_score * 100).toFixed(1)}%`);
        }
      }

      // 3. ✅ Optionally save tripwire alerts too
      if (result.tripwires && result.tripwires.length > 0) {
        const tripwireInserts = result.tripwires
          .filter(t => t.triggered)
          .map(t => ({
            patient_id: patientId,
            tripwire_code: t.name, // TW-TEMP, TW-HR, etc.
            severity: result.alert_level,
            value: t.value,
            threshold: t.threshold,
            reason: t.reason,
            hospital_id: profile.hospital_id,
            triggered_at: new Date().toISOString(),
          }));

        await supabase.from("tripwire_alerts").insert(tripwireInserts);
      }

    } catch (error) {
      console.error("Prediction error:", error);
      toast.error("Failed to get prediction from API");
    }
  };

  return (
    <div>
      <Button onClick={handlePredict}>Run Prediction</Button>
    </div>
  );
};
```

### Alternative: Use React Query Mutation

For better UX with loading states and error handling:

```typescript
import { useMutation, useQueryClient } from "@tanstack/react-query";

const useSavePrediction = () => {
  const { profile } = useAuth();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ patientId, prediction }: { patientId: string; prediction: any }) => {
      const { data, error } = await supabase
        .from("risk_assessments")
        .insert({
          patient_id: patientId,
          quantum_risk_score: prediction.risk_score,
          tier: prediction.alert_level,
          confidence_interval_lower: prediction.conformal_interval[0],
          confidence_interval_upper: prediction.conformal_interval[1],
          hospital_id: profile?.hospital_id,
          assessed_at: new Date().toISOString(),
        })
        .select()
        .single();

      if (error) throw error;
      return data;
    },
    onSuccess: () => {
      // Invalidate and refetch patient data
      queryClient.invalidateQueries({ queryKey: ["risk_assessments"] });
      toast.success("Risk assessment saved");
    },
    onError: (error) => {
      console.error(error);
      toast.error("Failed to save risk assessment");
    },
  });
};

// Usage:
const { mutate: savePrediction, isPending } = useSavePrediction();

const handlePredict = async () => {
  const result = await fetch('http://localhost:8000/predict', { ... });
  const prediction = await result.json();
  
  savePrediction({ patientId, prediction });
};
```

## 🎯 Benefits

| Approach | Time | Complexity | Dependencies |
|----------|------|------------|--------------|
| **Backend Integration** | 2-4 hours | High | SSH, supabase-py, env vars, restart |
| **Frontend Integration** ✅ | 2 minutes | Low | Already have everything |

## 🚀 Deployment Note

For production, you'd want the backend to write directly. But for a hackathon demo:
- Frontend saves = Working demo in 2 minutes
- Backend integration = Can add later without breaking anything

The database schema is the same either way!
