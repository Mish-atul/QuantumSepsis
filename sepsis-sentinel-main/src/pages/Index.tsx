import { usePatients } from "@/hooks/usePatients";
import { useLatestRiskForPatients } from "@/hooks/useRiskAssessments";
import { PatientCard } from "@/components/ward/PatientCard";
import { GlobalNav } from "@/components/layout/GlobalNav";
import { TIER_ORDER } from "@/types/database";
import type { RiskTier, RiskAssessment } from "@/types/database";
import { Shield, Activity } from "lucide-react";
import { AdmitPatientModal } from "@/components/ward/AdmitPatientModal";
import { PipelineActivityCompact } from "@/components/observability/PipelineActivityFeed";
import { useHospital } from "@/contexts/HospitalContext";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import type { TripwireAlert } from "@/types/database";

const Index = () => {
  const { data: patients, isLoading } = usePatients();
  const { connectionStatus } = useHospital();
  const patientIds = patients?.map((p) => p.id) ?? [];
  const { data: allRisks } = useLatestRiskForPatients(patientIds);

  // Fetch active tripwire counts for all patients
  const { data: allTripwires } = useQuery({
    queryKey: ["all_tripwires", patientIds],
    enabled: patientIds.length > 0,
    queryFn: async () => {
      const { data, error } = await supabase
        .from("tripwire_alerts")
        .select("*")
        .in("patient_id", patientIds)
        .eq("is_active", true);
      if (error) throw error;
      return data as TripwireAlert[];
    },
    refetchInterval: 15000,
  });

  // Group risks by patient
  const risksByPatient: Record<string, RiskAssessment[]> = {};
  const latestRiskByPatient: Record<string, RiskAssessment> = {};
  allRisks?.forEach((r) => {
    if (!risksByPatient[r.patient_id]) risksByPatient[r.patient_id] = [];
    risksByPatient[r.patient_id].push(r);
    const existing = latestRiskByPatient[r.patient_id];
    if (!existing || r.timestamp > existing.timestamp) {
      latestRiskByPatient[r.patient_id] = r;
    }
  });

  // Tripwire count by patient
  const tripwireCountByPatient: Record<string, number> = {};
  allTripwires?.forEach((t) => {
    tripwireCountByPatient[t.patient_id] = (tripwireCountByPatient[t.patient_id] || 0) + 1;
  });

  // Sort patients by tier priority
  const sortedPatients = [...(patients ?? [])].sort((a, b) => {
    const tierA = (latestRiskByPatient[a.id]?.tier as RiskTier) ?? "WATCH";
    const tierB = (latestRiskByPatient[b.id]?.tier as RiskTier) ?? "WATCH";
    const overrideA = (tripwireCountByPatient[a.id] ?? 0) >= 2;
    const overrideB = (tripwireCountByPatient[b.id] ?? 0) >= 2;
    const effectiveA = overrideA ? "CRITICAL" : tierA;
    const effectiveB = overrideB ? "CRITICAL" : tierB;
    return TIER_ORDER[effectiveA] - TIER_ORDER[effectiveB];
  });

  const criticalCount = sortedPatients.filter((p) => {
    const tier = (latestRiskByPatient[p.id]?.tier as RiskTier) ?? "WATCH";
    const override = (tripwireCountByPatient[p.id] ?? 0) >= 2;
    return tier === "CRITICAL" || override;
  }).length;

  return (
    <div className="min-h-screen bg-background">
      <GlobalNav />

      {/* Status Bar */}
      <div className="border-b border-border px-4 sm:px-6 py-2">
        <div className="flex items-center gap-4 max-w-[1600px] mx-auto flex-wrap">
          <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
            <Activity className="h-4 w-4 text-primary animate-pulse" />
            <span>{patients?.length ?? 0} Active Patients</span>
          </div>
          <AdmitPatientModal />
          {criticalCount > 0 && (
            <div className="flex items-center gap-2 px-3 py-1 rounded-sm bg-tier-critical/10 border border-tier-critical/40 animate-pulse-critical">
              <span className="text-xs font-mono font-bold text-tier-critical">
                {criticalCount} CRITICAL
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <main className="p-4 sm:p-6 max-w-[1600px] mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          {/* Patient Grid */}
          <div className="lg:col-span-3">
            {isLoading ? (
              <div className="flex items-center justify-center py-20">
                <p className="text-muted-foreground font-mono text-sm">Loading ward data...</p>
              </div>
            ) : sortedPatients.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20 space-y-3">
                <Shield className="h-12 w-12 text-muted-foreground/30" />
                <p className="text-muted-foreground font-mono text-sm">No active patients in ICU</p>
                <p className="text-muted-foreground/60 font-mono text-xs">
                  Waiting for patient data from the pipeline...
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
                {sortedPatients.map((patient) => (
                  <PatientCard
                    key={patient.id}
                    patient={patient}
                    latestRisk={latestRiskByPatient[patient.id]}
                    riskHistory={risksByPatient[patient.id] ?? []}
                    activeTripwireCount={tripwireCountByPatient[patient.id] ?? 0}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Observability: Live Pipeline Feed sidebar */}
          <div className="lg:col-span-1">
            <div className="sticky top-20 space-y-3">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="h-3.5 w-3.5 text-primary" />
                <h3 className="text-xs font-mono font-bold text-muted-foreground uppercase tracking-wider">
                  Live Pipeline Feed
                </h3>
              </div>
              <PipelineActivityCompact />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
