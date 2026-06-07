import { useParams } from "react-router-dom";
import { GlobalNav } from "@/components/layout/GlobalNav";
import { usePatients } from "@/hooks/usePatients";
import { usePatientVitals } from "@/hooks/usePatientVitals";
import { useRiskAssessments } from "@/hooks/useRiskAssessments";
import { useTripwireAlerts } from "@/hooks/useTripwireAlerts";
import { usePatientLabs } from "@/hooks/usePatientLabs";
import { PatientTopBar } from "@/components/patient/PatientTopBar";
import { VitalsPanel } from "@/components/patient/VitalsPanel";
import { VitalsChart } from "@/components/patient/VitalsChart";
import { LabsPanel } from "@/components/patient/LabsPanel";
import { RiskGauge } from "@/components/patient/RiskGauge";
import { ConfidenceInterval } from "@/components/patient/ConfidenceInterval";
import { HITLActionPanel } from "@/components/patient/HITLActionPanel";
import { TripwirePanel } from "@/components/patient/TripwirePanel";
import { LogVitalsDrawer } from "@/components/patient/LogVitalsDrawer";
import { LogLabsDrawer } from "@/components/patient/LogLabsDrawer";
import { DischargePatientDialog } from "@/components/patient/DischargePatientDialog";
import type { RiskTier, MentalStatus } from "@/types/database";

const PatientDetail = () => {
  const { id } = useParams<{ id: string }>();
  const { data: patients } = usePatients();
  const { data: vitals } = usePatientVitals(id);
  const { data: assessments } = useRiskAssessments(id);
  const { data: alerts } = useTripwireAlerts(id);
  const { data: lab } = usePatientLabs(id);

  const patient = patients?.find((p) => p.id === id);
  const latestVital = vitals?.[vitals.length - 1];
  const latestRisk = assessments?.[assessments.length - 1];
  const tier = (latestRisk?.tier as RiskTier) ?? "WATCH";

  // Calculate tripwire count including mental status
  const mentalStatus = (latestVital?.mental_status as MentalStatus) ?? "normal";
  const isAlteredMental = mentalStatus !== "normal";
  const totalTripwires = (alerts?.length ?? 0) + (isAlteredMental ? 1 : 0);
  const isCriticalOverride = totalTripwires >= 2;

  if (!patient) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-muted-foreground font-mono">Loading patient data...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <GlobalNav />
      <div className="p-4 space-y-4 max-w-[1600px] mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <PatientTopBar patient={patient} tier={tier} isCriticalOverride={isCriticalOverride} />
        <div className="flex items-center gap-2 shrink-0">
          {/* HITL: Manual data entry points */}
          <LogVitalsDrawer patientId={patient.id} />
          <LogLabsDrawer patientId={patient.id} />
          <DischargePatientDialog patientId={patient.id} patientName={patient.name} />
        </div>
      </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Left Column: Vitals & Labs (Generative UI: data → components) */}
          <div className="space-y-4">
            <VitalsPanel latestVital={latestVital} />
            <VitalsChart vitals={vitals ?? []} />
            <LabsPanel lab={lab} />
          </div>

          {/* Right Column: Risk + Tripwires + HITL Actions */}
          <div className="space-y-4">
            <RiskGauge assessment={latestRisk} />
            <ConfidenceInterval assessment={latestRisk} />
            {/* HITL: Replaces static ActionPanel with approval-required version */}
            <HITLActionPanel
              tier={tier}
              isCriticalOverride={isCriticalOverride}
              patientName={patient.name}
            />
            <TripwirePanel alerts={alerts ?? []} latestVital={latestVital} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default PatientDetail;
