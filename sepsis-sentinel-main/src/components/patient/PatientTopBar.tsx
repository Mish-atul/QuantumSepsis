import { TierBadge } from "@/components/shared/TierBadge";
import type { Patient, RiskTier } from "@/types/database";
import { ArrowLeft, Clock } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { format } from "date-fns";

interface PatientTopBarProps {
  patient: Patient;
  tier: RiskTier;
  isCriticalOverride: boolean;
}

export function PatientTopBar({ patient, tier, isCriticalOverride }: PatientTopBarProps) {
  const navigate = useNavigate();
  const displayTier = isCriticalOverride ? "CRITICAL" : tier;

  return (
    <div className="flex items-center gap-4 p-4 border-b border-border bg-card rounded-lg">
      <button
        onClick={() => navigate("/")}
        className="p-2 rounded-md hover:bg-accent transition-colors text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-5 w-5" />
      </button>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-mono font-bold text-foreground truncate">
            {patient.name}
          </h1>
          <TierBadge tier={displayTier} size="lg" />
          {isCriticalOverride && (
            <span className="text-xs font-mono text-tier-critical font-bold animate-pulse-critical">
              TRIPWIRE OVERRIDE
            </span>
          )}
        </div>
        <div className="flex items-center gap-4 mt-1 text-xs font-mono text-muted-foreground">
          <span>Bed {patient.bed_number}</span>
          <span>MRN: {patient.mrn}</span>
          <span className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            Admitted {format(new Date(patient.admission_time), "dd MMM yyyy HH:mm")}
          </span>
        </div>
      </div>
    </div>
  );
}
