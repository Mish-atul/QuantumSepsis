import { useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { TierBadge } from "@/components/shared/TierBadge";
import { RiskSparkline } from "./RiskSparkline";
import { cn } from "@/lib/utils";
import type { Patient, RiskAssessment, RiskTier } from "@/types/database";
import { Activity } from "lucide-react";

interface PatientCardProps {
  patient: Patient;
  latestRisk?: RiskAssessment;
  riskHistory: RiskAssessment[];
  activeTripwireCount: number;
}

export function PatientCard({ patient, latestRisk, riskHistory, activeTripwireCount }: PatientCardProps) {
  const navigate = useNavigate();
  const tier = (latestRisk?.tier as RiskTier) || "WATCH";
  const score = latestRisk?.quantum_risk_score;
  const isCriticalOverride = activeTripwireCount >= 2;

  const borderClass = {
    CRITICAL: "border-tier-critical/40",
    AMBER: "border-tier-amber/30",
    WATCH: "border-border",
  };

  return (
    <Card
      onClick={() => navigate(`/patient/${patient.id}`)}
      className={cn(
        "cursor-pointer transition-all hover:scale-[1.02] hover:shadow-lg",
        borderClass[isCriticalOverride ? "CRITICAL" : tier],
        isCriticalOverride && "animate-flash-critical"
      )}
    >
      <CardContent className="p-4 space-y-3">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="font-mono font-semibold text-foreground text-sm">{patient.name}</h3>
            <p className="text-muted-foreground text-xs font-mono">
              Bed {patient.bed_number} · {patient.mrn}
            </p>
          </div>
          <TierBadge tier={isCriticalOverride ? "CRITICAL" : tier} size="sm" />
        </div>

        <div className="flex items-center gap-3">
          <Activity className="h-3.5 w-3.5 text-muted-foreground" />
          <div className="flex-1">
            <RiskSparkline assessments={riskHistory} tier={tier} />
          </div>
          {score !== undefined && (
            <span className="font-mono text-xs font-bold text-foreground">
              {score.toFixed(2)}
            </span>
          )}
        </div>

        {isCriticalOverride && (
          <div className="text-[10px] font-mono text-tier-critical font-bold uppercase tracking-wider">
            ⚠ {activeTripwireCount} TRIPWIRES ACTIVE — CRITICAL OVERRIDE
          </div>
        )}
      </CardContent>
    </Card>
  );
}
