import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { Vital, MentalStatus } from "@/types/database";
import { MENTAL_STATUS_LABELS } from "@/types/database";
import { Heart, Thermometer, Wind, Droplets, Brain, Activity } from "lucide-react";
import { MentalStatusAlert } from "./MentalStatusAlert";

interface VitalsPanelProps {
  latestVital: Vital | undefined;
}

interface VitalCardProps {
  label: string;
  value: string | number | null | undefined;
  unit: string;
  icon: React.ReactNode;
  danger?: boolean;
  warning?: boolean;
}

function VitalCard({ label, value, unit, icon, danger, warning }: VitalCardProps) {
  return (
    <div className={cn(
      "flex items-center gap-3 p-3 rounded-md border",
      danger ? "border-vital-danger/40 bg-vital-danger/5" :
      warning ? "border-vital-warning/40 bg-vital-warning/5" :
      "border-border bg-card"
    )}>
      <div className={cn(
        "text-muted-foreground",
        danger && "text-vital-danger",
        warning && "text-vital-warning",
      )}>
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider">{label}</p>
        <p className={cn(
          "text-lg font-mono font-bold",
          danger ? "text-vital-danger" : warning ? "text-vital-warning" : "text-foreground"
        )}>
          {value ?? "—"} <span className="text-xs font-normal text-muted-foreground">{unit}</span>
        </p>
      </div>
    </div>
  );
}

export function VitalsPanel({ latestVital }: VitalsPanelProps) {
  if (!latestVital) {
    return (
      <Card>
        <CardHeader><CardTitle className="text-sm font-mono">Vitals</CardTitle></CardHeader>
        <CardContent><p className="text-muted-foreground text-xs">No vitals data available</p></CardContent>
      </Card>
    );
  }

  const mentalStatus = latestVital.mental_status as MentalStatus;
  const isAlteredMental = mentalStatus !== "normal";

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-mono uppercase tracking-wider text-muted-foreground">
          Real-time Vitals — Layer 1
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {isAlteredMental && (
          <MentalStatusAlert status={mentalStatus} />
        )}

        <div className="grid grid-cols-2 gap-2">
          <VitalCard
            label="Heart Rate"
            value={latestVital.heart_rate}
            unit="bpm"
            icon={<Heart className="h-4 w-4" />}
            warning={latestVital.heart_rate !== null && latestVital.heart_rate > 90}
            danger={latestVital.heart_rate !== null && latestVital.heart_rate > 120}
          />
          <VitalCard
            label="Blood Pressure"
            value={latestVital.blood_pressure_sys && latestVital.blood_pressure_dia
              ? `${latestVital.blood_pressure_sys}/${latestVital.blood_pressure_dia}`
              : null}
            unit="mmHg"
            icon={<Activity className="h-4 w-4" />}
          />
          <VitalCard
            label="MAP"
            value={latestVital.map}
            unit="mmHg"
            icon={<Droplets className="h-4 w-4" />}
            danger={latestVital.map !== null && latestVital.map < 70}
          />
          <VitalCard
            label="SpO₂"
            value={latestVital.spo2}
            unit="%"
            icon={<Droplets className="h-4 w-4" />}
            warning={latestVital.spo2 !== null && latestVital.spo2 < 95}
            danger={latestVital.spo2 !== null && latestVital.spo2 < 90}
          />
          <VitalCard
            label="Temperature"
            value={latestVital.temperature}
            unit="°C"
            icon={<Thermometer className="h-4 w-4" />}
            warning={latestVital.temperature !== null && (latestVital.temperature < 36 || latestVital.temperature > 38.3)}
          />
          <VitalCard
            label="Resp. Rate"
            value={latestVital.respiratory_rate}
            unit="/min"
            icon={<Wind className="h-4 w-4" />}
            warning={latestVital.respiratory_rate !== null && latestVital.respiratory_rate > 20}
            danger={latestVital.respiratory_rate !== null && latestVital.respiratory_rate > 30}
          />
        </div>

        {!isAlteredMental && (
          <div className="flex items-center gap-2 p-2 rounded-md border border-border bg-card">
            <Brain className="h-4 w-4 text-tier-watch" />
            <span className="text-xs font-mono text-muted-foreground">
              Mental Status: <span className="text-tier-watch font-semibold">{MENTAL_STATUS_LABELS[mentalStatus]}</span>
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
