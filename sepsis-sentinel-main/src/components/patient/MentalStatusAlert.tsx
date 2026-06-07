import { cn } from "@/lib/utils";
import type { MentalStatus } from "@/types/database";
import { MENTAL_STATUS_LABELS } from "@/types/database";
import { Brain, AlertTriangle } from "lucide-react";

interface MentalStatusAlertProps {
  status: MentalStatus;
}

export function MentalStatusAlert({ status }: MentalStatusAlertProps) {
  if (status === "normal") return null;

  const isReduced = status === "reduced_gcs";

  return (
    <div
      className={cn(
        "flex items-center gap-3 p-3 rounded-md border-2",
        isReduced
          ? "border-tier-critical/60 bg-tier-critical/10 animate-pulse-critical"
          : "border-tier-amber/60 bg-tier-amber/10"
      )}
    >
      <div className={cn(
        "p-2 rounded-md",
        isReduced ? "bg-tier-critical/20" : "bg-tier-amber/20"
      )}>
        <Brain className={cn(
          "h-5 w-5",
          isReduced ? "text-tier-critical" : "text-tier-amber"
        )} />
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <AlertTriangle className={cn(
            "h-3.5 w-3.5",
            isReduced ? "text-tier-critical" : "text-tier-amber"
          )} />
          <span className={cn(
            "text-xs font-mono font-bold uppercase tracking-wider",
            isReduced ? "text-tier-critical" : "text-tier-amber"
          )}>
            Nurse-Flagged Altered Status
          </span>
        </div>
        <p className="text-xs font-mono text-muted-foreground mt-1">
          Assessment: <span className="font-semibold text-foreground">{MENTAL_STATUS_LABELS[status]}</span>
          {" — "}Clinical assessment often catches deterioration before numeric indicators
        </p>
      </div>
    </div>
  );
}
