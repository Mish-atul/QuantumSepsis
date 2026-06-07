import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { TripwireAlert, Vital, MentalStatus } from "@/types/database";
import { AlertTriangle, Brain, Zap } from "lucide-react";

interface TripwirePanelProps {
  alerts: TripwireAlert[];
  latestVital: Vital | undefined;
}

export function TripwirePanel({ alerts, latestVital }: TripwirePanelProps) {
  const mentalStatus = (latestVital?.mental_status as MentalStatus) ?? "normal";
  const isAlteredMental = mentalStatus !== "normal";

  // Mental status is always first and most prominent
  const allAlerts: { metric: string; value: string; threshold: string; isMental: boolean }[] = [];

  if (isAlteredMental) {
    allAlerts.push({
      metric: "Mental Status",
      value: mentalStatus,
      threshold: "Nurse-flagged altered status",
      isMental: true,
    });
  }

  alerts.forEach((a) => {
    allAlerts.push({
      metric: a.metric,
      value: String(a.value),
      threshold: a.threshold_breached,
      isMental: false,
    });
  });

  const totalActive = allAlerts.length;
  const isCriticalOverride = totalActive >= 2;

  return (
    <Card className={cn(isCriticalOverride && "border-tier-critical/50")}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-mono uppercase tracking-wider text-muted-foreground">
            Tripwire Alerts — Layer 4b (Red Team)
          </CardTitle>
          {totalActive > 0 && (
            <span className={cn(
              "text-xs font-mono font-bold px-2 py-0.5 rounded-sm",
              isCriticalOverride
                ? "bg-tier-critical/20 text-tier-critical"
                : "bg-tier-amber/20 text-tier-amber"
            )}>
              {totalActive} ACTIVE
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        {isCriticalOverride && (
          <div className="p-3 rounded-md border-2 border-tier-critical/60 bg-tier-critical/10 animate-flash-critical">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-tier-critical" />
              <span className="text-xs font-mono font-black text-tier-critical uppercase tracking-wider">
                CRITICAL OVERRIDE — {totalActive} tripwires breached
              </span>
            </div>
            <p className="text-[10px] font-mono text-tier-critical/80 mt-1">
              Automatic escalation to CRITICAL tier. Sepsis bundle protocol initiated.
            </p>
          </div>
        )}

        {allAlerts.length === 0 ? (
          <div className="flex items-center gap-2 p-3 rounded-md border border-tier-watch/30 bg-tier-watch/5">
            <span className="text-xs font-mono text-tier-watch">No active tripwires</span>
          </div>
        ) : (
          <div className="space-y-1.5">
            {allAlerts.map((alert, i) => (
              <div
                key={i}
                className={cn(
                  "flex items-center gap-3 p-2.5 rounded-md border",
                  alert.isMental
                    ? "border-tier-amber/60 bg-tier-amber/10"
                    : "border-vital-danger/30 bg-vital-danger/5"
                )}
              >
                {alert.isMental ? (
                  <Brain className="h-4 w-4 text-tier-amber shrink-0" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-vital-danger shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <span className={cn(
                    "text-xs font-mono font-bold",
                    alert.isMental ? "text-tier-amber" : "text-vital-danger"
                  )}>
                    {alert.metric}
                  </span>
                  {alert.isMental && (
                    <span className="text-[10px] font-mono text-tier-amber ml-2">
                      (NURSE ASSESSMENT — PRIMARY TRIGGER)
                    </span>
                  )}
                </div>
                <div className="text-right shrink-0">
                  <span className="text-xs font-mono font-bold text-foreground">{alert.value}</span>
                  <p className="text-[10px] font-mono text-muted-foreground">{alert.threshold}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Hardcoded threshold reference */}
        <div className="mt-3 pt-3 border-t border-border">
          <p className="text-[10px] font-mono text-muted-foreground mb-1.5 uppercase tracking-wider">Threshold Reference</p>
          <div className="grid grid-cols-2 gap-1 text-[10px] font-mono text-muted-foreground">
            <span>Temp: &lt;36°C or &gt;38.3°C</span>
            <span>HR: &gt;90 bpm + trend</span>
            <span>RR: &gt;20/min</span>
            <span>MAP: &lt;70 mmHg</span>
            <span className="col-span-2">Altered Mental Status (nurse-flagged)</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
