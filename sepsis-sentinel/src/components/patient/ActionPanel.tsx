import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { RiskTier } from "@/types/database";
import { Clock, Syringe, Phone } from "lucide-react";

interface ActionPanelProps {
  tier: RiskTier;
  isCriticalOverride: boolean;
}

const ACTIONS: Record<RiskTier, { icon: React.ReactNode; label: string; description: string }> = {
  WATCH: {
    icon: <Clock className="h-4 w-4" />,
    label: "MONITORING",
    description: "Standard monitoring — next cycle in 15 min",
  },
  AMBER: {
    icon: <Syringe className="h-4 w-4" />,
    label: "CONCURRENT ORDERS SENT",
    description: "Lactate, PCT, Blood Culture ordered. Fluid challenge initiated.",
  },
  CRITICAL: {
    icon: <Phone className="h-4 w-4" />,
    label: "SEPSIS BUNDLE INITIATED",
    description: "Attending paged. Broad-spectrum antibiotics within 1hr. Fluid resuscitation started. ICU team alerted.",
  },
};

export function ActionPanel({ tier, isCriticalOverride }: ActionPanelProps) {
  const effectiveTier = isCriticalOverride ? "CRITICAL" : tier;
  const action = ACTIONS[effectiveTier];

  const borderClass = {
    CRITICAL: "border-tier-critical/40 bg-tier-critical/5",
    AMBER: "border-tier-amber/40 bg-tier-amber/5",
    WATCH: "border-border",
  };

  const textClass = {
    CRITICAL: "text-tier-critical",
    AMBER: "text-tier-amber",
    WATCH: "text-tier-watch",
  };

  return (
    <Card className={cn(borderClass[effectiveTier])}>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider text-muted-foreground">
          Orchestrator Actions — Layer 5
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-start gap-3">
          <div className={cn("p-2 rounded-md bg-secondary", textClass[effectiveTier])}>
            {action.icon}
          </div>
          <div>
            <p className={cn("text-xs font-mono font-bold uppercase tracking-wider", textClass[effectiveTier])}>
              {action.label}
            </p>
            <p className="text-xs font-mono text-muted-foreground mt-1">
              {action.description}
            </p>
            {isCriticalOverride && effectiveTier === "CRITICAL" && (
              <p className="text-[10px] font-mono text-tier-critical mt-2 font-bold">
                ↑ Escalated via tripwire override (2+ active triggers)
              </p>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
