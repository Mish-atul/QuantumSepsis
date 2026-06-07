import type { RiskAssessment, Vital, Lab, TripwireAlert } from "@/types/database";
import { RiskGauge } from "@/components/patient/RiskGauge";
import { ConfidenceInterval } from "@/components/patient/ConfidenceInterval";
import { VitalsPanel } from "@/components/patient/VitalsPanel";
import { LabsPanel } from "@/components/patient/LabsPanel";
import { TripwirePanel } from "@/components/patient/TripwirePanel";
import { VitalsChart } from "@/components/patient/VitalsChart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle } from "lucide-react";

/**
 * Generative UI: Agent data types that map to dynamic UI components.
 * The backend returns structured data, and this renderer decides
 * which component to instantiate based on the data shape.
 */
export type AgentUIBlock =
  | { type: "risk_gauge"; data: RiskAssessment }
  | { type: "confidence_interval"; data: RiskAssessment }
  | { type: "vitals_panel"; data: Vital }
  | { type: "vitals_chart"; data: Vital[] }
  | { type: "labs_panel"; data: Lab }
  | { type: "tripwire_panel"; data: { alerts: TripwireAlert[]; latestVital?: Vital } }
  | { type: "alert"; data: { title: string; message: string; severity: "info" | "warning" | "critical" } }
  | { type: "key_value"; data: { title: string; entries: { label: string; value: string }[] } };

interface AgentDataRendererProps {
  blocks: AgentUIBlock[];
}

/**
 * Generative UI: Dynamically renders UI components based on structured agent data.
 * Instead of agents returning raw text, they return typed blocks that map to rich components.
 */
export function AgentDataRenderer({ blocks }: AgentDataRendererProps) {
  return (
    <div className="space-y-4">
      {blocks.map((block, i) => (
        <AgentBlock key={i} block={block} />
      ))}
    </div>
  );
}

function AgentBlock({ block }: { block: AgentUIBlock }) {
  switch (block.type) {
    case "risk_gauge":
      return <RiskGauge assessment={block.data} />;

    case "confidence_interval":
      return <ConfidenceInterval assessment={block.data} />;

    case "vitals_panel":
      return <VitalsPanel latestVital={block.data} />;

    case "vitals_chart":
      return <VitalsChart vitals={block.data} />;

    case "labs_panel":
      return <LabsPanel lab={block.data} />;

    case "tripwire_panel":
      return <TripwirePanel alerts={block.data.alerts} latestVital={block.data.latestVital} />;

    case "alert": {
      const severityStyles = {
        info: "border-primary/30 bg-primary/5 text-primary",
        warning: "border-tier-amber/40 bg-tier-amber/5 text-tier-amber",
        critical: "border-tier-critical/40 bg-tier-critical/5 text-tier-critical",
      };
      return (
        <Card className={severityStyles[block.data.severity]}>
          <CardContent className="flex items-start gap-3 p-4">
            <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
            <div>
              <p className="text-xs font-mono font-bold">{block.data.title}</p>
              <p className="text-xs font-mono opacity-80 mt-0.5">{block.data.message}</p>
            </div>
          </CardContent>
        </Card>
      );
    }

    case "key_value":
      return (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-mono uppercase tracking-wider text-muted-foreground">
              {block.data.title}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1.5">
              {block.data.entries.map((entry, i) => (
                <div key={i} className="flex items-center justify-between py-1 border-b border-border last:border-0">
                  <span className="text-xs font-mono text-muted-foreground">{entry.label}</span>
                  <span className="text-xs font-mono font-bold text-foreground">{entry.value}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      );

    default:
      return null;
  }
}
