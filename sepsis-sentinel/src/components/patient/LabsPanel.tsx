import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Lab } from "@/types/database";
import { format } from "date-fns";

interface LabsPanelProps {
  lab: Lab | undefined;
}

interface LabValueProps {
  label: string;
  value: number | null;
  unit: string;
  high?: number;
}

function LabValue({ label, value, unit, high }: LabValueProps) {
  const isHigh = high !== undefined && value !== null && value > high;
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-border last:border-0">
      <span className="text-xs font-mono text-muted-foreground">{label}</span>
      <span className={`text-xs font-mono font-bold ${isHigh ? "text-vital-danger" : "text-foreground"}`}>
        {value !== null ? `${value} ${unit}` : "—"}
      </span>
    </div>
  );
}

export function LabsPanel({ lab }: LabsPanelProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider text-muted-foreground">
          Latest Labs
        </CardTitle>
        {lab && (
          <p className="text-[10px] font-mono text-muted-foreground">
            {format(new Date(lab.timestamp), "dd MMM HH:mm")}
          </p>
        )}
      </CardHeader>
      <CardContent>
        {!lab ? (
          <p className="text-muted-foreground text-xs">No lab data</p>
        ) : (
          <div className="space-y-0">
            <LabValue label="Lactate" value={lab.lactate} unit="mmol/L" high={2} />
            <LabValue label="WBC" value={lab.wbc} unit="×10³/µL" high={12} />
            <LabValue label="Procalcitonin" value={lab.procalcitonin} unit="ng/mL" high={0.5} />
            <LabValue label="Creatinine" value={lab.creatinine} unit="mg/dL" high={1.2} />
            <LabValue label="Bilirubin" value={lab.bilirubin} unit="mg/dL" high={1.2} />
            <LabValue label="Platelets" value={lab.platelets} unit="×10³/µL" />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
