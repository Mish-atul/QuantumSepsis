import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { RiskAssessment } from "@/types/database";
import { CI_WIDTH_THRESHOLD } from "@/types/database";
import { AlertTriangle } from "lucide-react";

interface ConfidenceIntervalProps {
  assessment: RiskAssessment | undefined;
}

export function ConfidenceInterval({ assessment }: ConfidenceIntervalProps) {
  if (!assessment) {
    return null;
  }

  const lower = assessment.confidence_interval_lower;
  const upper = assessment.confidence_interval_upper;
  const width = upper - lower;
  const isWide = width > CI_WIDTH_THRESHOLD;
  const score = assessment.quantum_risk_score;

  // Position calculations (0-100% scale)
  const lowerPct = lower * 100;
  const upperPct = upper * 100;
  const scorePct = score * 100;

  return (
    <Card className={cn(isWide && "animate-pulse-amber border-ci-wide/40")}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-mono uppercase tracking-wider text-muted-foreground">
            Confidence Interval — Layer 4a
          </CardTitle>
          {isWide && (
            <div className="flex items-center gap-1.5 text-ci-wide">
              <AlertTriangle className="h-3.5 w-3.5" />
              <span className="text-[10px] font-mono font-bold uppercase">High Uncertainty</span>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="relative h-8 rounded-sm bg-secondary overflow-hidden">
          {/* CI Range bar */}
          <div
            className={cn(
              "absolute top-0 bottom-0 rounded-sm",
              isWide ? "bg-ci-wide/30" : "bg-ci-normal/20"
            )}
            style={{ left: `${lowerPct}%`, width: `${upperPct - lowerPct}%` }}
          />
          {/* Score marker */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-foreground"
            style={{ left: `${scorePct}%` }}
          />
        </div>

        <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
          <span>Lower: {lower.toFixed(3)}</span>
          <span className={cn("font-bold", isWide ? "text-ci-wide" : "text-foreground")}>
            Width: {width.toFixed(3)}
          </span>
          <span>Upper: {upper.toFixed(3)}</span>
        </div>

        {isWide && (
          <div className="p-2 rounded-sm border border-ci-wide/40 bg-ci-wide/5">
            <p className="text-[10px] font-mono text-ci-wide">
              ⚠ Interpret with caution — Confidence interval width ({width.toFixed(3)}) exceeds threshold ({CI_WIDTH_THRESHOLD}).
              The model has high uncertainty for this prediction.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
