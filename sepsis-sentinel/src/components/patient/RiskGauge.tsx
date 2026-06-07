import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { RiskAssessment, RiskTier } from "@/types/database";
import { cn } from "@/lib/utils";

interface RiskGaugeProps {
  assessment: RiskAssessment | undefined;
}

export function RiskGauge({ assessment }: RiskGaugeProps) {
  const score = assessment?.quantum_risk_score ?? 0;
  const tier = (assessment?.tier as RiskTier) ?? "WATCH";

  const tierColor = {
    CRITICAL: "hsl(0, 84%, 60%)",
    AMBER: "hsl(38, 92%, 50%)",
    WATCH: "hsl(142, 71%, 45%)",
  };

  // SVG gauge: semi-circle
  const radius = 45;
  const circumference = Math.PI * radius;
  const offset = circumference - (score * circumference);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider text-muted-foreground">
          Quantum Risk Score — Layers 3 & 5
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center">
        <div className="relative w-40 h-24">
          <svg viewBox="0 0 100 55" className="w-full h-full">
            {/* Background arc */}
            <path
              d="M 5 50 A 45 45 0 0 1 95 50"
              fill="none"
              stroke="hsl(220, 13%, 18%)"
              strokeWidth="6"
              strokeLinecap="round"
            />
            {/* Score arc */}
            <path
              d="M 5 50 A 45 45 0 0 1 95 50"
              fill="none"
              stroke={tierColor[tier]}
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={`${circumference}`}
              strokeDashoffset={offset}
              className="animate-gauge-fill"
              style={{ transition: "stroke-dashoffset 1s ease-out" }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-end pb-1">
            <span
              className="text-3xl font-mono font-black"
              style={{ color: tierColor[tier] }}
            >
              {score.toFixed(2)}
            </span>
          </div>
        </div>
        <p className="text-[10px] font-mono text-muted-foreground mt-1">
          Score Range: 0.00 (Safe) — 1.00 (Septic)
        </p>
      </CardContent>
    </Card>
  );
}
