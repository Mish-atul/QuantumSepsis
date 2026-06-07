import { LineChart, Line, ResponsiveContainer, YAxis } from "recharts";
import type { RiskAssessment, RiskTier } from "@/types/database";

interface RiskSparklineProps {
  assessments: RiskAssessment[];
  tier: RiskTier;
}

const TIER_COLORS: Record<RiskTier, string> = {
  CRITICAL: "hsl(0, 84%, 60%)",
  AMBER: "hsl(38, 92%, 50%)",
  WATCH: "hsl(142, 71%, 45%)",
};

export function RiskSparkline({ assessments, tier }: RiskSparklineProps) {
  const data = assessments.map((a) => ({ score: a.quantum_risk_score }));

  if (data.length === 0) {
    return <div className="h-10 w-full flex items-center justify-center text-muted-foreground text-[10px]">No data</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={40}>
      <LineChart data={data}>
        <YAxis domain={[0, 1]} hide />
        <Line
          type="monotone"
          dataKey="score"
          stroke={TIER_COLORS[tier]}
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
