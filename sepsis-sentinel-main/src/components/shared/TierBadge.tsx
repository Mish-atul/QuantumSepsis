import { cn } from "@/lib/utils";
import type { RiskTier } from "@/types/database";

interface TierBadgeProps {
  tier: RiskTier;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function TierBadge({ tier, size = "md", className }: TierBadgeProps) {
  const sizeClasses = {
    sm: "px-2 py-0.5 text-[10px]",
    md: "px-3 py-1 text-xs",
    lg: "px-4 py-1.5 text-sm",
  };

  const tierClasses = {
    CRITICAL: "bg-tier-critical/20 text-tier-critical border-tier-critical/50 animate-pulse-critical",
    AMBER: "bg-tier-amber/20 text-tier-amber border-tier-amber/50",
    WATCH: "bg-tier-watch/20 text-tier-watch border-tier-watch/50",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-sm border font-mono font-bold uppercase tracking-wider",
        sizeClasses[size],
        tierClasses[tier],
        className
      )}
    >
      {tier}
    </span>
  );
}
