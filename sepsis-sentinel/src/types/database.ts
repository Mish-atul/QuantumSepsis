import type { Tables } from "@/integrations/supabase/types";

export type Patient = Tables<"patients">;
export type Vital = Tables<"vitals">;
export type Lab = Tables<"labs">;
export type RiskAssessment = Tables<"risk_assessments">;
export type TripwireAlert = Tables<"tripwire_alerts">;

export type RiskTier = "WATCH" | "AMBER" | "CRITICAL";
export type MentalStatus = "normal" | "confused" | "agitated" | "reduced_gcs";

export const TIER_ORDER: Record<RiskTier, number> = {
  CRITICAL: 0,
  AMBER: 1,
  WATCH: 2,
};

export const TIER_LABELS: Record<RiskTier, string> = {
  CRITICAL: "CRITICAL",
  AMBER: "AMBER",
  WATCH: "WATCH",
};

export const MENTAL_STATUS_LABELS: Record<MentalStatus, string> = {
  normal: "Normal",
  confused: "Confused",
  agitated: "Agitated",
  reduced_gcs: "Reduced GCS",
};

export interface TripwireThreshold {
  metric: string;
  label: string;
  condition: string;
  evaluate: (value: number) => boolean;
}

export const TRIPWIRE_THRESHOLDS: TripwireThreshold[] = [
  { metric: "Temperature", label: "Temp < 36°C", condition: "< 36°C", evaluate: (v) => v < 36 },
  { metric: "Temperature", label: "Temp > 38.3°C", condition: "> 38.3°C", evaluate: (v) => v > 38.3 },
  { metric: "Heart Rate", label: "HR > 90 bpm", condition: "> 90 bpm", evaluate: (v) => v > 90 },
  { metric: "Respiratory Rate", label: "RR > 20/min", condition: "> 20/min", evaluate: (v) => v > 20 },
  { metric: "MAP", label: "MAP < 70 mmHg", condition: "< 70 mmHg", evaluate: (v) => v < 70 },
];

export const CI_WIDTH_THRESHOLD = 0.3;
