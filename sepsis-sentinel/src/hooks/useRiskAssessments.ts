import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import type { RiskAssessment } from "@/types/database";

export function useRiskAssessments(patientId: string | undefined) {
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: ["risk_assessments", patientId],
    enabled: !!patientId,
    queryFn: async () => {
      const sixHoursAgo = new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString();
      const { data, error } = await supabase
        .from("risk_assessments")
        .select("*")
        .eq("patient_id", patientId!)
        .gte("timestamp", sixHoursAgo)
        .order("timestamp", { ascending: true });
      if (error) throw error;
      return data as RiskAssessment[];
    },
    refetchInterval: 30000,
  });

  useEffect(() => {
    if (!patientId) return;
    const channel = supabase
      .channel(`risk-${patientId}`)
      .on("postgres_changes", {
        event: "INSERT",
        schema: "public",
        table: "risk_assessments",
        filter: `patient_id=eq.${patientId}`,
      }, () => {
        queryClient.invalidateQueries({ queryKey: ["risk_assessments", patientId] });
      })
      .subscribe();

    return () => { supabase.removeChannel(channel); };
  }, [patientId, queryClient]);

  return query;
}

export function useLatestRiskForPatients(patientIds: string[]) {
  return useQuery({
    queryKey: ["latest_risks", patientIds],
    enabled: patientIds.length > 0,
    queryFn: async () => {
      const sixHoursAgo = new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString();
      const { data, error } = await supabase
        .from("risk_assessments")
        .select("*")
        .in("patient_id", patientIds)
        .gte("timestamp", sixHoursAgo)
        .order("timestamp", { ascending: true });
      if (error) throw error;
      return data as RiskAssessment[];
    },
    refetchInterval: 30000,
  });
}
