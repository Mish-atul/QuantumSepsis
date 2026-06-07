import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import type { Vital } from "@/types/database";

export function usePatientVitals(patientId: string | undefined) {
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: ["vitals", patientId],
    enabled: !!patientId,
    queryFn: async () => {
      const sixHoursAgo = new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString();
      const { data, error } = await supabase
        .from("vitals")
        .select("*")
        .eq("patient_id", patientId!)
        .gte("timestamp", sixHoursAgo)
        .order("timestamp", { ascending: true });
      if (error) throw error;
      return data as Vital[];
    },
    refetchInterval: 30000,
  });

  useEffect(() => {
    if (!patientId) return;
    const channel = supabase
      .channel(`vitals-${patientId}`)
      .on("postgres_changes", {
        event: "INSERT",
        schema: "public",
        table: "vitals",
        filter: `patient_id=eq.${patientId}`,
      }, () => {
        queryClient.invalidateQueries({ queryKey: ["vitals", patientId] });
      })
      .subscribe();

    return () => { supabase.removeChannel(channel); };
  }, [patientId, queryClient]);

  return query;
}
