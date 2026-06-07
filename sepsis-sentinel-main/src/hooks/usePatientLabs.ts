import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import type { Lab } from "@/types/database";

export function usePatientLabs(patientId: string | undefined) {
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: ["labs", patientId],
    enabled: !!patientId,
    queryFn: async () => {
      const { data, error } = await supabase
        .from("labs")
        .select("*")
        .eq("patient_id", patientId!)
        .order("timestamp", { ascending: false })
        .limit(1);
      if (error) throw error;
      return data?.[0] as Lab | undefined;
    },
    refetchInterval: 60000,
  });

  // Bidirectional: Realtime subscription for labs
  useEffect(() => {
    if (!patientId) return;
    const channel = supabase
      .channel(`labs-${patientId}`)
      .on("postgres_changes", {
        event: "INSERT",
        schema: "public",
        table: "labs",
        filter: `patient_id=eq.${patientId}`,
      }, () => {
        queryClient.invalidateQueries({ queryKey: ["labs", patientId] });
      })
      .subscribe();

    return () => { supabase.removeChannel(channel); };
  }, [patientId, queryClient]);

  return query;
}
