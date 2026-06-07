import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import type { Patient } from "@/types/database";

export function usePatients() {
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: ["patients"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("patients")
        .select("*")
        .eq("status", "active")
        .order("name");
      if (error) throw error;
      return data as Patient[];
    },
  });

  useEffect(() => {
    const channel = supabase
      .channel("patients-realtime")
      .on("postgres_changes", { event: "*", schema: "public", table: "patients" }, () => {
        queryClient.invalidateQueries({ queryKey: ["patients"] });
      })
      .subscribe();

    return () => { supabase.removeChannel(channel); };
  }, [queryClient]);

  return query;
}
