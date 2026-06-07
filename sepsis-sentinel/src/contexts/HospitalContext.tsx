import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from "react";
import { useAuth } from "@/hooks/useAuth";
import { useRealtimeConnection, type ConnectionStatus } from "@/hooks/useRealtimeConnection";
import { supabase } from "@/integrations/supabase/client";
import { useQueryClient } from "@tanstack/react-query";

export interface PipelineEvent {
  id: string;
  timestamp: Date;
  type: "vitals_insert" | "risk_update" | "tripwire_fired" | "tripwire_cleared" | "patient_admitted" | "patient_discharged" | "labs_insert" | "critical_override";
  table: string;
  patientId?: string;
  patientName?: string;
  detail: string;
  severity: "info" | "warning" | "critical";
}

interface HospitalContextType {
  hospitalId: string | null;
  hospitalName: string | null;
  connectionStatus: ConnectionStatus;
  lastEventTime: Date | null;
  eventsReceived: number;
  pipelineEvents: PipelineEvent[];
  clearPipelineEvents: () => void;
}

const HospitalContext = createContext<HospitalContextType>({
  hospitalId: null,
  hospitalName: null,
  connectionStatus: "connecting",
  lastEventTime: null,
  eventsReceived: 0,
  pipelineEvents: [],
  clearPipelineEvents: () => {},
});

export function useHospital() {
  return useContext(HospitalContext);
}

const MAX_EVENTS = 50;

export function HospitalProvider({ children }: { children: ReactNode }) {
  const { profile } = useAuth();
  const { status, lastEventTime, eventsReceived, trackEvent } = useRealtimeConnection();
  const queryClient = useQueryClient();
  const [pipelineEvents, setPipelineEvents] = useState<PipelineEvent[]>([]);

  const hospitalId = profile?.hospital_id ?? null;
  const hospitalName = profile?.hospital_name ?? null;

  const addEvent = useCallback((event: Omit<PipelineEvent, "id" | "timestamp">) => {
    const newEvent: PipelineEvent = {
      ...event,
      id: crypto.randomUUID(),
      timestamp: new Date(),
    };
    setPipelineEvents((prev) => [newEvent, ...prev].slice(0, MAX_EVENTS));
    trackEvent(event.type);
  }, [trackEvent]);

  const clearPipelineEvents = useCallback(() => {
    setPipelineEvents([]);
  }, []);

  // Bidirectional: Subscribe to ALL hospital-scoped tables for shared state
  useEffect(() => {
    if (!hospitalId) return;

    const channel = supabase
      .channel(`hospital-${hospitalId}-global`)
      .on("postgres_changes", {
        event: "INSERT",
        schema: "public",
        table: "vitals",
      }, (payload) => {
        addEvent({
          type: "vitals_insert",
          table: "vitals",
          patientId: (payload.new as any).patient_id,
          detail: `Vitals recorded${(payload.new as any).is_manual_entry ? " (manual HITL entry)" : " (automated)"}`,
          severity: "info",
        });
        queryClient.invalidateQueries({ queryKey: ["vitals"] });
      })
      .on("postgres_changes", {
        event: "INSERT",
        schema: "public",
        table: "risk_assessments",
      }, (payload) => {
        const tier = (payload.new as any).tier;
        const score = (payload.new as any).quantum_risk_score;
        addEvent({
          type: "risk_update",
          table: "risk_assessments",
          patientId: (payload.new as any).patient_id,
          detail: `Risk score ${Number(score).toFixed(2)} → Tier ${tier}`,
          severity: tier === "CRITICAL" ? "critical" : tier === "AMBER" ? "warning" : "info",
        });
        queryClient.invalidateQueries({ queryKey: ["risk_assessments"] });
        queryClient.invalidateQueries({ queryKey: ["latest_risks"] });
      })
      .on("postgres_changes", {
        event: "*",
        schema: "public",
        table: "tripwire_alerts",
      }, (payload) => {
        const isActive = (payload.new as any)?.is_active;
        addEvent({
          type: isActive ? "tripwire_fired" : "tripwire_cleared",
          table: "tripwire_alerts",
          patientId: (payload.new as any)?.patient_id,
          detail: isActive
            ? `Tripwire fired: ${(payload.new as any)?.metric} — ${(payload.new as any)?.threshold_breached}`
            : `Tripwire cleared: ${(payload.new as any)?.metric ?? "unknown"}`,
          severity: isActive ? "warning" : "info",
        });
        queryClient.invalidateQueries({ queryKey: ["tripwire_alerts"] });
        queryClient.invalidateQueries({ queryKey: ["all_tripwires"] });
      })
      .on("postgres_changes", {
        event: "*",
        schema: "public",
        table: "patients",
      }, (payload) => {
        const newStatus = (payload.new as any)?.status;
        const name = (payload.new as any)?.name ?? "Unknown";
        if (payload.eventType === "INSERT") {
          addEvent({
            type: "patient_admitted",
            table: "patients",
            patientId: (payload.new as any)?.id,
            patientName: name,
            detail: `Patient "${name}" admitted to Bed ${(payload.new as any)?.bed_number}`,
            severity: "info",
          });
        } else if (payload.eventType === "UPDATE" && newStatus === "discharged") {
          addEvent({
            type: "patient_discharged",
            table: "patients",
            patientId: (payload.new as any)?.id,
            patientName: name,
            detail: `Patient "${name}" discharged`,
            severity: "info",
          });
        }
        queryClient.invalidateQueries({ queryKey: ["patients"] });
      })
      .on("postgres_changes", {
        event: "INSERT",
        schema: "public",
        table: "labs",
      }, (payload) => {
        addEvent({
          type: "labs_insert",
          table: "labs",
          patientId: (payload.new as any).patient_id,
          detail: "Lab results recorded",
          severity: "info",
        });
        queryClient.invalidateQueries({ queryKey: ["labs"] });
      })
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [hospitalId, queryClient, addEvent]);

  return (
    <HospitalContext.Provider
      value={{
        hospitalId,
        hospitalName,
        connectionStatus: status,
        lastEventTime,
        eventsReceived,
        pipelineEvents,
        clearPipelineEvents,
      }}
    >
      {children}
    </HospitalContext.Provider>
  );
}
