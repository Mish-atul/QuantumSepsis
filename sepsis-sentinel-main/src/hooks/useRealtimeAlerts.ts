/**
 * Real-time alerts WebSocket hook implementing the contract in
 * BACKEND_INTEGRATION_README.md §4.
 *
 * - Authenticates with the current Supabase JWT
 * - Subscribes to the supplied ICU unit channel
 * - Plays a sound on CRITICAL `new_alert`
 * - Mirrors every event into HospitalContext (observability/pipeline feed)
 *   and invalidates relevant React Query caches
 *
 * Only opens a connection when `VITE_USE_QS_API=true`. Otherwise it returns
 * an idle state so the existing Lovable Cloud realtime path remains the
 * single source of truth.
 */

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import {
  QS_API_ENABLED,
  QS_WS_URL,
  type AlertLevel,
  type QSAlert,
  type VitalSigns,
} from "@/lib/api/quantumSepsis";

type WsStatus = "idle" | "connecting" | "open" | "closed" | "error";

export interface NewAlertMessage {
  type: "new_alert";
  timestamp: string;
  data: {
    alert_id: string;
    patient_id: string;
    level: AlertLevel;
    risk_score: number;
    bed_number: string;
    icu_unit: string;
  };
}

export interface AlertLevelChangeMessage {
  type: "alert_level_change";
  timestamp: string;
  data: {
    alert_id: string;
    patient_id: string;
    old_level: AlertLevel;
    new_level: AlertLevel;
    risk_score: number;
  };
}

export interface VitalsUpdateMessage {
  type: "vitals_update";
  timestamp: string;
  data: { patient_id: string; vitals: Partial<VitalSigns> };
}

export interface SystemStatusMessage {
  type: "system_status";
  timestamp: string;
  data: { status: string; active_patients: number; active_alerts: number };
}

export type QsRealtimeMessage =
  | NewAlertMessage
  | AlertLevelChangeMessage
  | VitalsUpdateMessage
  | SystemStatusMessage;

export interface UseRealtimeAlertsOptions {
  icu_unit: string;
  /** Disable the audible CRITICAL alert chime. */
  muteSound?: boolean;
}

export function useRealtimeAlerts({ icu_unit, muteSound }: UseRealtimeAlertsOptions) {
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<WsStatus>(QS_API_ENABLED ? "connecting" : "idle");
  const [alerts, setAlerts] = useState<QSAlert[]>([]);
  const [lastMessage, setLastMessage] = useState<QsRealtimeMessage | null>(null);

  useEffect(() => {
    if (!QS_API_ENABLED) return;

    let cancelled = false;
    let retry = 0;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    async function connect() {
      const { data } = await supabase.auth.getSession();
      const token = data.session?.access_token;
      if (cancelled) return;

      const ws = new WebSocket(QS_WS_URL);
      wsRef.current = ws;
      setStatus("connecting");

      ws.onopen = () => {
        retry = 0;
        setStatus("open");
        if (token) ws.send(JSON.stringify({ type: "auth", token }));
        ws.send(
          JSON.stringify({ type: "subscribe", channel: "icu_unit", icu_unit })
        );
      };

      ws.onmessage = (event) => {
        let msg: QsRealtimeMessage | null = null;
        try {
          msg = JSON.parse(event.data) as QsRealtimeMessage;
        } catch {
          return;
        }
        if (!msg || typeof msg !== "object") return;
        setLastMessage(msg);

        switch (msg.type) {
          case "new_alert": {
            setAlerts((prev) => [
              {
                ...(msg!.data as NewAlertMessage["data"]),
                stay_id: "",
                confidence: 0,
                triggered_at: msg!.timestamp,
                status: "active",
                red_team_override: false,
                active_tripwires: [],
              } as QSAlert,
              ...prev,
            ]);
            if (!muteSound && msg.data.level === "CRITICAL") {
              try {
                new Audio("/sounds/critical-alert.mp3").play().catch(() => {});
              } catch {
                /* no-op */
              }
            }
            queryClient.invalidateQueries({ queryKey: ["qs", "alerts"] });
            break;
          }
          case "alert_level_change": {
            setAlerts((prev) =>
              prev.map((a) =>
                a.alert_id === msg!.data.alert_id
                  ? { ...a, level: (msg as AlertLevelChangeMessage).data.new_level }
                  : a
              )
            );
            queryClient.invalidateQueries({ queryKey: ["qs", "alerts"] });
            queryClient.invalidateQueries({
              queryKey: ["qs", "patient", msg.data.patient_id],
            });
            break;
          }
          case "vitals_update": {
            queryClient.invalidateQueries({
              queryKey: ["qs", "vitals", msg.data.patient_id],
            });
            break;
          }
          case "system_status": {
            queryClient.invalidateQueries({ queryKey: ["qs", "dashboard"] });
            break;
          }
        }
      };

      ws.onerror = () => setStatus("error");

      ws.onclose = () => {
        setStatus("closed");
        if (cancelled) return;
        // Exponential backoff: 1s, 2s, 4s … capped at 30s.
        const delay = Math.min(30_000, 1_000 * 2 ** retry++);
        retryTimer = setTimeout(connect, delay);
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [icu_unit, muteSound, queryClient]);

  return { alerts, status, lastMessage };
}
