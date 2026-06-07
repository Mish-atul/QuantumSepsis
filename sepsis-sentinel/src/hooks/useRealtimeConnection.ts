import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import type { RealtimeChannel } from "@supabase/supabase-js";

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

interface RealtimeState {
  status: ConnectionStatus;
  lastEvent: string | null;
  lastEventTime: Date | null;
  eventsReceived: number;
  channels: string[];
}

/**
 * Bidirectional Communication: Tracks the persistent WebSocket connection status
 * and provides observability into realtime event flow.
 */
export function useRealtimeConnection() {
  const [state, setState] = useState<RealtimeState>({
    status: "connecting",
    lastEvent: null,
    lastEventTime: null,
    eventsReceived: 0,
    channels: [],
  });

  useEffect(() => {
    // Monitor the global realtime connection
    const channel = supabase
      .channel("connection-monitor")
      .on("presence", { event: "sync" }, () => {
        setState((prev) => ({ ...prev, status: "connected" }));
      })
      .subscribe((status) => {
        if (status === "SUBSCRIBED") {
          setState((prev) => ({ ...prev, status: "connected" }));
        } else if (status === "CLOSED") {
          setState((prev) => ({ ...prev, status: "disconnected" }));
        } else if (status === "CHANNEL_ERROR") {
          setState((prev) => ({ ...prev, status: "error" }));
        } else {
          setState((prev) => ({ ...prev, status: "connecting" }));
        }
      });

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  const trackEvent = useCallback((eventName: string) => {
    setState((prev) => ({
      ...prev,
      lastEvent: eventName,
      lastEventTime: new Date(),
      eventsReceived: prev.eventsReceived + 1,
    }));
  }, []);

  return { ...state, trackEvent };
}
