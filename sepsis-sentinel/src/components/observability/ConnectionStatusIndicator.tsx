import { useHospital } from "@/contexts/HospitalContext";
import { Wifi, WifiOff, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export function ConnectionStatusIndicator() {
  const { connectionStatus, eventsReceived, lastEventTime } = useHospital();

  const config = {
    connected: { icon: Wifi, color: "text-tier-watch", pulse: false, label: "Live" },
    connecting: { icon: Loader2, color: "text-tier-amber", pulse: true, label: "Connecting" },
    disconnected: { icon: WifiOff, color: "text-muted-foreground", pulse: false, label: "Offline" },
    error: { icon: WifiOff, color: "text-vital-danger", pulse: true, label: "Error" },
  };

  const c = config[connectionStatus];
  const Icon = c.icon;

  const lastStr = lastEventTime
    ? `Last event: ${lastEventTime.toLocaleTimeString()}`
    : "No events yet";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className={cn("flex items-center gap-1.5 px-2 py-1 rounded-lg", c.color)}>
          <Icon className={cn("h-3 w-3", c.pulse && "animate-spin")} />
          <span className="text-[10px] font-mono font-medium hidden sm:inline">{c.label}</span>
          {connectionStatus === "connected" && (
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-tier-watch opacity-75" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-tier-watch" />
            </span>
          )}
        </div>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="font-mono text-xs">
        <p>WebSocket: {connectionStatus}</p>
        <p>{lastStr}</p>
        <p>Events received: {eventsReceived}</p>
      </TooltipContent>
    </Tooltip>
  );
}
