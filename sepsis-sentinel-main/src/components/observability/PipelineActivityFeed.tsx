import { useState } from "react";
import { useHospital, type PipelineEvent } from "@/contexts/HospitalContext";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import {
  Activity, AlertTriangle, Brain, Heart, FlaskConical,
  UserPlus, LogOut, Zap, Eye, Trash2
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";

const EVENT_ICONS: Record<PipelineEvent["type"], React.ReactNode> = {
  vitals_insert: <Heart className="h-3.5 w-3.5" />,
  risk_update: <Brain className="h-3.5 w-3.5" />,
  tripwire_fired: <Zap className="h-3.5 w-3.5" />,
  tripwire_cleared: <AlertTriangle className="h-3.5 w-3.5" />,
  patient_admitted: <UserPlus className="h-3.5 w-3.5" />,
  patient_discharged: <LogOut className="h-3.5 w-3.5" />,
  labs_insert: <FlaskConical className="h-3.5 w-3.5" />,
  critical_override: <Zap className="h-3.5 w-3.5" />,
};

const EVENT_LABELS: Record<PipelineEvent["type"], string> = {
  vitals_insert: "Layer 1 — Vitals",
  risk_update: "Layer 3 — Quantum Score",
  tripwire_fired: "Layer 4b — Tripwire",
  tripwire_cleared: "Layer 4b — Cleared",
  patient_admitted: "Admission",
  patient_discharged: "Discharge",
  labs_insert: "Labs",
  critical_override: "Layer 5 — Override",
};

const SEVERITY_STYLES = {
  info: "border-border bg-card",
  warning: "border-tier-amber/40 bg-tier-amber/5",
  critical: "border-tier-critical/40 bg-tier-critical/5 animate-flash-critical",
};

const SEVERITY_TEXT = {
  info: "text-muted-foreground",
  warning: "text-tier-amber",
  critical: "text-tier-critical",
};

function EventItem({ event }: { event: PipelineEvent }) {
  return (
    <div className={cn("flex items-start gap-3 p-3 rounded-lg border transition-all", SEVERITY_STYLES[event.severity])}>
      <div className={cn("mt-0.5", SEVERITY_TEXT[event.severity])}>
        {EVENT_ICONS[event.type]}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={cn("text-[10px] font-mono font-bold uppercase tracking-wider", SEVERITY_TEXT[event.severity])}>
            {EVENT_LABELS[event.type]}
          </span>
          <span className="text-[9px] font-mono text-muted-foreground ml-auto shrink-0">
            {formatDistanceToNow(event.timestamp, { addSuffix: true })}
          </span>
        </div>
        <p className="text-xs font-mono text-foreground/80 mt-0.5 truncate">
          {event.detail}
        </p>
        {event.patientName && (
          <p className="text-[10px] font-mono text-muted-foreground mt-0.5">
            Patient: {event.patientName}
          </p>
        )}
      </div>
    </div>
  );
}

/** Observability: Inline compact feed for ward overview */
export function PipelineActivityCompact() {
  const { pipelineEvents, connectionStatus } = useHospital();
  const recentEvents = pipelineEvents.slice(0, 3);

  if (recentEvents.length === 0 && connectionStatus === "connected") {
    return (
      <div className="flex items-center gap-2 text-[10px] font-mono text-muted-foreground px-2">
        <Activity className="h-3 w-3 animate-pulse" />
        Awaiting pipeline data...
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {recentEvents.map((event) => (
        <EventItem key={event.id} event={event} />
      ))}
    </div>
  );
}

/** Observability: Full pipeline activity drawer */
export function PipelineActivityDrawer() {
  const { pipelineEvents, clearPipelineEvents, eventsReceived } = useHospital();

  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="sm" className="font-mono text-xs gap-2 relative">
          <Eye className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Pipeline</span>
          {pipelineEvents.length > 0 && (
            <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-primary text-primary-foreground text-[9px] font-bold flex items-center justify-center">
              {pipelineEvents.length}
            </span>
          )}
        </Button>
      </SheetTrigger>
      <SheetContent className="w-[420px] sm:w-[480px]">
        <SheetHeader>
          <div className="flex items-center justify-between">
            <SheetTitle className="font-mono text-base">Pipeline Activity</SheetTitle>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-muted-foreground">
                {eventsReceived} total
              </span>
              {pipelineEvents.length > 0 && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6"
                  onClick={clearPipelineEvents}
                  title="Clear events"
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              )}
            </div>
          </div>
          <p className="text-[10px] font-mono text-muted-foreground">
            Real-time observability into agent pipeline decisions
          </p>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-120px)] mt-4">
          <div className="space-y-2 pr-2">
            {pipelineEvents.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <Activity className="h-8 w-8 mb-3 opacity-30" />
                <p className="text-xs font-mono">No pipeline events yet</p>
                <p className="text-[10px] font-mono mt-1 opacity-60">Events will appear as the backend processes data</p>
              </div>
            ) : (
              pipelineEvents.map((event) => (
                <EventItem key={event.id} event={event} />
              ))
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
