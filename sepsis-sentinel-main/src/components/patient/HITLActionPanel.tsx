import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/useAuth";
import { useToast } from "@/hooks/use-toast";
import type { RiskTier } from "@/types/database";
import { Clock, Syringe, Phone, CheckCircle, XCircle, ShieldAlert, UserCheck } from "lucide-react";

interface HITLActionPanelProps {
  tier: RiskTier;
  isCriticalOverride: boolean;
  patientName?: string;
}

interface HITLAction {
  icon: React.ReactNode;
  label: string;
  description: string;
  requiresApproval: boolean;
  approvalRole: string[];
  approvalPrompt: string;
  steps: string[];
}

const ACTIONS: Record<RiskTier, HITLAction> = {
  WATCH: {
    icon: <Clock className="h-4 w-4" />,
    label: "MONITORING",
    description: "Standard monitoring — next cycle in 15 min",
    requiresApproval: false,
    approvalRole: [],
    approvalPrompt: "",
    steps: [
      "Continue 15-min vitals cycle",
      "Monitor for tripwire threshold breaches",
      "Review next risk assessment on arrival",
    ],
  },
  AMBER: {
    icon: <Syringe className="h-4 w-4" />,
    label: "CONCURRENT ORDERS RECOMMENDED",
    description: "The pipeline recommends concurrent diagnostic orders. Attending approval is required before execution.",
    requiresApproval: true,
    approvalRole: ["attending", "admin"],
    approvalPrompt: "Approve concurrent orders? This will initiate Lactate, PCT, Blood Culture, and Fluid Challenge.",
    steps: [
      "🔬 Order: Lactate level",
      "🔬 Order: Procalcitonin (PCT)",
      "🧫 Order: Blood culture (2 sets)",
      "💧 Initiate: IV fluid challenge (30 mL/kg)",
      "⏱ Reassess: 30-minute follow-up vitals",
    ],
  },
  CRITICAL: {
    icon: <Phone className="h-4 w-4" />,
    label: "SEPSIS BUNDLE — IMMEDIATE ACTION",
    description: "Critical threshold reached. Attending must acknowledge and initiate the sepsis bundle protocol.",
    requiresApproval: true,
    approvalRole: ["attending", "admin"],
    approvalPrompt: "Acknowledge CRITICAL alert and initiate sepsis bundle? This pages the attending and starts the 1-hour antibiotic window.",
    steps: [
      "📟 Page: Attending physician",
      "💉 Order: Broad-spectrum antibiotics (within 1 hr)",
      "💧 Initiate: Aggressive fluid resuscitation",
      "🔬 Order: Repeat lactate at 3 hrs",
      "🏥 Alert: ICU team for rapid response",
      "📋 Document: Time-zero for sepsis bundle",
    ],
  },
};

export function HITLActionPanel({ tier, isCriticalOverride, patientName }: HITLActionPanelProps) {
  const { profile } = useAuth();
  const { toast } = useToast();
  const [approvalDialog, setApprovalDialog] = useState(false);
  const [acknowledged, setAcknowledged] = useState(false);
  const [acknowledgedAt, setAcknowledgedAt] = useState<Date | null>(null);

  const effectiveTier = isCriticalOverride ? "CRITICAL" : tier;
  const action = ACTIONS[effectiveTier];
  const canApprove = profile && action.approvalRole.includes(profile.role);

  const borderClass = {
    CRITICAL: "border-tier-critical/40 bg-tier-critical/5",
    AMBER: "border-tier-amber/40 bg-tier-amber/5",
    WATCH: "border-border",
  };

  const textClass = {
    CRITICAL: "text-tier-critical",
    AMBER: "text-tier-amber",
    WATCH: "text-tier-watch",
  };

  const handleApprove = () => {
    setAcknowledged(true);
    setAcknowledgedAt(new Date());
    setApprovalDialog(false);
    toast({
      title: `${effectiveTier} Protocol Acknowledged`,
      description: `${profile?.full_name ?? "Staff"} approved ${effectiveTier.toLowerCase()} actions for ${patientName ?? "patient"}.`,
    });
  };

  return (
    <>
      <Card className={cn(borderClass[effectiveTier])}>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-mono uppercase tracking-wider text-muted-foreground">
              Orchestrator Actions — Layer 5
            </CardTitle>
            {action.requiresApproval && (
              <div className="flex items-center gap-1.5">
                <UserCheck className="h-3 w-3 text-muted-foreground" />
                <span className="text-[10px] font-mono text-muted-foreground uppercase">HITL Required</span>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-start gap-3">
            <div className={cn("p-2 rounded-lg bg-secondary", textClass[effectiveTier])}>
              {action.icon}
            </div>
            <div className="flex-1">
              <p className={cn("text-xs font-mono font-bold uppercase tracking-wider", textClass[effectiveTier])}>
                {action.label}
              </p>
              <p className="text-xs font-mono text-muted-foreground mt-1">
                {action.description}
              </p>
              {isCriticalOverride && effectiveTier === "CRITICAL" && (
                <p className="text-[10px] font-mono text-tier-critical mt-2 font-bold">
                  ↑ Escalated via tripwire override (2+ active triggers)
                </p>
              )}
            </div>
          </div>

          {/* Observability: Show pipeline recommended steps */}
          <div className="border-t border-border pt-3">
            <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-2">
              Pipeline Recommended Steps
            </p>
            <div className="space-y-1.5">
              {action.steps.map((step, i) => (
                <div
                  key={i}
                  className={cn(
                    "flex items-center gap-2 text-xs font-mono py-1 px-2 rounded-md",
                    acknowledged ? "text-foreground/60 bg-tier-watch/5" : "text-foreground/80"
                  )}
                >
                  {acknowledged ? (
                    <CheckCircle className="h-3 w-3 text-tier-watch shrink-0" />
                  ) : (
                    <span className="h-3 w-3 flex items-center justify-center text-[9px] text-muted-foreground shrink-0">{i + 1}</span>
                  )}
                  <span>{step}</span>
                </div>
              ))}
            </div>
          </div>

          {/* HITL: Approval button */}
          {action.requiresApproval && (
            <div className="border-t border-border pt-3">
              {acknowledged ? (
                <div className="flex items-center gap-2 p-3 rounded-lg border border-tier-watch/30 bg-tier-watch/5">
                  <CheckCircle className="h-4 w-4 text-tier-watch" />
                  <div>
                    <p className="text-xs font-mono font-bold text-tier-watch">
                      Acknowledged by {profile?.full_name ?? "Staff"}
                    </p>
                    <p className="text-[10px] font-mono text-muted-foreground">
                      {acknowledgedAt?.toLocaleTimeString()} — Protocol in progress
                    </p>
                  </div>
                </div>
              ) : canApprove ? (
                <Button
                  onClick={() => setApprovalDialog(true)}
                  className={cn(
                    "w-full font-mono text-xs gap-2 glow-cta",
                    effectiveTier === "CRITICAL"
                      ? "bg-tier-critical hover:bg-tier-critical/90 text-white"
                      : "bg-tier-amber hover:bg-tier-amber/90 text-black"
                  )}
                >
                  <ShieldAlert className="h-4 w-4" />
                  Acknowledge & Approve Protocol
                </Button>
              ) : (
                <div className="flex items-center gap-2 p-3 rounded-lg border border-border bg-secondary/50">
                  <ShieldAlert className="h-4 w-4 text-muted-foreground" />
                  <p className="text-xs font-mono text-muted-foreground">
                    Awaiting attending approval — only attending/admin can authorize
                  </p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* HITL: Confirmation dialog */}
      <Dialog open={approvalDialog} onOpenChange={setApprovalDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className={cn("font-mono text-base flex items-center gap-2", textClass[effectiveTier])}>
              <ShieldAlert className="h-5 w-5" />
              Confirm {effectiveTier} Protocol
            </DialogTitle>
            <DialogDescription className="font-mono text-xs">
              {action.approvalPrompt}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2 mt-2">
            {action.steps.map((step, i) => (
              <div key={i} className="flex items-center gap-2 text-xs font-mono text-foreground/80">
                <span className="text-[10px] text-muted-foreground">{i + 1}.</span>
                {step}
              </div>
            ))}
          </div>
          <div className="flex items-center gap-2 mt-2 p-2 rounded-lg border border-tier-amber/30 bg-tier-amber/5">
            <UserCheck className="h-4 w-4 text-tier-amber shrink-0" />
            <p className="text-[10px] font-mono text-tier-amber">
              This action will be logged as approved by {profile?.full_name ?? "you"} ({profile?.role ?? "staff"})
            </p>
          </div>
          <div className="flex gap-3 justify-end mt-2">
            <Button variant="ghost" className="font-mono text-xs gap-1" onClick={() => setApprovalDialog(false)}>
              <XCircle className="h-3.5 w-3.5" />
              Cancel
            </Button>
            <Button
              className={cn(
                "font-mono text-xs gap-1",
                effectiveTier === "CRITICAL"
                  ? "bg-tier-critical hover:bg-tier-critical/90 text-white"
                  : "bg-tier-amber hover:bg-tier-amber/90 text-black"
              )}
              onClick={handleApprove}
            >
              <CheckCircle className="h-3.5 w-3.5" />
              Approve Protocol
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
