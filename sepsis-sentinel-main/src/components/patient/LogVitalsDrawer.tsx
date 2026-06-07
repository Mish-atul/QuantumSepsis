import { useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { useToast } from "@/hooks/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { ClipboardPlus, AlertTriangle } from "lucide-react";
import type { MentalStatus } from "@/types/database";

interface LogVitalsDrawerProps {
  patientId: string;
}

export function LogVitalsDrawer({ patientId }: LogVitalsDrawerProps) {
  const { profile } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const [hr, setHr] = useState("");
  const [bpSys, setBpSys] = useState("");
  const [bpDia, setBpDia] = useState("");
  const [temp, setTemp] = useState("");
  const [spo2, setSpo2] = useState("");
  const [rr, setRr] = useState("");
  const [mentalStatus, setMentalStatus] = useState<MentalStatus>("normal");
  const [flagAms, setFlagAms] = useState(false);

  // Only visible to attending and nurse
  if (!profile || !["attending", "nurse"].includes(profile.role)) return null;

  const resetForm = () => {
    setHr(""); setBpSys(""); setBpDia(""); setTemp("");
    setSpo2(""); setRr(""); setMentalStatus("normal"); setFlagAms(false);
  };

  // Compute MAP from sys/dia
  const computeMap = (sys: number, dia: number) => Math.round(dia + (sys - dia) / 3);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);

    const sys = parseFloat(bpSys);
    const dia = parseFloat(bpDia);
    const effectiveMental = flagAms && mentalStatus === "normal" ? "confused" : mentalStatus;

    try {
      const { error } = await supabase.from("vitals").insert({
        patient_id: patientId,
        hospital_id: profile.hospital_id,
        heart_rate: hr ? parseFloat(hr) : null,
        blood_pressure_sys: bpSys ? sys : null,
        blood_pressure_dia: bpDia ? dia : null,
        map: bpSys && bpDia ? computeMap(sys, dia) : null,
        temperature: temp ? parseFloat(temp) : null,
        spo2: spo2 ? parseFloat(spo2) : null,
        respiratory_rate: rr ? parseFloat(rr) : null,
        mental_status: effectiveMental,
        is_manual_entry: true,
      });
      if (error) throw error;

      toast({ title: "Vitals logged", description: "Manual assessment recorded successfully." });
      queryClient.invalidateQueries({ queryKey: ["vitals", patientId] });
      resetForm();
      setOpen(false);
    } catch (error: any) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="outline" className="font-mono text-xs gap-2" size="sm">
          <ClipboardPlus className="h-4 w-4" />
          Log Vitals
        </Button>
      </SheetTrigger>
      <SheetContent className="w-[400px] sm:w-[450px] overflow-y-auto">
        <SheetHeader>
          <SheetTitle className="font-mono text-base">Log Vitals & Assessment</SheetTitle>
        </SheetHeader>
        <form onSubmit={handleSubmit} className="space-y-4 mt-6">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label className="text-xs font-mono">Heart Rate (bpm)</Label>
              <Input
                type="number" step="1" min="20" max="300"
                placeholder="88"
                value={hr} onChange={(e) => setHr(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs font-mono">SpO₂ (%)</Label>
              <Input
                type="number" step="1" min="50" max="100"
                placeholder="97"
                value={spo2} onChange={(e) => setSpo2(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label className="text-xs font-mono">BP Systolic (mmHg)</Label>
              <Input
                type="number" step="1" min="40" max="300"
                placeholder="120"
                value={bpSys} onChange={(e) => setBpSys(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs font-mono">BP Diastolic (mmHg)</Label>
              <Input
                type="number" step="1" min="20" max="200"
                placeholder="80"
                value={bpDia} onChange={(e) => setBpDia(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label className="text-xs font-mono">Temperature (°C)</Label>
              <Input
                type="number" step="0.1" min="30" max="45"
                placeholder="37.0"
                value={temp} onChange={(e) => setTemp(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs font-mono">Resp. Rate (/min)</Label>
              <Input
                type="number" step="1" min="4" max="60"
                placeholder="16"
                value={rr} onChange={(e) => setRr(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
          </div>

          {/* Mental Status — Highly Visible */}
          <div className="space-y-3 p-3 rounded-md border border-border bg-secondary/50">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-tier-amber" />
              <Label className="text-xs font-mono font-bold text-foreground">Mental Status Assessment</Label>
            </div>
            <Select value={mentalStatus} onValueChange={(v) => setMentalStatus(v as MentalStatus)}>
              <SelectTrigger className="font-mono text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="normal">Normal</SelectItem>
                <SelectItem value="confused">Confused</SelectItem>
                <SelectItem value="agitated">Agitated</SelectItem>
                <SelectItem value="reduced_gcs">Reduced GCS</SelectItem>
              </SelectContent>
            </Select>
            <div className="flex items-center gap-2">
              <Checkbox
                id="flag-ams"
                checked={flagAms}
                onCheckedChange={(v) => setFlagAms(v === true)}
              />
              <Label htmlFor="flag-ams" className="text-xs font-mono text-tier-amber cursor-pointer">
                Flag as Altered Mental Status (AMS)
              </Label>
            </div>
            {(mentalStatus !== "normal" || flagAms) && (
              <div className="text-[10px] font-mono text-tier-critical bg-tier-critical/10 border border-tier-critical/30 rounded px-2 py-1">
                ⚠ This will trigger a primary tripwire alert
              </div>
            )}
          </div>

          <Button type="submit" className="w-full font-mono text-sm" disabled={submitting}>
            {submitting ? "Logging..." : "Submit Manual Assessment"}
          </Button>
        </form>
      </SheetContent>
    </Sheet>
  );
}
