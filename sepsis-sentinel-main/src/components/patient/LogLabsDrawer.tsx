import { useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { useToast } from "@/hooks/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { FlaskConical } from "lucide-react";

interface LogLabsDrawerProps {
  patientId: string;
}

export function LogLabsDrawer({ patientId }: LogLabsDrawerProps) {
  const { profile } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const [lactate, setLactate] = useState("");
  const [wbc, setWbc] = useState("");
  const [procalcitonin, setProcalcitonin] = useState("");
  const [creatinine, setCreatinine] = useState("");
  const [bilirubin, setBilirubin] = useState("");
  const [platelets, setPlatelets] = useState("");

  if (!profile || !["attending", "nurse"].includes(profile.role)) return null;

  const resetForm = () => {
    setLactate(""); setWbc(""); setProcalcitonin("");
    setCreatinine(""); setBilirubin(""); setPlatelets("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      const { error } = await supabase.from("labs").insert({
        patient_id: patientId,
        hospital_id: profile.hospital_id,
        lactate: lactate ? parseFloat(lactate) : null,
        wbc: wbc ? parseFloat(wbc) : null,
        procalcitonin: procalcitonin ? parseFloat(procalcitonin) : null,
        creatinine: creatinine ? parseFloat(creatinine) : null,
        bilirubin: bilirubin ? parseFloat(bilirubin) : null,
        platelets: platelets ? parseFloat(platelets) : null,
      });
      if (error) throw error;

      toast({ title: "Labs recorded", description: "Lab results saved successfully." });
      queryClient.invalidateQueries({ queryKey: ["labs", patientId] });
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
          <FlaskConical className="h-4 w-4" />
          Log Labs
        </Button>
      </SheetTrigger>
      <SheetContent className="w-[400px] sm:w-[450px] overflow-y-auto">
        <SheetHeader>
          <SheetTitle className="font-mono text-base">Log Lab Results</SheetTitle>
        </SheetHeader>
        <form onSubmit={handleSubmit} className="space-y-4 mt-6">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label className="text-xs font-mono">Lactate (mmol/L)</Label>
              <Input
                type="number" step="0.1" min="0" max="30"
                placeholder="1.2"
                value={lactate} onChange={(e) => setLactate(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs font-mono">WBC (×10³/µL)</Label>
              <Input
                type="number" step="0.1" min="0" max="100"
                placeholder="8.5"
                value={wbc} onChange={(e) => setWbc(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label className="text-xs font-mono">Procalcitonin (ng/mL)</Label>
              <Input
                type="number" step="0.01" min="0" max="200"
                placeholder="0.25"
                value={procalcitonin} onChange={(e) => setProcalcitonin(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs font-mono">Creatinine (mg/dL)</Label>
              <Input
                type="number" step="0.1" min="0" max="30"
                placeholder="0.9"
                value={creatinine} onChange={(e) => setCreatinine(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label className="text-xs font-mono">Bilirubin (mg/dL)</Label>
              <Input
                type="number" step="0.1" min="0" max="50"
                placeholder="0.8"
                value={bilirubin} onChange={(e) => setBilirubin(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs font-mono">Platelets (×10³/µL)</Label>
              <Input
                type="number" step="1" min="0" max="1000"
                placeholder="250"
                value={platelets} onChange={(e) => setPlatelets(e.target.value)}
                className="font-mono text-sm"
              />
            </div>
          </div>

          <Button type="submit" className="w-full font-mono text-sm" disabled={submitting}>
            {submitting ? "Saving..." : "Submit Lab Results"}
          </Button>
        </form>
      </SheetContent>
    </Sheet>
  );
}
