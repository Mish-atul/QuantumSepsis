import { useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogTrigger } from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { LogOut, AlertTriangle } from "lucide-react";

interface DischargePatientDialogProps {
  patientId: string;
  patientName: string;
}

export function DischargePatientDialog({ patientId, patientName }: DischargePatientDialogProps) {
  const { profile } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // Only attending and admin can discharge
  if (!profile || !["attending", "admin"].includes(profile.role)) return null;

  const handleDischarge = async () => {
    setSubmitting(true);
    try {
      const { error } = await supabase
        .from("patients")
        .update({ status: "discharged" })
        .eq("id", patientId);
      if (error) throw error;

      toast({ title: "Patient discharged", description: `${patientName} has been discharged.` });
      queryClient.invalidateQueries({ queryKey: ["patients"] });
      setOpen(false);
      navigate("/dashboard");
    } catch (error: any) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="font-mono text-xs gap-2 border-destructive/30 text-destructive hover:bg-destructive/10 hover:text-destructive">
          <LogOut className="h-4 w-4" />
          Discharge
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="font-mono text-base flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-tier-amber" />
            Confirm Discharge
          </DialogTitle>
          <DialogDescription className="font-mono text-xs">
            Are you sure you want to discharge <strong className="text-foreground">{patientName}</strong>? 
            This will remove them from the active ward overview.
          </DialogDescription>
        </DialogHeader>
        <div className="flex gap-3 justify-end mt-2">
          <Button variant="ghost" className="font-mono text-xs" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            className="font-mono text-xs"
            onClick={handleDischarge}
            disabled={submitting}
          >
            {submitting ? "Discharging..." : "Confirm Discharge"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
