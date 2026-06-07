import { useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { UserPlus, User, Hash, BedDouble } from "lucide-react";

export function AdmitPatientModal() {
  const { profile } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [mrn, setMrn] = useState("");
  const [bedNumber, setBedNumber] = useState("");
  const [submitting, setSubmitting] = useState(false);

  // Only visible to attending and admin
  if (!profile || !["attending", "admin"].includes(profile.role)) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !mrn.trim() || !bedNumber.trim()) {
      toast({ title: "Missing fields", description: "All fields are required.", variant: "destructive" });
      return;
    }
    setSubmitting(true);
    try {
      const { error } = await supabase.from("patients").insert({
        name: name.trim(),
        mrn: mrn.trim(),
        bed_number: bedNumber.trim(),
        hospital_id: profile.hospital_id,
        status: "active",
      });
      if (error) throw error;

      toast({ title: "Patient admitted", description: `${name} assigned to Bed ${bedNumber}.` });
      queryClient.invalidateQueries({ queryKey: ["patients"] });
      setOpen(false);
      setName("");
      setMrn("");
      setBedNumber("");
    } catch (error: any) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="font-mono text-xs gap-2" size="sm">
          <UserPlus className="h-4 w-4" />
          Admit Patient
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="font-mono text-base">Admit New Patient</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label className="text-xs font-mono">Full Name *</Label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Patient full name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="pl-10 font-mono text-sm"
                required
              />
            </div>
          </div>
          <div className="space-y-2">
            <Label className="text-xs font-mono">MRN *</Label>
            <div className="relative">
              <Hash className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="MRN-001234"
                value={mrn}
                onChange={(e) => setMrn(e.target.value)}
                className="pl-10 font-mono text-sm"
                required
              />
            </div>
          </div>
          <div className="space-y-2">
            <Label className="text-xs font-mono">Bed Number *</Label>
            <div className="relative">
              <BedDouble className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="ICU-12"
                value={bedNumber}
                onChange={(e) => setBedNumber(e.target.value)}
                className="pl-10 font-mono text-sm"
                required
              />
            </div>
          </div>
          <Button type="submit" className="w-full font-mono text-sm" disabled={submitting}>
            {submitting ? "Admitting..." : "Admit Patient"}
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  );
}
