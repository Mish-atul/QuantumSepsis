import { useState } from "react";
import { useAuth } from "@/hooks/useAuth";
import { Navigate } from "react-router-dom";
import { GlobalNav } from "@/components/layout/GlobalNav";
import { supabase } from "@/integrations/supabase/client";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import { UserPlus, Shield, ShieldAlert } from "lucide-react";
import type { StaffRole } from "@/hooks/useAuth";

const ROLE_BADGE: Record<string, string> = {
  attending: "bg-primary/20 text-primary border-primary/30",
  nurse: "bg-tier-watch/20 text-tier-watch border-tier-watch/30",
  admin: "bg-purple-500/20 text-purple-400 border-purple-500/30",
};

export default function Admin() {
  const { profile, loading } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newRole, setNewRole] = useState<StaffRole>("nurse");
  const [newEmployeeId, setNewEmployeeId] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const { data: staffList, isLoading } = useQuery({
    queryKey: ["admin_staff", profile?.hospital_id],
    enabled: !!profile?.hospital_id && profile?.role === "admin",
    queryFn: async () => {
      const { data, error } = await supabase
        .from("profiles")
        .select("*")
        .eq("hospital_id", profile!.hospital_id!)
        .order("full_name");
      if (error) throw error;
      return data;
    },
  });

  if (loading) return null;
  if (!profile || profile.role !== "admin") {
    return <Navigate to="/dashboard" replace />;
  }

  const handleAddStaff = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      // Create auth user (simulated — in production this would use admin API)
      const { data: authData, error: authError } = await supabase.auth.signUp({
        email: newEmail,
        password: "TempPass123!", // Demo only
        options: { data: { full_name: newName } },
      });
      if (authError) throw authError;

      if (authData.user) {
        // Update the auto-created profile with hospital info
        const { error: profileError } = await supabase
          .from("profiles")
          .update({
            hospital_id: profile.hospital_id,
            role: newRole,
            employee_id: newEmployeeId || null,
            full_name: newName,
          })
          .eq("user_id", authData.user.id);
        if (profileError) throw profileError;
      }

      toast({ title: "Staff added", description: `${newName} has been invited as ${newRole}.` });
      queryClient.invalidateQueries({ queryKey: ["admin_staff"] });
      setOpen(false);
      setNewName("");
      setNewEmail("");
      setNewRole("nurse");
      setNewEmployeeId("");
    } catch (error: any) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <GlobalNav />
      <main className="p-6 max-w-[1200px] mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-mono font-bold text-foreground">Staff Management</h1>
            <p className="text-xs font-mono text-muted-foreground mt-1">
              {profile.hospital_name} · {staffList?.length ?? 0} staff members
            </p>
          </div>
          <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
              <Button className="font-mono text-xs gap-2">
                <UserPlus className="h-4 w-4" />
                Add Staff
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle className="font-mono text-base">Add Staff Member</DialogTitle>
              </DialogHeader>
              <form onSubmit={handleAddStaff} className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs font-mono">Full Name</Label>
                  <Input
                    placeholder="Dr. Sharma"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    className="font-mono text-sm"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-xs font-mono">Email</Label>
                  <Input
                    type="email"
                    placeholder="staff@aiims.edu"
                    value={newEmail}
                    onChange={(e) => setNewEmail(e.target.value)}
                    className="font-mono text-sm"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-xs font-mono">Role</Label>
                  <Select value={newRole} onValueChange={(v) => setNewRole(v as StaffRole)}>
                    <SelectTrigger className="font-mono text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="nurse">Nurse</SelectItem>
                      <SelectItem value="attending">Attending</SelectItem>
                      <SelectItem value="admin">Admin</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs font-mono">Employee ID</Label>
                  <Input
                    placeholder="EMP-001"
                    value={newEmployeeId}
                    onChange={(e) => setNewEmployeeId(e.target.value)}
                    className="font-mono text-sm"
                  />
                </div>
                <Button type="submit" className="w-full font-mono text-sm" disabled={submitting}>
                  {submitting ? "Adding..." : "Add Staff Member"}
                </Button>
              </form>
            </DialogContent>
          </Dialog>
        </div>

        <div className="rounded-lg border border-border bg-card overflow-hidden">
          {isLoading ? (
            <div className="p-8 text-center text-muted-foreground font-mono text-sm">Loading staff...</div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="font-mono text-xs">Name</TableHead>
                  <TableHead className="font-mono text-xs">Role</TableHead>
                  <TableHead className="font-mono text-xs">Department</TableHead>
                  <TableHead className="font-mono text-xs">Employee ID</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {staffList?.map((s) => (
                  <TableRow key={s.id}>
                    <TableCell className="font-mono text-sm">
                      <div className="flex items-center gap-2">
                        {s.role === "admin" ? (
                          <ShieldAlert className="h-4 w-4 text-purple-400" />
                        ) : (
                          <Shield className="h-4 w-4 text-muted-foreground" />
                        )}
                        {s.full_name || "—"}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className={`font-mono text-[10px] ${ROLE_BADGE[s.role] || ""}`}>
                        {s.role}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {s.department || "ICU"}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {s.employee_id || "—"}
                    </TableCell>
                  </TableRow>
                ))}
                {(!staffList || staffList.length === 0) && (
                  <TableRow>
                    <TableCell colSpan={4} className="text-center text-muted-foreground font-mono text-sm py-8">
                      No staff members found
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </div>
      </main>
    </div>
  );
}
