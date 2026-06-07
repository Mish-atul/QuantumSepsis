import { useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { Mail, Lock, User, Building2, BadgeCheck } from "lucide-react";

interface SignUpFormProps {
  onSwitchToLogin: () => void;
}

export default function SignUpForm({ onSwitchToLogin }: SignUpFormProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [department, setDepartment] = useState("ICU");
  const [employeeId, setEmployeeId] = useState("");
  const [role, setRole] = useState("nurse");
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!fullName.trim() || !employeeId.trim()) {
      toast({ title: "Missing fields", description: "Please fill all required fields.", variant: "destructive" });
      return;
    }
    setLoading(true);
    try {
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: { full_name: fullName, department, employee_id: employeeId, role },
          emailRedirectTo: window.location.origin,
        },
      });
      if (error) throw error;
      toast({
        title: "Account created",
        description: "Check your email to verify your account.",
      });
    } catch (error: any) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSignUp} className="space-y-3">
      <div className="space-y-2">
        <Label htmlFor="name" className="text-xs font-mono">Full Name *</Label>
        <div className="relative">
          <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="name"
            placeholder="Dr. Sharma"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            className="pl-10 font-mono text-sm"
            required
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="signup-email" className="text-xs font-mono">Email *</Label>
        <div className="relative">
          <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="signup-email"
            type="email"
            placeholder="staff@hospital.org"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="pl-10 font-mono text-sm"
            required
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="signup-password" className="text-xs font-mono">Password *</Label>
        <div className="relative">
          <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="signup-password"
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="pl-10 font-mono text-sm"
            minLength={6}
            required
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="employee-id" className="text-xs font-mono">Employee ID *</Label>
        <div className="relative">
          <BadgeCheck className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="employee-id"
            placeholder="EMP-001"
            value={employeeId}
            onChange={(e) => setEmployeeId(e.target.value)}
            className="pl-10 font-mono text-sm"
            required
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-2">
          <Label className="text-xs font-mono">Department</Label>
          <Select value={department} onValueChange={setDepartment}>
            <SelectTrigger className="font-mono text-sm">
              <Building2 className="h-4 w-4 mr-2 text-muted-foreground" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ICU">ICU</SelectItem>
              <SelectItem value="Emergency">Emergency</SelectItem>
              <SelectItem value="Surgery">Surgery</SelectItem>
              <SelectItem value="Internal Medicine">Internal Med</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label className="text-xs font-mono">Role</Label>
          <Select value={role} onValueChange={setRole}>
            <SelectTrigger className="font-mono text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="nurse">Nurse</SelectItem>
              <SelectItem value="attending">Attending</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <Button type="submit" className="w-full font-mono text-sm" disabled={loading}>
        {loading ? "Creating account..." : "Create Account"}
      </Button>

      <div className="text-center">
        <button
          type="button"
          onClick={onSwitchToLogin}
          className="text-xs font-mono text-muted-foreground hover:text-foreground transition-colors"
        >
          Already have an account? Sign in
        </button>
      </div>
    </form>
  );
}
